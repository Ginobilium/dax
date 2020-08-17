import collections
import typing
import abc

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool
from dax.util.artiq import is_kernel

__all__ = ['PmtMonitor', 'MultiPmtMonitor']


class _PmtMonitorBase(DaxClient, EnvExperiment, abc.ABC):
    """Base PMT monitor class."""

    APPLET_NAME: str
    """Name of the applet in the dashboard."""
    APPLET_GROUP: str
    """Group of the applet."""
    DEFAULT_DATASET: str
    """Default dataset for output."""

    COUNT_SCALES = collections.OrderedDict(GHz=GHz, MHz=MHz, kHz=kHz, Hz=Hz, mHz=mHz)
    """Scales that can be used for the Y-axis."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore
        assert isinstance(self.APPLET_NAME, str), 'Applet name must be of type str'
        assert isinstance(self.APPLET_GROUP, str), 'Applet group must be of type str'
        assert isinstance(self.DEFAULT_DATASET, str), 'Default dataset must be of type str'
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'

        # Obtain the detection interface
        detection = self.registry.find_interface(DetectionInterface)  # type: ignore[misc]
        # Obtain the PMT array
        self.pmt_array = detection.get_pmt_array()
        self.update_kernel_invariants('pmt_array')
        self.logger.debug(f'Found PMT array with {len(self.pmt_array)} channel(s)')

        # Get the scheduler and CCB tool
        self.scheduler = self.get_device('scheduler')
        self.ccb = get_ccb_tool(self)
        self.update_kernel_invariants('scheduler')

        # Standard arguments
        self.detection_window = self.get_argument('PMT detection window size',
                                                  NumberValue(default=100 * ms, unit='ms', min=0.0),
                                                  tooltip='Detection window duration')
        self.detection_delay = self.get_argument('PMT detection delay',
                                                 NumberValue(default=10 * ms, unit='ms', min=0.0),
                                                 tooltip='Delay between detection windows')
        self.count_scale_label = self.get_argument('PMT count scale',
                                                   EnumerationValue(list(self.COUNT_SCALES), default='kHz'),
                                                   tooltip='Scaling factor for the PMT counts graph')
        self.update_kernel_invariants('detection_window', 'detection_delay')

        # Add custom arguments here
        self._add_custom_arguments()

        # Dataset related arguments
        self.reset_data = self.get_argument('Reset data',
                                            BooleanValue(default=True),
                                            group='Dataset',
                                            tooltip='Clear old data at start')
        self.sliding_window = self.get_argument('Data window size',
                                                NumberValue(default=120 * s, unit='s', min=0, ndecimals=0, step=60),
                                                group='Dataset',
                                                tooltip='Data window size (use 0 for infinite window size)')
        self.dataset_key = self.get_argument('Dataset key',
                                             StringValue(default=self.DEFAULT_DATASET),
                                             group='Dataset',
                                             tooltip='Dataset key to which plotting data will be written')

        # Applet specific arguments
        self.create_applet = self.get_argument('Create applet',
                                               BooleanValue(default=True),
                                               group='Applet',
                                               tooltip='Call CCB create applet command at start')
        self.applet_update_delay = self.get_argument('Applet update delay',
                                                     NumberValue(default=0.1 * s, unit='s', min=0.0),
                                                     group='Applet',
                                                     tooltip='Delay between plot interface updates')
        self.applet_auto_close = self.get_argument('Close applet automatically',
                                                   BooleanValue(default=True),
                                                   group='Applet',
                                                   tooltip='Close applet when experiment is terminated')

    def _add_custom_arguments(self) -> None:
        """Add custom arguments."""
        pass

    @abc.abstractmethod
    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Create applet."""
        pass

    def prepare(self) -> None:
        if self.sliding_window > 0 and self.detection_window > 0.0:
            # Convert window size to dataset size
            self.sliding_window = int(self.sliding_window / self.detection_window)
            self.logger.debug(f'Window size set to {self.sliding_window}')

        # Pre-calculate Y-scalar
        self.y_scalar = 1.0 / self.detection_window / self.COUNT_SCALES[self.count_scale_label]

    def run(self) -> None:
        # NOTE: there is no dax_init() in this experiment!

        # Initial value is reset to an empty list or try to obtain the previous value defaulting to an empty list
        init_value = [] if self.reset_data else self.get_dataset(self.dataset_key, default=[], archive=False)
        self.logger.debug('Appending to previous data' if init_value else 'Starting with empty list')

        # Set the result datasets to the correct mode
        self.set_dataset(self.dataset_key, init_value, broadcast=True, archive=False)

        if self.create_applet:
            # Create the applet
            y_label = f'Counts per second ({self.count_scale_label})'
            self._create_applet(self.APPLET_NAME, self.dataset_key,
                                group=self.APPLET_GROUP, update_delay=self.applet_update_delay,
                                sliding_window=self.sliding_window, x_label='Sample', y_label=y_label)

        try:
            # Only stop when termination is requested
            while True:
                # Host setup
                self.host_setup()
                # Monitor
                self.monitor()

                # To pause, close communications and call the pause function
                self.core.comm.close()
                self.scheduler.pause()  # Can raise a TerminationRequested exception

        except TerminationRequested:
            # Experiment was terminated, gracefully end the experiment
            self.logger.debug('Terminated gracefully')

        finally:
            if self.applet_auto_close:
                # Disable the applet
                self.ccb.disable_applet(self.APPLET_NAME, self.APPLET_GROUP)

    @kernel
    def monitor(self):  # type: () -> None
        # Device setup
        self.device_setup()

        while True:
            # Check for pause condition and return if true
            if self.scheduler.check_pause():
                return

            # Guarantee slack
            self.core.break_realtime()

            # Insert delay
            delay(self.detection_delay)

            # Perform detection and store count
            self._count()

    @abc.abstractmethod
    def _count(self) -> None:
        """Perform detection, get counts, and store result."""
        pass

    """Customization functions"""

    def host_setup(self) -> None:
        """Preparation on the host, called once at entry and after a pause."""
        pass

    @kernel
    def device_setup(self):  # type: () -> None
        """Preparation on the core device, called once at entry and after a pause.

        Should at least reset the core.
        """
        # Reset the core
        self.core.reset()


@dax_client_factory
class PmtMonitor(_PmtMonitorBase):
    """PMT monitor utility to monitor a single PMT channel."""

    APPLET_NAME = 'pmt_monitor'
    APPLET_GROUP = 'dax'
    DEFAULT_DATASET = 'plot.dax.pmt_monitor_count'

    NUM_DIGITS_BIG_NUMBER: int = 5
    """Number of digits to display for the big number applet."""

    _PLOT_XY: str = 'Plot XY'
    """Key for plot XY applet type."""
    _BIG_NUMBER: str = 'Big number'
    """Key for big number applet type."""

    def _add_custom_arguments(self) -> None:
        # Get max for PMT channel argument
        pmt_channel_max = len(self.pmt_array) - 1
        assert pmt_channel_max >= 0, 'PMT array can not be empty'

        # Dict with available applet types
        self._applet_types: typing.Dict[str, typing.Callable[..., None]] = {
            self._PLOT_XY: self.ccb.plot_xy,
            self._BIG_NUMBER: self.ccb.big_number,
        }

        # Arguments
        self.pmt_channel = self.get_argument('PMT channel',
                                             NumberValue(default=0, step=1, min=0, max=pmt_channel_max, ndecimals=0),
                                             tooltip='PMT channel to monitor')
        self.applet_type = self.get_argument('Applet type',
                                             EnumerationValue(list(self._applet_types), self._PLOT_XY),
                                             tooltip='Choose an applet type (requires applet restart)')
        self.update_kernel_invariants('pmt_channel')

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if self.applet_type == self._PLOT_XY:
            # Modify keyword arguments
            kwargs.setdefault('last', True)
        elif self.applet_type == self._BIG_NUMBER:
            # Modify keyword arguments
            kwargs = {k: v for k, v in kwargs.items() if k in {'group', 'update_delay'}}
            kwargs.setdefault('digit_count', self.NUM_DIGITS_BIG_NUMBER)

        # Create applet based on chosen applet type
        self._applet_types[self.applet_type](*args, **kwargs)

    @kernel
    def _count(self):  # type: () -> None
        # Perform detection and get count
        self.pmt_array[self.pmt_channel].gate_rising(self.detection_window)
        count = self.pmt_array[self.pmt_channel].fetch_count()
        # Store obtained count
        self._store(count)

    @rpc(flags={'async'})
    def _store(self, count):  # type: (int) -> None
        # Calculate value to store
        value = count * self.y_scalar
        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)


@dax_client_factory
class MultiPmtMonitor(_PmtMonitorBase):
    """PMT monitor utility to monitor multiple PMT channels simultaneously."""

    APPLET_NAME = 'multi_pmt_monitor'
    APPLET_GROUP = 'dax'
    DEFAULT_DATASET = 'plot.dax.multi_pmt_monitor'

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Use multi-plot XY applet
        kwargs.setdefault('plot_names', 'PMT')
        self.ccb.plot_xy_multi(*args, **kwargs)

    @kernel
    def _count(self):  # type: () -> None
        # Perform detection
        with parallel:
            for c in self.pmt_array:
                c.gate_rising(self.detection_window)
        # Get counts
        counts = [c.fetch_count() for c in self.pmt_array]
        # Store obtained counts
        self._store(counts)

    @rpc(flags={'async'})
    def _store(self, counts):  # type: (typing.List[int]) -> None
        # Calculate value to store
        value = [c * self.y_scalar for c in counts]
        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)
