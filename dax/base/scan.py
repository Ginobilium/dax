from __future__ import annotations

import abc
import typing
import itertools
import numpy as np
import collections
import collections.abc
import os.path
import h5py

from artiq.language.core import kernel, portable, host_only, TerminationRequested
from artiq.language.environment import BooleanValue
from artiq.language.scan import ScanObject, Scannable

import dax.base.system
import dax.util.artiq

__all__ = ['DaxScan', 'DaxScanReader', 'DaxScanChain']

_S_T = typing.Union[ScanObject, typing.Sequence[typing.Any]]  # Scan object type

# Workaround required for Python<3.9
if typing.TYPE_CHECKING:  # pragma: no cover
    _SD_T = collections.OrderedDict[str, _S_T]  # Scan dict type
else:
    _SD_T = collections.OrderedDict


class _ScanProductGenerator:
    """Generator class for a cartesian product of scans.

    This class is inspired by the ARTIQ MultiScanManager class.
    """

    _keys: typing.Sequence[str]
    _scans: typing.Sequence[_S_T]
    _enable_index: bool

    class _ScanItem:
        kernel_invariants: typing.Set[str]

        def __init__(self, **kwargs: typing.Any):
            # Create kernel invariants attribute
            super(_ScanProductGenerator._ScanItem, self).__setattr__('kernel_invariants', set())
            # Set all kwargs as attributes
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __repr__(self) -> str:
            """Return a string representation of this object."""
            attributes: str = ', '.join(f'{k}={getattr(self, k)}' for k in self.kernel_invariants)
            return f'{self.__class__.__name__}: {attributes}'

        def __setattr__(self, key: str, value: typing.Any) -> None:
            # Set attribute by calling super
            super(_ScanProductGenerator._ScanItem, self).__setattr__(key, value)
            # Add attribute to kernel invariants
            self.kernel_invariants.add(key)

    class ScanPoint(_ScanItem):
        pass

    class ScanIndex(_ScanItem):
        def __init__(self, **kwargs: int):
            # Convert all values and call super
            super(_ScanProductGenerator.ScanIndex, self).__init__(**{k: np.int32(v) for k, v in kwargs.items()})

    def __init__(self, scans: _SD_T, *, enable_index: bool = True):
        """Create a new scan product generator.

        The ``enable_index`` parameter can be used to disable index objects,
        potentially reducing the memory footprint of the scan.

        :param scans: A list of tuples with the key and values of the scan
        :param enable_index: If :const:`False`, empty index objects are returned
        """
        assert isinstance(scans, collections.OrderedDict), 'Scans must be of type OrderedDict'
        assert isinstance(enable_index, bool), 'The enable index flag must be of type bool'

        # Unpack scan tuples
        self._keys = list(scans.keys())
        self._scans = list(scans.values())
        # Store enable index flag
        self._enable_index = enable_index

    def _point_generator(self) -> typing.Iterator[ScanPoint]:
        """Returns a generator for scan points."""
        for values in itertools.product(*self._scans):
            yield self.ScanPoint(**{k: v for k, v in zip(self._keys, values)})

    def _index_generator(self) -> typing.Iterator[ScanIndex]:
        """Returns a generator for scan indices."""
        if self._enable_index:
            for indices in itertools.product(*(range(len(s)) for s in self._scans)):
                # Yield a scan index object for every set of indices
                yield self.ScanIndex(**{k: v for k, v in zip(self._keys, indices)})
        else:
            # Create one empty scan index
            si = self.ScanIndex()
            for _ in range(np.prod([len(s) for s in self._scans])):
                # Yield the empty scan index object for all
                yield si

    def __iter__(self) -> typing.Iterator[typing.Tuple[ScanPoint, ScanIndex]]:
        """Returns a generator that returns tuples of a scan point and a scan index."""
        # Storing a list of tuples instead of two list hopefully results in better data locality
        return zip(self._point_generator(), self._index_generator())


class DaxScan(dax.base.system.DaxBase, abc.ABC):
    """Scanning class for standardized scanning functionality.

    Users can inherit this class to implement their scanning experiments.
    The first step is to build the scan by overriding the :func:`build_scan` function.
    Use the :func:`add_scan` function to add normal ARTIQ scannables to this scan object.
    Static scans can be added using the :func:`add_static_scan` function.
    Regular ARTIQ functions are available to obtain other arguments.

    Adding multiple scans results automatically in a multidimensional scan by scanning over
    all value combinations in the cartesian product of the scans.
    The first scan added represents the dimension which is only scanned once.
    All next scans are repeatedly performed to form the cartesian product.

    The scan class implements a :func:`run` function that controls the overall scanning flow.
    The :func:`prepare` and :func:`analyze` functions are not implemented by the scan class, but users
    are free to provide their own implementations.

    The following functions can be overridden to define scanning behavior:

    1. :func:`host_setup`
    2. :func:`device_setup`
    3. :func:`run_point` (must be implemented)
    4. :func:`device_cleanup`
    5. :func:`host_cleanup`

    Finally, the :func:`host_enter` and :func:`host_exit` functions can be overridden to implement any
    functionality executed once at the start of or just before leaving the :func:`run` function.

    To exit a scan early, call the :func:`stop_scan` function.

    By default, an infinite scan option is added to the arguments which allows infinite looping
    over the available points.
    An infinite scan can be stopped by using the :func:`stop_scan` function or by using
    the "Terminate experiment" button in the dashboard.
    It is possible to disable the infinite scan argument for an experiment by setting the
    :attr:`INFINITE_SCAN_ARGUMENT` class attribute to :const:`False`.
    The default setting of the infinite scan argument can be modified by setting the
    :attr:`INFINITE_SCAN_DEFAULT` class attribute.

    The :func:`run_point` function has access to a point and an index argument.
    Users can disable the index argument to reduce the memory footprint of the experiment.
    The :attr:`ENABLE_SCAN_INDEX` attribute can be used to configure this behavior (default: :const:`True`).
    When the index is disabled, the passed index argument will be empty.

    In case scanning is performed in a kernel, users are responsible for setting
    up the right devices to actually run a kernel.

    Arguments passed to the :func:`build` function are passed to the super class for
    compatibility with other libraries and abstraction layers.
    It is still possible to pass arguments to the :func:`build_scan` function by using special
    keyword arguments of the :func:`build` function of which the keywords are defined
    in the :attr:`SCAN_ARGS_KEY` and :attr:`SCAN_KWARGS_KEY` attributes.
    """

    INFINITE_SCAN_ARGUMENT: typing.ClassVar[bool] = True
    """Flag to enable the infinite scan argument."""
    INFINITE_SCAN_DEFAULT: typing.ClassVar[bool] = False
    """Default setting of the infinite scan."""

    ENABLE_SCAN_INDEX: typing.ClassVar[bool] = True
    """Flag to enable the index argument in the run_point() function."""

    SCAN_GROUP: typing.ClassVar[str] = 'scan'
    """The group name for archiving data."""
    SCAN_KEY_FORMAT: typing.ClassVar[str] = SCAN_GROUP + '/{key}'
    """Dataset key format for archiving independent scans."""
    SCAN_PRODUCT_GROUP: typing.ClassVar[str] = 'product'
    """The sub-group name for archiving product data."""
    SCAN_PRODUCT_KEY_FORMAT: typing.ClassVar[str] = f'{SCAN_GROUP}/{SCAN_PRODUCT_GROUP}/{{key}}'
    """Dataset key format for archiving scan products."""

    SCAN_ARGS_KEY: typing.ClassVar[str] = 'scan_args'
    """:func:`build` keyword argument for positional arguments passed to :func:`build_scan`."""
    SCAN_KWARGS_KEY: typing.ClassVar[str] = 'scan_kwargs'
    """:func:`build` keyword argument for keyword arguments passed to :func:`build_scan`."""

    SCAN_CORE_ATTR: typing.ClassVar[str] = 'core'
    """Attribute name of the core device."""

    __in_build: bool
    __terminated: bool
    _dax_scan_scannables: _SD_T
    _dax_scan_scheduler: typing.Any
    _dax_scan_infinite: bool
    _dax_scan_elements: typing.List[typing.Any]
    _dax_scan_index: np.int32

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the scan object using the :func:`build_scan` function.

        Normally users would build their scan object by overriding the :func:`build_scan` function.
        In specific cases where this function might be overridden, do not forget to call ``super().build()``.

        :param args: Positional arguments forwarded to the superclass
        :param kwargs: Keyword arguments forwarded to the superclass (includes args and kwargs for :func:`build_scan`)
        """

        assert isinstance(self.INFINITE_SCAN_ARGUMENT, bool), 'Infinite scan argument flag must be of type bool'
        assert isinstance(self.INFINITE_SCAN_DEFAULT, bool), 'Infinite scan default flag must be of type bool'
        assert isinstance(self.ENABLE_SCAN_INDEX, bool), 'Enable scan index flag must be of type bool'
        assert isinstance(self.SCAN_ARGS_KEY, str), 'Scan args keyword must be of type str'
        assert isinstance(self.SCAN_KWARGS_KEY, str), 'Scan kwargs keyword must be of type str'
        assert isinstance(self.SCAN_CORE_ATTR, str), 'Scan core attribute name must be of type str'

        # Obtain the scan args and kwargs
        scan_args: typing.Sequence[typing.Any] = kwargs.pop(self.SCAN_ARGS_KEY, ())
        scan_kwargs: typing.Dict[str, typing.Any] = kwargs.pop(self.SCAN_KWARGS_KEY, {})
        assert isinstance(scan_args, collections.abc.Sequence), 'Scan args must be a sequence'
        assert isinstance(scan_kwargs, dict), 'Scan kwargs must be a dict'
        assert all(isinstance(k, str) for k in scan_kwargs), 'All scan kwarg keys must be of type str'

        # Check if host_*() functions are all non-kernel functions
        if any(dax.util.artiq.is_kernel(f) for f in [self.host_enter, self.host_setup,
                                                     self.host_cleanup, self.host_exit]):
            raise TypeError('host_*() functions can not be kernels')
        # Check that _run_dax_scan_setup/cleanup aren't kernels
        if any(dax.util.artiq.is_kernel(f) for f in [self._run_dax_scan_setup, self._run_dax_scan_cleanup]):
            raise TypeError('_run_dax_scan_*() functions cannot be kernels')

        # Initialize variables
        self.__terminated = False

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxScan, self).build(*args, **kwargs)

        # Collection of scannables
        self._dax_scan_scannables = collections.OrderedDict()

        # The scheduler object
        self._dax_scan_scheduler = self.get_device('scheduler')

        # Build this scan (no args or kwargs available)
        self.logger.debug('Building scan')
        self.__in_build = True
        # noinspection PyArgumentList
        self.build_scan(*scan_args, **scan_kwargs)  # type: ignore[call-arg]
        self.__in_build = False

        # Confirm we have a core attribute
        if not hasattr(self, self.SCAN_CORE_ATTR):
            raise AttributeError(f'DAX.scan could not find the core attribute "{self.SCAN_CORE_ATTR}"')

        if self.INFINITE_SCAN_ARGUMENT:
            # Add an argument for infinite scan
            self._dax_scan_infinite = self.get_argument('Infinite scan', BooleanValue(self.INFINITE_SCAN_DEFAULT),
                                                        group='DAX.scan',
                                                        tooltip='Loop infinitely over the scan points')
        else:
            # If infinite scan argument is disabled, the value is always the default one
            self._dax_scan_infinite = self.INFINITE_SCAN_DEFAULT

        # Update kernel invariants
        self.update_kernel_invariants('_dax_scan_scheduler', '_dax_scan_infinite')

    @abc.abstractmethod
    def build_scan(self) -> None:  # pragma: no cover
        """Users should override this method to build their scan.

        To build the scan, use the :func:`add_scan` and :func:`add_static_scan` functions.
        Additionally, users can also add normal arguments using the standard ARTIQ functions.

        It is possible to pass arguments from the constructor to this function using the
        keyword arguments defined in :attr:`SCAN_ARGS_KEY` and :attr:`SCAN_KWARGS_KEY` (see :class:`DaxScan`).
        """
        pass

    @property
    def is_infinite_scan(self) -> bool:
        """:const:`True` if the scan was set to be an infinite scan."""
        if hasattr(self, '_dax_scan_infinite'):
            return self._dax_scan_infinite
        else:
            raise AttributeError('is_scan_infinite can only be obtained after build() was called')

    @property
    def is_terminated_scan(self) -> bool:
        """:const:`True` if the scan was terminated."""
        return self.__terminated

    def add_scan(self, key: str, name: str, scannable: Scannable, *,
                 group: typing.Optional[str] = None, tooltip: typing.Optional[str] = None) -> None:
        """Register a scannable.

        The first scan added represents the dimension which is only scanned once.
        All next scans are repeatedly performed to form the cartesian product.
        Hence, the last added scan will be repeated the most times.

        Note that scans can be reordered using the :func:`set_scan_order` function.
        The order in the ARTIQ dashboard will not change, but the product will be generated differently.

        Scannables are normal ARTIQ :class:`Scannable` objects and will appear in the user interface.

        This function can only be called in the :func:`build_scan` function.

        :param key: Unique key of the scan, used to obtain the value later
        :param name: The name of the argument
        :param scannable: An ARTIQ :class:`Scannable` object
        :param group: The argument group name
        :param tooltip: The shown tooltip
        """
        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(name, str), 'Name must be of type str'
        assert isinstance(group, str) or group is None, 'Group must be of type str or None'
        assert isinstance(tooltip, str) or tooltip is None, 'Tooltip must be of type str or None'

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise RuntimeError('add_scan() can only be called in the build_scan() method')

        # Verify type of the given scannable
        if not isinstance(scannable, Scannable):
            raise TypeError('The given processor is not a scannable')

        # Verify the key is valid and not in use
        if not key.isidentifier():
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self._dax_scan_scannables:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add argument to the list of scannables
        self._dax_scan_scannables[key] = self.get_argument(name, scannable, group=group, tooltip=tooltip)

    def add_static_scan(self, key: str, points: typing.Sequence[typing.Any]) -> None:
        """Register a static scan.

        A static scan is handled the same as a regular scan (see :func:`add_scan`).
        The difference is that a static scan has predefined points and
        therefore does not appear in the user interface.

        :param key: Unique key of the scan, used to obtain the value later
        :param points: A sequence with points
        """
        assert isinstance(key, str), 'Key must be of type str'

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise RuntimeError('add_scan() can only be called in the build_scan() method')

        # Verify type of the points
        if not isinstance(points, collections.abc.Sequence):
            raise TypeError('Points must be a sequence')
        if isinstance(points, np.ndarray):
            if not any(np.issubdtype(points.dtype, t)
                       for t in [np.int32, np.int64, np.floating, np.bool_, np.character]):
                raise TypeError('The NumPy point type is not supported')
            if points.ndim != 1:
                raise TypeError('Only NumPy arrays with one dimension are supported')
        else:
            if not all(isinstance(e, type(points[0])) for e in points):
                raise TypeError('The point types must be homogeneous')
            if len(points) > 0 and not isinstance(points[0], (int, float, bool, str)):
                raise TypeError('The point type is not supported')

        # Verify the key is valid and not in use
        if not key.isidentifier():
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self._dax_scan_scannables:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add points to the list of scannables
        self._dax_scan_scannables[key] = points

    @host_only
    def set_scan_order(self, *keys: str) -> None:
        """Set the scan order given a sequence of keys.

        The given sequence of scan keys will determine the new order of the scans.
        In the list of scans, the given keys will be moved to the end of the list in the given order.
        Keys of scans that are not in the new scan order will remain in their relative position.
        For more information about the order of scans and the cartesian product, see :func:`add_scan`.

        This function can only be called in the :func:`build_scan` function.

        :param keys: The scan keys in the desired order
        """

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise RuntimeError('set_scan_order() can only be called in the build_scan() method')

        for k in keys:
            # Move key to the end of the list
            self._dax_scan_scannables.move_to_end(k)

    @host_only
    def get_scan_points(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the cartesian product of scan points for analysis.

        A list of values is returned on a per-key basis.
        The values are returned with the same order as provided to the actual run,
        as the cartesian product of all scannables.

        To get the values without applying the product, see :func:`get_scannables`.

        This function can only be used after the :func:`run` function was called
        which normally means it is not available during the build and prepare phase.

        :return: A dict containing all the scan points on a per-key basis
        """
        if hasattr(self, '_dax_scan_elements'):
            return {key: [getattr(point, key) for point, _ in self._dax_scan_elements]
                    for key in self._dax_scan_scannables}
        else:
            raise AttributeError('Scan points can only be obtained after run() was called')

    @host_only
    def get_scannables(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the scan points without product.

        For every key, a list of scan values is returned.
        These values are the individual values for each key, without applying the product.

        To get the values including the product, see :func:`get_scan_points`.

        :return: A dict containing the individual scan values on a per-key basis
        """
        return {key: list(scannable) for key, scannable in self._dax_scan_scannables.items()}

    @host_only
    def init_scan_elements(self) -> None:
        """Initialize the list of scan elements.

        By default, this is called at the beginning of :func:`run`, however it may be called in :func:`prepare` if the
        user desires the ability to call :func:`get_scan_points` before :func:`run`.
        """
        # Check if build() was called
        assert hasattr(self, '_dax_scan_scannables'), 'DaxScan.build() was not called'

        # Make the scan elements
        if self._dax_scan_scannables:
            self._dax_scan_elements = list(_ScanProductGenerator(self._dax_scan_scannables,
                                                                 enable_index=self.ENABLE_SCAN_INDEX))
        else:
            self._dax_scan_elements = []
        self.update_kernel_invariants('_dax_scan_elements')
        self.logger.debug(f'Prepared {len(self._dax_scan_elements)} scan point(s) '
                          f'with {len(self._dax_scan_scannables)} scan parameter(s)')

    """Run functions"""

    @host_only
    def run(self) -> None:
        """Entry point of the experiment implemented by the scan class.

        Normally users do not have to override this method.
        Once-executed entry code can use the :func:`host_enter` function instead.
        """

        # Initialize scan elements if not already done
        if not hasattr(self, '_dax_scan_elements'):
            self.init_scan_elements()

        if not self._dax_scan_elements:
            # There are no scan points
            self.logger.warning('No scan points found, aborting experiment')
            return

        # Reporting infinite scan flag
        self.logger.debug(f'Infinite scan: {self.is_infinite_scan}')

        for key, scannable in self._dax_scan_scannables.items():
            # Archive values of independent scan
            self.set_dataset(self.SCAN_KEY_FORMAT.format(key=key), [e for e in scannable], archive=True)
            # Archive cartesian product of scan point (separate dataset for every key)
            self.set_dataset(self.SCAN_PRODUCT_KEY_FORMAT.format(key=key),
                             [getattr(point, key) for point, _ in self._dax_scan_elements], archive=True)

        # Index of current scan element
        self._dax_scan_index = np.int32(0)

        try:
            # Call the host enter code
            self.logger.debug('Performing host enter procedure')
            self.host_enter()

            while self._dax_scan_index < len(self._dax_scan_elements):
                while self._dax_scan_scheduler.check_pause():
                    # Pause the scan
                    self.logger.debug('Pausing scan')
                    getattr(self, self.SCAN_CORE_ATTR).comm.close()  # Close communications before pausing
                    self._dax_scan_scheduler.pause()  # Can raise a TerminationRequested exception
                    self.logger.debug('Resuming scan')

                try:
                    # Coming from a host context, perform host setup
                    self.logger.debug('Performing host setup')
                    self.host_setup()
                    self.logger.debug('Performing run_dax_scan setup')
                    self._run_dax_scan_setup()

                    # Run the scan
                    if dax.util.artiq.is_kernel(self.run_point):
                        self.logger.debug('Running scan on core device')
                        self._run_dax_scan_in_kernel()
                    else:
                        self.logger.debug('Running scan on host')
                        self._run_dax_scan()

                finally:
                    # One time host cleanup
                    self.logger.debug('Performing run_dax_scan cleanup')
                    self._run_dax_scan_cleanup()
                    self.logger.debug('Performing host cleanup')
                    self.host_cleanup()

        except TerminationRequested:
            # Scan was terminated
            self.__terminated = True
            if self.is_infinite_scan:
                self.logger.info('Infinite scan was terminated by user request')
            else:
                self.logger.warning('Scan was terminated by user request')

        else:
            # Call the host exit code
            self.logger.debug('Performing host exit procedure')
            self.host_exit()

    @kernel
    def _run_dax_scan_in_kernel(self):  # type: () -> None
        """Run scan on the core device."""
        self._run_dax_scan()

    @portable
    def _run_dax_scan(self):  # type: () -> None
        """Portable run scan function."""
        try:
            # Perform device setup
            self.device_setup()

            while self._dax_scan_index < len(self._dax_scan_elements):
                # Check for pause condition
                if self._dax_scan_scheduler.check_pause():
                    break  # Break to exit the run scan function

                # Run for one point
                point, index = self._dax_scan_elements[self._dax_scan_index]
                self.run_point(point, index)

                # Increment index
                self._dax_scan_index += np.int32(1)

                # Handle infinite scan
                if self._dax_scan_infinite and self._dax_scan_index == len(self._dax_scan_elements):
                    self._dax_scan_index = np.int32(0)

        finally:
            # Perform device cleanup
            self.device_cleanup()

    @portable
    def stop_scan(self):  # type: () -> None
        """Stop the scan after the current point.

        This function can only be called from the :func:`run_point` function.
        """

        # Stop the scan by moving the index to the end (+1 to differentiate with infinite scan)
        self._dax_scan_index = np.int32(len(self._dax_scan_elements) + 1)

    """Functions to be implemented by the user"""

    def host_enter(self) -> None:  # pragma: no cover
        """0. Entry code on the host, called once."""
        pass

    def host_setup(self) -> None:  # pragma: no cover
        """1. Setup on the host, called once at entry and after a pause."""
        pass

    def _run_dax_scan_setup(self) -> None:  # pragma: no cover
        """1.5. Called on the host after :func:`host_setup` and just before :func:`_run_dax_scan`.

        Useful for e.g. confirming that necessary attributes created in :func:`host_enter` or :func:`host_setup`
        haven't been overwritten by a subclass.
        """
        pass

    @portable
    def device_setup(self):  # type: () -> None  # pragma: no cover
        """2. Setup on the core device, called once at entry and after a pause.

        Can for example be used to reset the core.
        """
        pass

    @abc.abstractmethod
    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None  # pragma: no cover
        """3. Code to run for a single point, called as many times as there are points.

        :param point: Point object containing the current scan parameter values
        :param index: Index object containing the current scan parameter indices
        """
        pass

    @portable
    def device_cleanup(self):  # type: () -> None  # pragma: no cover
        """4. Cleanup on the core device, called once after scanning and before a pause.

        In case the device cleanup function is a kernel, it is good to add a ``self.core.break_realtime()``
        at the start of this function to make sure operations can execute in case of an
        underflow exception.
        """
        pass

    def _run_dax_scan_cleanup(self) -> None:  # pragma: no cover
        """4.5. Called on the host after :func:`_run_dax_scan`."""
        pass

    def host_cleanup(self) -> None:  # pragma: no cover
        """5. Cleanup on the host, called once after scanning and before a pause."""
        pass

    def host_exit(self) -> None:  # pragma: no cover
        """6. Exit code on the host, called if the scan finished without exceptions."""
        pass


class DaxScanReader:
    """Reader class to retrieve scan data from and HDF5 archive or a live :class:`DaxScan` object.

    This class will read an HDF5 file, extract scan data, and expose it through attributes.
    It is also possible to provide a :class:`DaxScan` object as source and the same data
    will be exposed, though :class:`DaxScan` also provides methods to obtain data directly.

    :attr:`keys` is a list of keys for which scan data is available.

    :attr:`scannables` is a dict which for each key contains the list of values.
    These values are the individual values for each key, without applying the product.

    :attr:`scan_points` is a dict with for each key contains the list of scan points.
    The values are returned with the same order as was provided to the actual run,
    as the cartesian product of all scannables.
    """

    keys: typing.List[str]
    """A list of keys for which scan data is available."""
    scannables: typing.Dict[str, typing.List[typing.Any]]
    """A dict which for each key contains the list of values."""
    scan_points: typing.Dict[str, typing.List[typing.Any]]
    """A dict which for each key contains the list of scan points."""

    def __init__(self, source: typing.Union[DaxScan, str, h5py.File], *,
                 hdf5_group: typing.Optional[str] = None):
        """Create a new DAX.scan reader object.

        :param source: The source of the scan data
        :param hdf5_group: HDF5 group containing the data, defaults to root of the HDF5 file
        """
        assert isinstance(hdf5_group, str) or hdf5_group is None

        # Input conversion
        if isinstance(source, str):
            # Open HDF5 file
            source = h5py.File(os.path.expanduser(source), mode='r')

        if isinstance(source, DaxScan):
            # Get data from scan object
            self.scannables = source.get_scannables()
            self.scan_points = source.get_scan_points()
            self.keys = list(self.scannables.keys())

        elif isinstance(source, h5py.File):
            # Construct HDF5 group name
            path = [] if hdf5_group is None else [hdf5_group]
            group_name = '/'.join(path + ['datasets', DaxScan.SCAN_GROUP])
            # Verify format of HDF5 file
            if group_name not in source:
                raise KeyError('The HDF5 file does not contain scanning data')

            # Get the group which contains all data
            group = source[group_name]

            # Read and convert data from HDF5 file
            self.keys = [k for k in group.keys() if k != DaxScan.SCAN_PRODUCT_GROUP]
            self.scannables = {k: group[k][()] for k in self.keys}
            group = group[DaxScan.SCAN_PRODUCT_GROUP]  # Switch to group that contains scan points
            self.scan_points = {k: group[k][()] for k in self.keys}

        else:
            raise TypeError('Unsupported source type')


class DaxScanChain:
    """DAX.scan utility class to allow linear chaining of multiple scan ranges.

    Users may use this class within :class:`DaxScan` experiments to create a single scan of disparate ranges.
    The resulting scan will be treated as a single value with a key defined in the :func:`__init__` method.
    As a result, this class must be only used within the :func:`build_scan` function.

    The :func:`add_scan` will allow the scans to appear in the ARTIQ Dashboard. Take care to call the
    function from this class, not from the :class:`DaxScan` experiment class.

    This class must be used as a context, with example code below::

        with DaxScanChain(self, 'key', group='group') as chain:
            chain.add_scan('name', scannable, tooltip='tooltip')
    """

    __in_context: typing.ClassVar[bool] = False  # Treat as a static class variable to ensure no reentrant context

    _dax_scan: DaxScan
    _key: str
    _group: typing.Optional[str]
    _scannables: _SD_T
    _added_scan: bool

    def __init__(self, dax_scan: DaxScan, key: str, *, group: typing.Optional[str] = None):
        """Create a new :class:`DaxScanChain` object.

        :param dax_scan: The :class:`DaxScan` object to which this chain will be added
        :param key: Unique key of the chained scan, used to obtain the value later
        :param group: The argument group name
        """
        assert isinstance(dax_scan, DaxScan), 'Must be given a DaxScan class'
        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(group, str) or group is None, 'Group must be of type str or None'

        # The DaxScan class and key will be used to add the chained scans into the experiment
        self._dax_scan = dax_scan
        self._key = key
        self._group = group

        # Set default values
        self._scannables = collections.OrderedDict()
        self._added_scan = False

    def add_scan(self, name: str, scannable: Scannable, *, tooltip: typing.Optional[str] = None) -> None:
        """Register a scannable.

        Scans will be chained linearly. The first scan will represent the first range to be scanned over, the second
        scan the second range, etc.

        Note that scans can be reordered using the :func:`set_scan_order` function, but only in the context.
        The order in the ARTIQ dashboard will not change, but the scans will be scanned over in a different order.

        Scannables are normal ARTIQ :class:`Scannable` objects and will appear in the user interface.

        This function can only be called in the :class:`DaxScanChain` context,
        and it may only be used in the :func:`DaxScan.build_scan` function.

        :param name: The name of the argument
        :param scannable: An ARTIQ :class:`Scannable` object
        :param tooltip: The shown tooltip
        """
        assert isinstance(name, str), 'Name must be of type str'
        assert isinstance(tooltip, str) or tooltip is None, 'Tooltip must be of type str or None'

        # Verify this function was called in the DaxScanChain context
        if not DaxScanChain.__in_context:
            raise RuntimeError('add_scan() can only be called in the DaxScanChain context')

        # Verify type of the given scannable
        if not isinstance(scannable, Scannable):
            raise TypeError('The given processor is not a scannable')

        # Verify that names are unique within the group
        if name in self._scannables.keys():
            raise LookupError('Scan names must be unique within a group')

        # Add argument to the ordered dict of scannables
        self._scannables[name] = self._dax_scan.get_argument(name, scannable, group=self._group, tooltip=tooltip)

    def _get_chained_scan(self) -> typing.Sequence[typing.Any]:
        """Get the scan points chained into a single list.

        Create the chained scan object. The values are returned in the same order as provided on the actual run.

        :return: All the scan points chained into a single list
        """
        # Function will only need to run successfully during actual experiment runs, not during the build phase
        return [] if None in self._scannables.values() else [p for s in self._scannables.values() for p in s]

    def __enter__(self) -> DaxScanChain:
        """Enter the scan chain context.

        Normally this function should not be called directly but by the ``with`` statement instead.
        """
        if DaxScanChain.__in_context or self._added_scan:
            raise RuntimeError('DaxScanChain is not reentrant')
        if self._added_scan:
            raise RuntimeError('DaxScanChain cannot be entered twice')

        # Set flags
        DaxScanChain.__in_context = True

        # Must return self to allow setting variables using the with statement
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        """Exit the scan chain context.

        Normally this function should not be called directly but by the ``with`` statement instead.
        """

        # Add all scan points as a static scan
        self._dax_scan.add_static_scan(self._key, self._get_chained_scan())

        # Set flags
        DaxScanChain.__in_context = False
        self._added_scan = True
