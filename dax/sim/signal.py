import abc
import typing
import operator
import datetime
import collections
import numpy as np
import vcd.writer
import sortedcontainers

import artiq.language.core
from artiq.language.units import ns

from dax.sim.device import DaxSimDevice
from dax import __version__ as _dax_version
import dax.util.units

__all__ = ['Signal', 'SignalNotSetError', 'SignalNotFoundError', 'DaxSignalManager',
           'NullSignalManager', 'VcdSignalManager', 'PeekSignalManager',
           'get_signal_manager', 'set_signal_manager']

_T_T = np.int64  # Timestamp type
_O_T = typing.Union[int, np.int32, np.int64]  # Time offset type

_ST_T = typing.Union[typing.Type[bool], typing.Type[int], typing.Type[float],  # The signal type type
                     typing.Type[str], typing.Type[object]]
_SS_T = typing.Optional[int]  # The signal size type
_SV_T = typing.Union[bool, int, np.int32, np.int64, float, str]  # The signal value type


class Signal(abc.ABC):
    """Abstract class to represent a signal."""

    __scope: DaxSimDevice
    __name: str
    __type: _ST_T
    __size: _SS_T

    _EXPECTED_TYPES: typing.ClassVar[typing.Dict[_ST_T, typing.Union[type, typing.Tuple[type, ...]]]] = {
        bool: bool,
        int: (int, np.int32, np.int64),
        float: float,
        str: str,
        object: bool,
    }
    """Valid value types for each signal type."""

    _SPECIAL_VALUES: typing.ClassVar[typing.Dict[_ST_T, typing.Set[typing.Any]]] = {
        bool: {'x', 'X', 'z', 'Z', 0, 1},  # Also matches float and NumPy int
        int: {'x', 'X', 'z', 'Z'},
        float: set(),
        str: set(),
        object: set(),
    }
    """Valid special values for each signal type."""

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T = None):
        """Initialize a new signal object."""
        assert isinstance(scope, DaxSimDevice), 'Signal scope must be of type DaxSimDevice'
        assert isinstance(name, str), 'Signal name must be of type str'
        assert name.isidentifier(), 'Invalid signal name (must be a valid identifier)'
        assert type_ in self._EXPECTED_TYPES, 'Invalid signal type'
        if type_ is bool:
            assert isinstance(size, int) and size > 0, 'Signal size must be an integer > 0 for signal type bool'
        else:
            assert size is None, f'Size not supported for signal type "{type_}"'

        # Store attributes
        self.__scope = scope
        self.__name = name
        self.__type = type_
        self.__size = size

    @property
    def scope(self) -> DaxSimDevice:
        """Scope of the signal, which is the device object."""
        return self.__scope

    @property
    def name(self) -> str:
        """Name of the signal."""
        return self.__name

    @property
    def type(self) -> _ST_T:
        """Type of the signal."""
        return self.__type

    @property
    def size(self) -> _SS_T:
        """Size of the signal."""
        return self.__size

    def normalize(self, value: typing.Any) -> _SV_T:
        """Normalize a value for this signal.

        :param value: The value to normalize
        :return: The normalized value
        :raises ValueError: Raised if the value is invalid
        """
        if self.size in {None, 1}:
            # noinspection PyTypeHints
            if isinstance(value, self._EXPECTED_TYPES[self.type]):
                return typing.cast(_SV_T, value)  # Value is legal (expected type), cast required for mypy
            if value in self._SPECIAL_VALUES[self.type]:
                return typing.cast(_SV_T, value)  # Value is legal (special value), cast required for mypy
        elif self.type is bool and isinstance(value, str) and len(value) == self.size and all(
                v in {'x', 'X', 'z', 'Z', '0', '1'} for v in value):
            return value.lower()  # Value is legal (bool vector) (store lower case)

        # Value did not pass check
        raise ValueError(f'Invalid value "{value}" for signal type "{self.type}"')

    @abc.abstractmethod
    def push(self, value: typing.Any, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        """Push an event to this signal (i.e. change the value of this signal at the given time).

        Values are automatically normalized before inserted into the signal manager (see :func:`normalize`).

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to set events at a different
        time without modifying the timeline.

        Bool type signals can have values ``0``, ``1``, ``'X'``, ``'Z'``.
        A vector of a bool type signal has a value of type ``str`` (e.g. ``'1001XZ'``).
        An integer can be converted to a bool vector with the following example code:
        ``f'{value & 0xFF:08b}'`` (size 8 bool vector).

        Integer type variables can have any int value or any value legal for a bool type signal.

        Float type variables can only be assigned float values.

        Event (``object``) type signals represent timestamps and do not have a value.
        We recommend to always use value :const:`True` for event type signals.

        String type signals can use value :const:`None` which is equivalent to ``'Z'``.

        :param value: The new value of this signal
        :param time: Optional time in machine units when the event happened (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :raises ValueError: Raised if the value is invalid
        """
        pass

    @abc.abstractmethod
    def pull(self, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        """Pull the value of this signal at the given time.

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to get values at a different
        time without modifying the timeline.

        :param time: Optional time in machine units to obtain the signal value (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :return: The value of the given signal at the given time and offset
        :raises SignalNotSetError: Raised if the signal was not set at the given time
        """
        pass

    def __str__(self) -> str:
        """The key of the corresponding device followed by the name of this signal."""
        return f'{self.scope}.{self.name}'

    def __repr__(self) -> str:
        """See :func:`__str__`."""
        return str(self)


class SignalNotSetError(RuntimeError):
    """This exception is raised when a signal value is requested but the signal is not set."""

    def __init__(self, signal: Signal, time: _T_T, msg: str = ''):
        msg_ = f'Signal "{signal}" not set at time {time}{f": {msg}" if msg else ""}'
        super(SignalNotSetError, self).__init__(msg_)


class SignalNotFoundError(KeyError):
    """This exception is raised when a requested signal does not exist."""

    def __init__(self, scope: DaxSimDevice, name: str):
        super(SignalNotFoundError, self).__init__(f'Signal "{scope}.{name}" could not be found')


_S_T = typing.TypeVar('_S_T', bound=Signal)  # The abstract signal type variable


class DaxSignalManager(abc.ABC, typing.Generic[_S_T]):
    """Base class for classes that manage simulated signals."""

    __signals: typing.Dict[typing.Tuple[DaxSimDevice, str], _S_T]
    """Registered signals"""

    def __init__(self) -> None:
        self.__signals = {}

    def register(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                 size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> _S_T:
        """Register a signal.

        Signals have to be registered before any events are committed.
        Used by the device driver to register signals.

        Possible types and expected arguments:

        - ``bool`` (a register with bit values ``0``, ``1``, ``'X'``, ``'Z'``), provide a size of the register
        - ``int``
        - ``float``
        - ``str``
        - ``object`` (an event type with no value)

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to :const:`None` (``'X'``))
        :return: The signal object used to call other functions of this class
        :raises LookupError: Raised if the signal was already registered
        """
        # Create the key
        key = (scope, name)

        if key in self.__signals:
            # A signal can not be registered more than once
            raise LookupError(f'Signal "{self.__signals[key]}" was already registered')

        # Create, register, and return signal
        signal = self._create_signal(scope, name, type_, size=size, init=init)
        self.__signals[key] = signal
        return signal

    def signal(self, scope: DaxSimDevice, name: str) -> _S_T:
        """Obtain an existing signal object.

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :return: The signal object used to call other functions of this class
        :raises SignalNotFoundError: Raised if the signal could not be found
        """
        # Create the key
        key = (scope, name)

        if key not in self.__signals:
            # Signal not found
            raise SignalNotFoundError(scope, name)

        # Return key
        return self.__signals[key]

    def __iter__(self) -> typing.Iterator[_S_T]:
        """Obtain an iterator over the registered signals."""
        return iter(self.__signals.values())

    def __len__(self) -> int:
        """Get the number of registered signals."""
        return len(self.__signals)

    @abc.abstractmethod
    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,  # pragma: no cover
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> _S_T:
        """Create a new signal object.

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to :const:`None` (``'X'``))
        :return: The signal object used to call other functions of this class
        :raises LookupError: Raised if the signal was already registered
        """
        pass

    @abc.abstractmethod
    def flush(self, ref_period: float) -> None:  # pragma: no cover
        """Flush the output of the signal manager.

        :param ref_period: The reference period (i.e. the time of one machine unit)
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:  # pragma: no cover
        """Close the signal manager.

        Note that this function must be reentrant!
        """
        pass


def _get_timestamp(time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _T_T:
    """Calculate the timestamp of an event."""
    if time is None:
        time = artiq.language.core.now_mu()  # noqa: ATQ101
    else:
        assert isinstance(time, np.int64), 'Time must be of type np.int64'
    return time + offset if offset else time


class ConstantSignal(Signal):
    """Class to represent a constant signal."""

    _init: typing.Optional[_SV_T]

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T]):
        super(ConstantSignal, self).__init__(scope, name, type_, size)
        self._init = None if init is None else self.normalize(init)

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        self.normalize(value)  # Do normalization (for exceptions) before dropping the event

    def pull(self, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if self._init is None:
            # Signal was not set
            raise SignalNotSetError(self, _get_timestamp(time, offset), msg='Signal not initialized')
        else:
            # Return the init value
            return self._init


class NullSignalManager(DaxSignalManager[ConstantSignal]):
    """A signal manager with constant signals (i.e. all push events to signals are dropped)."""

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> ConstantSignal:
        return ConstantSignal(scope, name, type_, size, init=init)

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass


class VcdSignal(ConstantSignal):
    """Class to represent a VCD signal."""

    __VCD_T = vcd.writer.Variable[vcd.writer.VarValue]  # VCD variable type
    E_T = typing.Tuple[typing.Union[int, np.int64], 'VcdSignal', _SV_T]  # Event type (string literal forward reference)

    _event_buffer: typing.List[E_T]
    _vcd: __VCD_T

    _VCD_TYPE: typing.ClassVar[typing.Dict[_ST_T, str]] = {
        bool: 'reg',
        int: 'integer',
        float: 'real',
        str: 'string',
        object: 'event',
    }
    """Dict to convert Python types to VCD types."""

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T],
                 vcd_: vcd.writer.VCDWriter, event_buffer: typing.List[E_T]):
        # Call super
        super(VcdSignal, self).__init__(scope, name, type_, size, init=init)
        # Store reference to shared and mutable event buffer
        self._event_buffer = event_buffer

        # Workaround for str init values (shows up as `z` instead of string value 'x')
        init = '' if type_ is str and init is None else init

        # Register this variable with the VCD writer
        self._vcd = vcd_.register_var(scope.key, name, var_type=self._VCD_TYPE[type_], size=size, init=init)

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Add event to buffer
        self._event_buffer.append((_get_timestamp(time, offset), self, self.normalize(value)))

    def normalize(self, value: typing.Any) -> _SV_T:
        # Call super
        v = super(VcdSignal, self).normalize(value)

        # Workaround for int values (NumPy int objects are not accepted)
        if self.type is int and isinstance(v, (np.int32, np.int64)):
            v = int(v)

        # Return value
        return v

    @property
    def vcd(self) -> __VCD_T:
        return self._vcd


class VcdSignalManager(DaxSignalManager[VcdSignal]):
    """VCD signal manager."""

    _timescale: float
    _file: typing.IO[str]
    _vcd: vcd.writer.VCDWriter
    _event_buffer: typing.List[VcdSignal.E_T]

    def __init__(self, file_name: str, timescale: float = 1 * ns):
        assert isinstance(file_name, str), 'Output file name must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert timescale > 0.0, 'Timescale must be > 0.0'

        # Call super
        super(VcdSignalManager, self).__init__()
        # Store timescale
        self._timescale = timescale

        # Open file
        self._file = open(file_name, mode='w')
        # Create VCD writer
        self._vcd = vcd.writer.VCDWriter(self._file,
                                         timescale=dax.util.units.time_to_str(timescale, precision=0),
                                         date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         comment=file_name,
                                         version=_dax_version)

        # Create the shared event buffer
        self._event_buffer = []

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> VcdSignal:
        return VcdSignal(scope, name, type_, size, init=init, vcd_=self._vcd, event_buffer=self._event_buffer)

    def flush(self, ref_period: float) -> None:
        # Sort the list of events (VCD writer can only handle a linear timeline)
        self._event_buffer.sort(key=operator.itemgetter(0))
        # Get a timestamp for now
        now: int = int(_get_timestamp())

        if ref_period == self._timescale:
            # Just iterate over the event buffer
            event_buffer_iter: typing.Iterator[VcdSignal.E_T] = iter(self._event_buffer)
        else:
            # Scale the timestamps if the reference period does not match the timescale
            scalar = ref_period / self._timescale
            event_buffer_iter = ((int(time * scalar), signal, value) for time, signal, value in self._event_buffer)
            # Scale the timestamp for now
            now = int(now * scalar)

        try:
            # Submit sorted events to the VCD writer
            for time, signal, value in event_buffer_iter:
                self._vcd.change(signal.vcd, time, value)
        except vcd.writer.VCDPhaseError as e:
            # Occurs when we try to submit a timestamp which is earlier than the last submitted timestamp
            raise RuntimeError('Attempt to go back in time too much') from e
        else:
            # Flush the VCD writer
            self._vcd.flush(now)

        # Clear the event buffer
        self._event_buffer.clear()

    def close(self) -> None:
        # Clear the event buffer
        self._event_buffer.clear()
        # Close the VCD writer (reentrant)
        self._vcd.close()
        # Close the VCD file (reentrant)
        self._file.close()


class PeekSignal(Signal):
    """Class to represent a peek signal."""

    # Workaround required for the local stubs of the sorted containers library
    if typing.TYPE_CHECKING:  # pragma: no cover
        _EB_T = sortedcontainers.SortedDict[_T_T, _SV_T]  # The peek signal event buffer type
        _TV_T = sortedcontainers.SortedKeysView[_T_T]  # The peek signal event buffer timestamp view type
    else:
        _EB_T = sortedcontainers.SortedDict
        _TV_T = typing.KeysView[_T_T]  # Using generic KeysView, helps the PyCharm type checker

    _push_buffer: typing.Deque[_SV_T]
    _event_buffer: _EB_T
    _timestamps: _TV_T

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T]):
        # Call super
        super(PeekSignal, self).__init__(scope, name, type_, size)

        # Create push buffer
        self._push_buffer = collections.deque()
        # Create event buffer
        self._event_buffer = sortedcontainers.SortedDict()
        if init is not None:
            self._event_buffer[np.int64(0)] = self.normalize(init)
        # Create timestamp view
        self._timestamps = self._event_buffer.keys()

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Normalize value and add value to event buffer
        # An existing value at the same timestamp will be overwritten, just as the ARTIQ RTIO system does
        self._event_buffer[_get_timestamp(time, offset)] = self.normalize(value)

    def pull(self, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if self._push_buffer:
            # Take an item from the buffer, push it, and return the value
            value = self._push_buffer.popleft()
            self.push(value, time=time, offset=offset)
            return value

        else:
            # Binary search for the insertion point (right) of the given timestamp
            index = self._event_buffer.bisect_right(_get_timestamp(time, offset))

            if index:
                # Return the value
                return self._event_buffer[self._timestamps[index - 1]]
            else:
                # Signal was not set, raise an exception
                raise SignalNotSetError(self, _get_timestamp(time, offset))

    def push_buffer(self, buffer: typing.Sequence[typing.Any]) -> None:
        """Push a buffer of values this signal.

        Values in the buffer will be pushed automatically at the next call to :func:`pull`. See also :func:`push`.

        :param buffer: The buffer of values to queue
        :raises ValueError: Raised if the value is invalid
        """
        # Add values to the push buffer
        self._push_buffer.extend(self.normalize(v) for v in buffer)

    def clear(self) -> None:
        """Clear buffers."""
        self._push_buffer.clear()
        self._event_buffer.clear()


class PeekSignalManager(DaxSignalManager[PeekSignal]):
    """Peek signal manager."""

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> PeekSignal:
        return PeekSignal(scope, name, type_, size, init=init)

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        # Clear all signals
        for signal in self:
            signal.clear()


_signal_manager: DaxSignalManager[typing.Any] = NullSignalManager()
"""Singleton instance of the signal manager."""


def get_signal_manager() -> DaxSignalManager[typing.Any]:
    """Get the signal manager instance.

    The signal manager is used by simulated devices to register and change signals during simulation.

    :return: The signal manager object
    """
    return _signal_manager


def set_signal_manager(signal_manager: DaxSignalManager[typing.Any]) -> None:
    """Set a new signal manager.

    The old signal manager will be closed.

    :param signal_manager: The new signal manager object to use
    """

    # Close the current signal manager
    global _signal_manager
    _signal_manager.close()

    # Set the new signal manager
    _signal_manager = signal_manager
