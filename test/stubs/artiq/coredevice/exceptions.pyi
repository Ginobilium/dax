import builtins

ZeroDivisionError = builtins.ZeroDivisionError
ValueError = builtins.ValueError
IndexError = builtins.IndexError
RuntimeError = builtins.RuntimeError
AssertionError = builtins.AssertionError


class CoreException:
    ...


class InternalError(Exception):
    artiq_builtin: bool = ...


class CacheError(Exception):
    artiq_builtin: bool = ...


class RTIOUnderflow(Exception):
    artiq_builtin: bool = ...


class RTIOOverflow(Exception):
    artiq_builtin: bool = ...


class RTIODestinationUnreachable(Exception):
    artiq_builtin: bool = ...


class DMAError(Exception):
    ...


class ClockFailure(Exception):
    ...


class I2CError(Exception):
    ...


class SPIError(Exception):
    ...
