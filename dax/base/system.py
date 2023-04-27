from __future__ import annotations  # Postponed evaluation of annotations

import abc
import logging
import re
import natsort
import typing
import collections.abc
import types
import numpy as np

from artiq import __version__ as _artiq_version
from artiq.language.types import TStr
import artiq.language.core
import artiq.language.environment
import artiq.master.worker_db

import artiq.coredevice.core
import artiq.coredevice.dma  # type: ignore[import]
import artiq.coredevice.cache  # type: ignore[import]

from dax import __version__ as _dax_version
import dax.base.exceptions
import dax.base.interface
import dax.sim.ddb
import dax.sim.device
import dax.util.logging
import dax.util.git

__all__ = ['DaxModule', 'DaxSystem', 'DaxService',
           'DaxClient', 'dax_client_factory']

_KEY_SEPARATOR: str = '.'
"""Key separator for datasets."""

_NAME_RE: typing.Pattern[str] = re.compile(r'\w+')
"""Regex for matching valid names."""


def _is_valid_name(name: str) -> bool:
    """Return true if the given name is valid."""
    assert isinstance(name, str), 'The given name should be a string'
    return bool(_NAME_RE.fullmatch(name))


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return all(_NAME_RE.fullmatch(n) for n in key.split(_KEY_SEPARATOR))


_ARTIQ_VIRTUAL_DEVICES: typing.FrozenSet[str] = frozenset(['scheduler', 'ccb'])
"""ARTIQ virtual devices."""


class DaxBase(artiq.language.environment.HasEnvironment, abc.ABC):
    """Base class for all DAX base classes."""

    kernel_invariants: typing.Set[str]
    """Set of kernel invariant attributes."""

    __logger: logging.Logger

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize DAX base.

        :param managers_or_parent: ARTIQ manager or parent of this environment
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Logger object
        dax.util.logging.decorate_logger_class(logging.getLoggerClass())
        self.__logger = logging.getLogger(self.get_identifier())
        self.update_kernel_invariants('logger')

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except dax.base.exceptions.BuildError:
            raise  # Re-raise build errors
        except Exception as e:
            # Raise a build error and add context
            raise dax.base.exceptions.BuildError(f'Build error in {self.get_identifier()}') from e
        else:
            self.logger.debug('Build finished')

    @property
    def logger(self) -> logging.Logger:
        """Get the logger object of this object.

        The logging functions are decorated as async RPC functions and can be called from kernels.

        :return: The logger object
        """
        return self.__logger

    @artiq.language.core.host_only
    def update_kernel_invariants(self, *keys: str) -> None:
        """Add one or more keys to the set of kernel invariants.

        Kernel invariants are attributes that are not changed during kernel execution.
        Marking attributes as invariant enables more aggressive compiler optimizations.

        :param keys: The keys to add to the set of kernel invariants
        """

        assert all(isinstance(k, str) for k in keys), 'All keys must be of type str'

        # Get kernel invariants using getattr() such that we do not overwrite a user-defined variable
        kernel_invariants: typing.Set[str] = getattr(self, 'kernel_invariants', set())
        # Update the set with the given keys
        self.kernel_invariants = kernel_invariants | {*keys}

    @abc.abstractmethod
    def get_identifier(self) -> str:  # pragma: no cover
        pass

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        :return: The object identifier string
        """
        return self.get_identifier()


class DaxHasKey(DaxBase, abc.ABC):
    """Intermediate base class for DAX classes that have a key."""

    __name: str
    __system_key: str
    __data_store: DaxDataStore

    def __init__(self, managers_or_parent: typing.Any, *args: typing.Any,
                 name: str, system_key: str, **kwargs: typing.Any):
        """Constructor of a DAX base class which has a system key.

        :param managers_or_parent: The manager or parent object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param name: The name of this object
        :param system_key: The unique system key, used for object identification
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        assert isinstance(name, str), 'Name must be a string'
        assert isinstance(system_key, str), 'System key must be a string'

        # Check name and system key
        if not _is_valid_name(name):
            raise ValueError(f'Invalid name "{name}" for class "{self.__class__.__name__}"')
        if not _is_valid_key(system_key) or not system_key.endswith(name):
            raise ValueError(f'Invalid system key "{system_key}" for class "{self.__class__.__name__}"')

        # Store constructor arguments as attributes
        self.__name = name
        self.__system_key = system_key

        # Call super, which will result in a call to build()
        super(DaxHasKey, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all key attributes are available
        key_attributes: typing.List[str] = ['data_store']
        if not all(hasattr(self, n) for n in key_attributes):  # hasattr() checks properties too
            msg: str = 'Missing key attributes (super().build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make key attributes kernel invariants
        self.update_kernel_invariants(*key_attributes)

    @property
    def data_store(self) -> DaxDataStore:
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    def _take_parent_key_attributes(self, parent: DaxHasKey) -> None:
        """Take key attributes from parent.

        If this object does not construct its own attributes, it should take them from their parent.
        """
        try:
            # Take attributes from parent
            self.__data_store = parent.data_store
        except AttributeError:
            parent.logger.exception('Missing key attributes (super().build() was probably not called)')
            raise

    @artiq.language.core.host_only
    def get_name(self) -> str:
        """Get the name of this component."""
        return self.__name

    @artiq.language.core.rpc
    def get_system_key(self, *keys: str) -> TStr:
        """Get the full key based on the system key.

        If no keys are provided, the system key is returned.
        If one or more keys are provided, the provided keys are appended to the system key.

        :param keys: The keys to append to the system key
        :return: The system key with provided keys appended
        :raises ValueError: Raised if any key has an invalid format
        """

        assert all(isinstance(k, str) for k in keys), 'Keys must be of type str'

        # Check if the given keys are valid
        for k in keys:
            if not _is_valid_key(k):
                raise ValueError(f'Invalid key "{k}"')

        # Return the assigned key
        return _KEY_SEPARATOR.join([self.__system_key, *keys])

    # noinspection PyTypeHints
    @artiq.language.core.rpc(flags={'async'})
    def set_dataset_sys(self, key, value, *,
                        archive=True, data_store=True):  # type: (str, typing.Any, bool, bool) -> None
        """Sets the contents of a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        :param archive Flag to archive the value
        :param data_store: Flag to archive the value in the data store
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(archive, bool), 'Archive flag must be of type bool'
        assert isinstance(data_store, bool), 'Data store flag must be of type bool'

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Set value in system dataset with extra flags
        self.logger.debug(f'System dataset key "{key}" set to value "{value}"')
        self.set_dataset(system_key, value, broadcast=True, persist=True, archive=archive)

        if data_store:
            # Archive value using the data store
            self.data_store.set(system_key, value)

    # noinspection PyTypeHints
    @artiq.language.core.rpc(flags={'async'})
    def mutate_dataset_sys(self, key, index, value, *,
                           data_store=True):  # type: (str, typing.Any, typing.Any, bool) -> None
        """Mutate an existing system dataset at the given index.

        :param key: The key of the system dataset
        :param index: The array index to mutate, slicing and multidimensional indexing allowed
        :param value: The value to store
        :param data_store: Flag to archive the value in the data store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(data_store, bool), 'Data store flag must be of type bool'

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Mutate system dataset
        self.logger.debug(f'System dataset key "{key}"[{index}] mutate to value "{value}"')
        self.mutate_dataset(system_key, index, value)

        if data_store:
            # Archive value using the data store
            self.data_store.mutate(system_key, index, value)

    # noinspection PyTypeHints
    @artiq.language.core.rpc(flags={'async'})
    def append_to_dataset_sys(self, key, value, *,
                              data_store=True):  # type: (str, typing.Any, bool) -> None
        """Append a value to a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        :param data_store: Flag to archive the value in the data store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(data_store, bool), 'Data store flag must be of type bool'

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Append value to system dataset
        self.logger.debug(f'System dataset key "{key}" append value "{value}"')
        self.append_to_dataset(system_key, value)

        if data_store:
            # Archive value using the data store
            self.data_store.append(system_key, value)

    @artiq.language.core.host_only
    def get_dataset_sys(self, key: str, default: typing.Any = artiq.language.environment.NoDefault, *,
                        fallback: typing.Any = artiq.language.environment.NoDefault,
                        archive: bool = True, data_store: bool = True) -> typing.Any:
        """Returns the contents of a system dataset.

        If the key is present, its value will be returned.
        If the key is not present and no default is provided, a :class:`KeyError` will be raised unless a fallback
        value is provided, in which case that fallback value will be returned and nothing will be written to a dataset.
        If the key is not present and a default is provided, the default value will
        be written to the dataset and the same value will be returned.

        The above behavior differs slightly from :func:`get_dataset` since it will write
        the default value to the dataset in case the key was not present.

        Values that are retrieved using this method can not be added to the kernel invariants.
        The user is responsible for adding the attribute to the list of kernel invariants.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :param fallback: The fallback value to return if the system dataset is not present and no default provided
        :param archive Flag to archive the value
        :param data_store: Flag to archive the value in the data store if the default value is used
        :return: The value of the system dataset or the default value
        :raises KeyError: Raised if the key was not present and no default or fallback was provided
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(archive, bool), 'Archive flag must be of type bool'
        assert isinstance(data_store, bool), 'Data store flag must be of type bool'

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Modify logging level of worker_db logger to suppress an unwanted warning message in get_dataset()
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        try:
            # Get value from system dataset with extra flags
            value: typing.Any = self.get_dataset(system_key, archive=archive)
        except KeyError:
            if default is artiq.language.environment.NoDefault:
                if fallback is artiq.language.environment.NoDefault:
                    # The value was not available in the system dataset and no default was provided
                    raise KeyError(f'System dataset key "{system_key}" not found') from None
                else:
                    # Use fallback value
                    self.logger.debug(f'System dataset key "{key}" not found, returning fallback value "{fallback}"')
                    value = fallback
            else:
                # If the value does not exist, write the default value to the system dataset, but do not archive yet
                self.logger.debug(f'System dataset key "{key}" set to default value "{default}"')
                self.set_dataset(system_key, default, broadcast=True, persist=True, archive=False)
                # Get the value again
                value = self.get_dataset(system_key, archive=archive)  # Should never raise a KeyError

                if data_store:
                    # Archive value using the data store
                    self.data_store.set(system_key, value)
        else:
            self.logger.debug(f'System dataset key "{key}" returned value "{value}"')
        finally:
            # Restore original logging level of worker_db logger
            artiq.master.worker_db.logger.setLevel(logging.NOTSET)

        # Return value
        return value

    @artiq.language.core.host_only
    def setattr_dataset_sys(self, key: str, default: typing.Any = artiq.language.environment.NoDefault, *,
                            fallback: typing.Any = artiq.language.environment.NoDefault,
                            data_store: bool = True, kernel_invariant: bool = True) -> None:
        """Sets the contents of a system dataset as attribute.

        If the key is present, its value will be loaded to the attribute.
        If the key is not present and no default is provided, the attribute is not set, unless a fallback
        value is provided, in which case that fallback value will be set and nothing will be written to a dataset.
        If the key is not present and a default is provided, the default value will
        be written to the dataset and the attribute will be set to the same value.

        The above behavior differs slightly from :func:`setattr_dataset` since it will never raise an exception.
        This behavior was chosen to make sure initialization can always pass, even when keys are not available.
        Exceptions will be raised when an attribute is missing while being accessed in Python
        or when a kernel is compiled that needs the attribute.

        The function :func:`hasattr` is a helper function used for conditional initialization based on
        the presence of certain attributes (i.e. when this function is used without a default value).

        Attributes set using this function will by default be added to the kernel invariants.
        It is possible to disable this behavior by setting the appropriate function parameter.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :param fallback: The fallback value to set if the system dataset is not present and no default provided
        :param data_store: Flag to archive the value in the data store if the default value is used
        :param kernel_invariant: Flag to set the attribute as kernel invariant
        :raises KeyError: Raised if the key was not present and no default or fallback was provided
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(kernel_invariant, bool), 'Kernel invariant flag must be of type bool'

        try:
            # Get the value from system dataset
            value: typing.Any = self.get_dataset_sys(key, default, fallback=fallback, data_store=data_store)
        except KeyError:
            # The value was not available in the system dataset and no default was provided, attribute will not be set
            self.logger.debug(f'System attribute "{key}" not set')
        else:
            # Set the value as attribute (reassigning is possible, required for re-loading attributes)
            setattr(self, key, value)

            if kernel_invariant:
                # Update kernel invariants
                self.update_kernel_invariants(key)

            # Debug message
            msg_postfix: str = ' (kernel invariant)' if kernel_invariant else ''
            self.logger.debug(f'System attribute "{key}" set to value "{value}"{msg_postfix}')

    @artiq.language.core.host_only
    def hasattr(self, *keys: str) -> bool:
        """Returns if this object has the given attributes.

        Helper function to check the presence of attributes when using :func:`setattr_dataset_sys`
        without a default value.

        :param keys: The attribute names to check
        :return: True if all attributes are set
        """
        assert all(isinstance(k, str) for k in keys), 'Keys must be of type str'
        return all(hasattr(self, k) for k in keys)

    @artiq.language.core.host_only
    def get_identifier(self) -> str:
        """Return the system key with the class name."""
        return f'[{self.get_system_key()}]({self.__class__.__name__})'


class DaxHasSystem(DaxHasKey, abc.ABC):
    """Intermediate base class for DAX classes that are dependent on a DAX system."""

    __D_T = typing.TypeVar('__D_T')  # Device type verification
    __core: artiq.coredevice.core.Core
    __core_dma: artiq.coredevice.dma.CoreDMA
    __core_cache: artiq.coredevice.cache.CoreCache
    __registry: DaxNameRegistry

    def __init__(self, managers_or_parent: typing.Any, *args: typing.Any,
                 name: str, system_key: str, **kwargs: typing.Any):
        """Constructor of a DAX base class which has a system.

        :param managers_or_parent: The manager or parent object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param name: The name of this object
        :param system_key: The unique system key, used for object identification
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Call super, which will result in a call to build()
        super(DaxHasSystem, self).__init__(managers_or_parent, *args,
                                           name=name, system_key=system_key, **kwargs)

        # Verify that all core attributes are available
        core_devices: typing.List[str] = ['core', 'core_dma', 'core_cache']
        core_attributes: typing.List[str] = core_devices + ['registry']
        if not all(hasattr(self, n) for n in core_attributes):  # hasattr() checks properties too
            msg: str = 'Missing core attributes (super().build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*core_devices)

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core device driver.

        :return: The core object
        """
        return self.__core

    @property
    def core_dma(self) -> artiq.coredevice.dma.CoreDMA:
        """Get the core DMA device driver.

        :return: The core DMA object
        """
        return self.__core_dma

    @property
    def core_cache(self) -> artiq.coredevice.cache.CoreCache:
        """Get the core cache device driver.

        :return: The core cache object
        """
        return self.__core_cache

    @property
    def registry(self) -> DaxNameRegistry:
        """Get the DAX registry.

        :return: The registry object
        """
        return self.__registry

    def _take_parent_core_attributes(self, parent: DaxHasSystem) -> None:
        """Take core attributes from parent.

        If this object does not construct its own core attributes, it should take them from their parent.
        """
        try:
            # Take core attributes from parent
            self.__core = parent.core
            self.__core_dma = parent.core_dma
            self.__core_cache = parent.core_cache
            self.__registry = parent.registry
        except AttributeError:
            parent.logger.exception('Missing core attributes (super().build() was probably not called)')
            raise

    @artiq.language.core.host_only
    def _init_system(self) -> None:
        """Initialize the DAX system, for dataset access, device initialization, and recording DMA traces."""

        # Initialize all children
        self.call_child_method('_init_system')

        # Initialize this object
        self.logger.debug('Initializing...')
        try:
            self.init()
        except artiq.coredevice.core.CompileError:
            # Log a fixed message, the exception message is empty
            self.logger.exception('Compilation error occurred during initialization')
            raise
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            raise
        else:
            self.logger.debug('Initialization finished')

    @artiq.language.core.host_only
    def _post_init_system(self) -> None:
        """DAX system post-initialization (e.g. obtaining DMA handles)."""

        # Post-initialize all children
        self.call_child_method('_post_init_system')

        # Post-initialize this object
        self.logger.debug('Post-initializing...')
        try:
            self.post_init()
        except artiq.coredevice.core.CompileError:
            # Log a fixed message, the exception message is empty
            self.logger.exception('Compilation error occurred during post-initialization')
            raise
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            raise
        else:
            self.logger.debug('Post-initialization finished')

    @abc.abstractmethod
    def init(self) -> None:  # pragma: no cover
        """Override this method to access the dataset (r/w), initialize devices, and record DMA traces.

        The :func:`init` function will be called when the user calls :func:`dax_init`
        in the experiment :func:`run` function.
        """
        pass

    @abc.abstractmethod
    def post_init(self) -> None:  # pragma: no cover
        """Override this method for post-initialization procedures (e.g. obtaining DMA handles).

        The :func:`post_init` function will be called when the user calls :func:`dax_init`
        in the experiment :func:`run` function.
        The :func:`post_init` function is called after all :func:`init` functions have been called.
        This function is used to perform initialization tasks that are dependent on the initialization
        of other components, for example to obtain a DMA handle.
        """
        pass

    @typing.overload
    def get_device(self, key: str) -> typing.Any:
        ...

    @typing.overload
    def get_device(self, key: str, type_: typing.Type[__D_T]) -> __D_T:
        ...

    @artiq.language.core.host_only
    def get_device(self, key: str, type_: typing.Optional[typing.Type[__D_T]] = None) -> typing.Any:
        """Get a device driver.

        Users can optionally specify an expected device type.
        If the device does not match the expected type, an exception is raised.
        Note that drivers of controllers are SiPyCo RPC objects and type checks are therefore not useful.

        Devices that are retrieved using :func:`get_device` can not be added to the kernel invariants.
        The user is responsible for adding the attribute to the list of kernel invariants.

        :param key: The key of the device to obtain
        :param type_: The expected type of the device
        :return: The requested device driver
        :raises KeyError: Raised when the device could not be obtained from the device DB
        :raises TypeError: Raised when the device does not match the expected type
        """

        assert isinstance(key, str) and key, 'Key must be of type str and not empty'
        assert isinstance(type_, type) or type_ is None, 'Type must be a type or None'

        # Debug message
        self.logger.debug(f'Requesting device "{key}"')

        try:
            # Get the unique key, which will also check the keys and aliases
            unique: str = self.registry.get_unique_device_key(key)
        except (LookupError, TypeError) as e:
            # Device was not found in the device DB
            raise KeyError(f'Device "{key}" could not be found in the device DB') from e

        # Get the device using the unique key
        device: typing.Any = super(DaxHasSystem, self).get_device(unique)

        if type_ is not None and not isinstance(device, (type_, artiq.master.worker_db.DummyDevice,
                                                         dax.sim.device.DaxSimDevice)):
            # Device has an unexpected type
            raise TypeError(f'Device "{key}" does not match the expected type')

        # Register the requested device with the unique key
        self.registry.add_device(unique, device, self)

        # Return the device
        return device

    @artiq.language.core.host_only
    def setattr_device(self, key: str, type_: typing.Optional[typing.Type[__D_T]] = None) -> None:
        """Sets a device driver as attribute.

        Users can optionally specify an expected device type.
        If the device does not match the expected type, an exception is raised.

        The attribute used to set the device driver is automatically added to the kernel invariants.

        :param key: The key of the device
        :param type_: The expected type of the device
        :raises KeyError: Raised when the device could not be obtained from the device DB
        :raises TypeError: Raised when the device does not match the expected type
        :raises ValueError: Raised if the attribute name is not valid
        :raises AttributeError: Raised if the attribute name was already assigned
        """

        # Get the device
        device = self.get_device(key, type_=type_)  # type: ignore[arg-type]

        # Set the device as attribute
        if not _is_valid_name(key):
            raise ValueError(f'Attribute name "{key}" not valid')
        if hasattr(self, key):
            raise AttributeError(f'Attribute name "{key}" was already assigned')
        setattr(self, key, device)

        # Add attribute to kernel invariants
        self.update_kernel_invariants(key)


class DaxModuleBase(DaxHasSystem, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent: typing.Any, *args: typing.Any,
                 module_name: str, module_key: str, **kwargs: typing.Any):
        """Construct the module base class.

        :param managers_or_parent: Manager or parent of this module
        :param args: Positional arguments forwarded to the :func:`build` function
        :param module_name: Name of the module
        :param module_key: Unique and complete key of this module
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Call super
        super(DaxModuleBase, self).__init__(managers_or_parent, *args,
                                            name=module_name, system_key=module_key, **kwargs)

        # Register this module
        self.registry.add_module(self)


class DaxModule(DaxModuleBase, abc.ABC):
    """Base class for DAX modules."""

    def __init__(self, managers_or_parent: DaxHasSystem, module_name: str,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX module.

        :param managers_or_parent: The parent of this module
        :param module_name: The name of this module
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check module name
        if not _is_valid_name(module_name):
            raise ValueError(f'Invalid module name "{module_name}"')
        # Check parent type
        if not isinstance(managers_or_parent, DaxHasSystem):
            raise TypeError(f'Parent of module "{module_name}" is not of type DaxHasSystem')

        # Take key and core attributes from parent
        self._take_parent_key_attributes(managers_or_parent)
        self._take_parent_core_attributes(managers_or_parent)

        # Call super
        super(DaxModule, self).__init__(managers_or_parent, *args,
                                        module_name=module_name,
                                        module_key=managers_or_parent.get_system_key(module_name), **kwargs)


class DaxSystem(DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    SYS_ID: typing.ClassVar[str]
    """Identifier of the system."""
    SYS_VER: typing.ClassVar[int]
    """Version of the system."""

    SYS_NAME: typing.ClassVar[str] = 'system'
    """System name, used as top key for modules."""
    SYS_SERVICES: typing.ClassVar[str] = 'services'
    """System services, used as top key for services."""

    CORE_KEY: typing.ClassVar[str] = 'core'
    """Key of the core device."""
    CORE_DMA_KEY: typing.ClassVar[str] = 'core_dma'
    """Key of the core DMA device."""
    CORE_CACHE_KEY: typing.ClassVar[str] = 'core_cache'
    """Key of the core cache device."""
    CORE_LOG_KEY: typing.ClassVar[typing.Optional[str]] = 'core_log'
    """Key of the core log controller."""
    DAX_INFLUX_DB_KEY: typing.ClassVar[typing.Optional[str]] = 'dax_influx_db'
    """Key of the DAX Influx DB controller."""

    DAX_INIT_TIME_KEY: typing.ClassVar[str] = 'dax_init_time'
    """DAX initialization time system dataset key."""

    __core: artiq.coredevice.core.Core
    __core_dma: artiq.coredevice.dma.CoreDMA
    __core_cache: artiq.coredevice.cache.CoreCache
    __registry: DaxNameRegistry
    __data_store: DaxDataStore
    __dax_sim_enabled: bool

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX system.

        :param managers_or_parent: The manager or parent of this system
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Validate this system class
        _validate_system_class(type(self))

        # Call super, add names, add a new registry
        super(DaxSystem, self).__init__(managers_or_parent, *args,
                                        module_name=self.SYS_NAME, module_key=self.SYS_NAME, **kwargs)

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core device driver.

        :return: The core object
        """
        return self.__core

    @property
    def core_dma(self) -> artiq.coredevice.dma.CoreDMA:
        """Get the core DMA device driver.

        :return: The core DMA object
        """
        return self.__core_dma

    @property
    def core_cache(self) -> artiq.coredevice.cache.CoreCache:
        """Get the core cache device driver.

        :return: The core cache object
        """
        return self.__core_cache

    @property
    def registry(self) -> DaxNameRegistry:
        """Get the DAX registry.

        :return: The registry object
        """
        return self.__registry

    @property
    def data_store(self) -> DaxDataStore:
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    @property
    def dax_sim_enabled(self) -> bool:
        """Bool that indicates if DAX simulation is enabled.

        :return: True if DAX simulation is enabled
        """
        return self.__sim_enabled

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Override this method to build your DAX system. (Do not forget to call ``super().build()`` first!)

        :param args: Positional arguments forwarded to the super class
        :param kwargs: Keyword arguments forwarded to the super class
        """

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxSystem, self).build(*args, **kwargs)

        # Log DAX version
        self.logger.debug(f'DAX version {_dax_version}')

        # Create registry
        self.__registry = DaxNameRegistry(self)

        try:
            # Get the virtual simulation configuration device
            self.get_device(dax.sim.ddb.DAX_SIM_CONFIG_KEY)
        except KeyError:
            # Simulation disabled
            self.__sim_enabled = False
        except artiq.master.worker_db.DeviceError:
            # Error while initializing simulation
            self.logger.exception('Failed to initialize DAX simulation')
            raise
        else:
            # Simulation enabled
            self.__sim_enabled = True
        finally:
            # Add dax_sim_enabled property as a kernel invariant
            self.update_kernel_invariants('dax_sim_enabled')

        # Core devices
        self.__core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.__core_dma = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.__core_cache = self.get_device(self.CORE_CACHE_KEY, artiq.coredevice.cache.CoreCache)

        if self.CORE_LOG_KEY is not None:
            # Verify existence of core log controller
            try:
                # Register the core log controller with the system
                self.get_device(self.CORE_LOG_KEY)
            except KeyError:
                # Core log controller was not found in the device DB
                self.logger.warning(f'Core log controller "{self.CORE_LOG_KEY}" not found in device DB')
            except artiq.master.worker_db.DeviceError:
                # Failed to create core log driver
                self.logger.warning(f'Failed to create core log driver "{self.CORE_LOG_KEY}"', exc_info=True)

        # Instantiate the data store
        if self.DAX_INFLUX_DB_KEY is not None:
            # Create an Influx DB data store
            self.__data_store = DaxDataStoreInfluxDb.get_instance(self, type(self))
        else:
            # No data store configured
            self.__data_store = DaxDataStore()

    @artiq.language.core.host_only
    def dax_init(self) -> None:
        """Initialize the DAX system.

        When initializing, first the :func:`init` function of child objects are called in hierarchical order.
        The :func:`init` function of this system is called last.
        Finally, all :func:`post_init` functions are called in the same order.
        """

        # Store system information in local archive
        self.set_dataset('dax/system_id', self.SYS_ID, archive=True)
        self.set_dataset('dax/system_version', self.SYS_VER, archive=True)
        self.set_dataset('dax/dax_version', _dax_version, archive=True)
        self.set_dataset('dax/dax_sim_enabled', self.dax_sim_enabled, archive=True)
        if dax.util.git.in_repository():
            for k, v in dax.util.git.get_repository_info().as_dict().items():
                self.set_dataset(f'dax/git_{k}', v, archive=True)

        # Perform system initialization
        self.logger.debug('Starting DAX system initialization...')
        self._init_system()
        self._post_init_system()
        self.logger.debug('Finished DAX system initialization')

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass


def _validate_system_class(system_class: typing.Type[DaxSystem]) -> None:
    """Check if this system class is correctly implemented."""

    # Check if system ID was overridden
    assert hasattr(system_class, 'SYS_ID'), 'Every DAX system class must override the SYS_ID class attribute'
    assert isinstance(system_class.SYS_ID, str), 'System ID must be of type str'
    assert _is_valid_name(system_class.SYS_ID), f'Invalid system ID "{system_class.SYS_ID}"'

    # Check if system version was overridden
    assert hasattr(system_class, 'SYS_VER'), 'Every DAX system class must override the SYS_VER class attribute'
    assert isinstance(system_class.SYS_VER, int), 'System version must be of type int'
    assert system_class.SYS_VER >= 0, 'Invalid system version, set version number larger or equal to zero'

    # Check names and keys
    assert _is_valid_name(system_class.SYS_NAME), 'System name is not a valid'
    assert _is_valid_key(system_class.SYS_SERVICES), 'System services key is not valid'


class DaxService(DaxHasSystem, abc.ABC):
    """Base class for system services."""

    SERVICE_NAME: typing.ClassVar[str]
    """The unique name of this service."""

    def __init__(self, managers_or_parent: typing.Union[DaxSystem, DaxService],
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX service base class.

        :param managers_or_parent: The manager or parent of this object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check if service name was overridden
        assert hasattr(self, 'SERVICE_NAME'), 'Every DAX service class must override the SERVICE_NAME class attribute'
        assert isinstance(self.SERVICE_NAME, str), 'Service name must be of type str'

        # Check parent type
        if not isinstance(managers_or_parent, (DaxSystem, DaxService)):
            raise TypeError(f'Parent of service "{self.SERVICE_NAME}" is not a DAX system or service')

        # Take key and core attributes from parent
        self._take_parent_key_attributes(managers_or_parent)
        self._take_parent_core_attributes(managers_or_parent)

        # Use name registry of parent to obtain a system key
        system_key: str = managers_or_parent.registry.make_service_key(self.SERVICE_NAME)

        # Call super
        super(DaxService, self).__init__(managers_or_parent, *args,
                                         name=self.SERVICE_NAME,
                                         system_key=system_key, **kwargs)

        # Register this service
        self.registry.add_service(self)


class DaxClient(DaxHasSystem, abc.ABC):
    """Base class for DAX clients.

    Clients are template experiments that will later be joined with a user-provided system.
    When the template is instantiated, the client identifies itself as the system,
    just like a regular experiment that inherits a system would do.
    Though the client is actually a child of the system and therefore does not share
    a namespace with the system.

    The client class should be decorated using the :func:`dax_client_factory` decorator.
    This decorator creates a factory function that allows users to provide their system
    to be used with this experiment template.

    Normally, a concrete client would inherit the ARTIQ :class:`Experiment` or :class:`EnvExperiment`
    class and implement the :func:`prepare`, :func:`run`, and :func:`analyze` functions to
    define an execution flow. Additionally, a :func:`build` function can be implemented to
    provide a user interface for configuring the client.

    Note that the :func:`build` function does not need to call ``super().build()``.
    The decorator will make sure all classes are build in the correct order.
    """

    DAX_INIT: typing.ClassVar[bool] = True
    """Flag if dax_init() should run for this client."""
    MANAGERS_KWARG: typing.ClassVar[typing.Optional[str]] = None
    """Pass the ARTIQ managers as a keyword argument to the :func:`build()` function."""

    def __init__(self, managers_or_parent: DaxSystem,
                 *args: typing.Any, **kwargs: typing.Any):
        """Construct the DAX client object.

        :param managers_or_parent: Manager or parent of this client
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """
        assert isinstance(self.DAX_INIT, bool), 'The DAX_INIT flag must be of type bool'
        assert self.MANAGERS_KWARG is None or isinstance(self.MANAGERS_KWARG, str), \
            'MANAGERS_KWARG must be of type str or None'

        # Check if the decorator was used
        if not isinstance(managers_or_parent, DaxSystem):
            raise TypeError(f'DAX client class {self.__class__.__name__} must be decorated with @dax_client_factory')

        # Take key and core attributes from parent
        self._take_parent_key_attributes(managers_or_parent)
        self._take_parent_core_attributes(managers_or_parent)

        # Call super and identify with system name and system key
        super(DaxClient, self).__init__(managers_or_parent, *args,
                                        name=managers_or_parent.SYS_NAME, system_key=managers_or_parent.SYS_NAME,
                                        **kwargs)

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass


class DaxNameRegistry:
    """A class for unique name registration."""

    __M_T = typing.TypeVar('__M_T', bound=DaxModuleBase)  # Module base type variable
    __S_T = typing.TypeVar('__S_T', bound=DaxService)  # Service type variable
    __I_T = typing.TypeVar('__I_T', bound=dax.base.interface.DaxInterface)  # Interface type variable
    _sys_services_key: str
    _device_db: typing.Mapping[str, typing.Any]
    _devices: typing.Dict[str, typing.Tuple[typing.Any, DaxHasSystem]]
    _modules: typing.Dict[str, DaxModuleBase]
    _services: typing.Dict[str, DaxService]

    def __init__(self, system: DaxSystem):
        """Create a new DAX name registry.

        :param system: The DAX system this registry belongs to
        """

        assert isinstance(system, DaxSystem), 'System must be of type DAX system'

        # Check system services key
        if not _is_valid_key(system.SYS_SERVICES):
            raise ValueError(f'Invalid system services key "{system.SYS_SERVICES}"')

        # Store system services key
        self._sys_services_key = system.SYS_SERVICES  # Access attribute directly
        # Store device DB (read-only)
        self._device_db = types.MappingProxyType(system.get_device_db())

        # A dict containing registered devices
        self._devices = {}
        # A dict containing registered modules
        self._modules = {}
        # A dict containing registered services
        self._services = {}

    @property
    def device_db(self) -> typing.Mapping[str, typing.Any]:
        """Return the current device DB.

        Requesting the device DB using :func:``artiq.language.environment.HasEnvironment.get_device_db()``
        is slow as it connects to the ARTIQ master to obtain the database.
        The registry caches the device DB and by using this property, the number
        of calls to the ARTIQ master can be minimized.

        :return: A mapping proxy to the current device DB
        """
        return self._device_db

    def add_module(self, module: __M_T) -> None:
        """Register a module.

        :param module: The module to register
        :raises NonUniqueRegistrationError: Raised if the module key was already registered by another module
        """

        assert isinstance(module, DaxModuleBase), 'Module is not a DAX module base'

        # Get the module key
        key: str = module.get_system_key()

        # Get the module that registered the module key (None if the key is available)
        reg_module: typing.Optional[DaxModuleBase] = self._modules.get(key)

        if reg_module is not None:
            # Key already in use by another module
            msg: str = f'Module key "{key}" was already registered by module {reg_module.get_identifier()}'
            raise dax.base.exceptions.NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    @typing.overload
    def get_module(self, key: str) -> DaxModuleBase:
        ...

    @typing.overload
    def get_module(self, key: str, type_: typing.Type[__M_T]) -> __M_T:
        ...

    def get_module(self, key: str, type_: typing.Type[DaxModuleBase] = DaxModuleBase) -> DaxModuleBase:
        """Return the requested module by key.

        :param key: The key of the module
        :param type_: The expected type of the module
        :raises KeyError: Raised if the module could not be found
        :raises TypeError: Raised if the module type does not match the expected type
        """

        assert isinstance(key, str), 'Key must be a string'

        if key not in self._modules:
            # Module was not found
            raise KeyError(f'Module "{key}" could not be found')

        # Get the module
        module: DaxModuleBase = self._modules[key]

        if not isinstance(module, type_):
            # Module does not have the correct type
            raise TypeError(f'Module "{key}" does not match the expected type')

        # Return the module
        return module

    def find_module(self, type_: typing.Type[__M_T]) -> __M_T:
        """Find a unique module that matches the requested type.

        :param type_: The type of the module
        :return: The unique module of the requested type
        :raises KeyError: Raised if no modules of the desired type were found
        :raises LookupError: Raised if more than one module of the desired type was found
        """

        # Search for all modules matching the type
        results: typing.Dict[str, DaxNameRegistry.__M_T] = self.search_modules(type_)

        if not results:
            # No modules were found
            raise KeyError(f'Could not find modules with type "{type_.__name__}"')
        elif len(results) > 1:
            # More than one module was found
            raise LookupError(f'Could not find a unique module with type "{type_.__name__}"')

        # Return the only result
        _, module = results.popitem()
        return module

    def search_modules(self, type_: typing.Type[__M_T]) -> typing.Dict[str, __M_T]:
        """Search for modules that match the requested type and return results as a dict.

        :param type_: The type of the modules
        :return: A dict with key-module pairs
        """

        assert issubclass(type_, DaxModuleBase), 'Provided type must be a subclass of DaxModuleBase'

        # Search for all modules matching the type
        results: typing.Dict[str, DaxNameRegistry.__M_T] = {k: m for k, m in self._modules.items()
                                                            if isinstance(m, type_)}

        # Return the dict with results
        return results

    def get_module_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered module keys.

        :return: A list with module keys
        """
        return natsort.natsorted(self._modules.keys())  # Natural sort the list

    def get_module_list(self) -> typing.List[DaxModuleBase]:
        """Return the list of registered modules.

        :return: A list with module objects
        """
        return list(self._modules.values())

    def get_unique_device_key(self, key: str) -> str:
        """Get the unique device key by resolving it recursively in the device DB.

        :param key: The key to resolve
        :return: The resolved unique device key
        :raises LookupError: Raised if an alias loop was detected in the trace while resolving
        :raises TypeError: Raised if a key returned an unexpected type
        :raises KeyError: Raised if a key was not found
        """

        assert isinstance(key, str), 'Key must be of type str'

        if key in _ARTIQ_VIRTUAL_DEVICES:
            # Virtual devices always have unique names
            return key
        else:
            # Resolve the unique device key
            return self._resolve_unique_device_key(key, set())

    def _resolve_unique_device_key(self, key: str, trace: typing.Set[str]) -> str:
        """Recursively resolve aliases until we find the unique device name.

        :param key: The key to resolve
        :param trace: A set with already visited keys
        :return: The resolved unique device key
        :raises LookupError: Raised if an alias loop was detected in the trace while resolving
        :raises TypeError: Raised if a key returned an unexpected type
        :raises KeyError: Raised if a key was not found
        """

        # Check if we are not stuck in a loop
        if key in trace:
            # We are in an alias loop
            raise LookupError(f'Key "{key}" caused an alias loop')
        # Add key to the trace
        trace.add(key)

        # Get value (could raise KeyError)
        v: typing.Any = self._device_db[key]

        if isinstance(v, str):
            # Recurse if we are still dealing with an alias
            return self._resolve_unique_device_key(v, trace)
        elif isinstance(v, dict):
            # We reached a dict, key must be the unique key
            return key
        else:
            # We ended up with an unexpected type
            raise TypeError(f'Key "{key}" returned an unexpected type')

    def add_device(self, key: str, device: typing.Any, parent: DaxHasSystem) -> None:
        """Register a device.

        Devices are added to the registry to ensure every device is only owned by a single parent.

        :param key: The key of the device
        :param device: The device object
        :param parent: The parent that requested the device
        :return: The requested device driver
        :raises NonUniqueRegistrationError: Raised if the device was already registered by another parent
        """

        assert isinstance(key, str), 'Device key must be a string'
        assert isinstance(parent, DaxHasSystem), 'Parent is not a DaxHasSystem type'

        if key in _ARTIQ_VIRTUAL_DEVICES:
            return  # Virtual devices always have unique names and are excluded from the registry

        # Ensure key is unique
        key = self.get_unique_device_key(key)

        # Get the device value object (None if the device was not registered before)
        device_value = self._devices.get(key)

        if device_value is not None:
            # Device was already registered
            _, reg_parent = device_value  # Unpack tuple
            msg: str = f'Device "{key}" was already registered by parent "{reg_parent.get_system_key()}"'
            raise dax.base.exceptions.NonUniqueRegistrationError(msg)

        # Add unique device key to the dict of registered devices
        self._devices[key] = (device, parent)

    def search_devices(self, type_: typing.Union[type, typing.Tuple[type, ...]]) -> typing.Set[str]:
        """Search for registered devices that match the requested type and return their keys.

        :param type_: The type of the devices
        :return: A set with unique device keys
        """

        # Search for all registered devices matching the type
        results = {k for k, (device, _) in self._devices.items() if isinstance(device, type_)}

        # Return the dict with results
        return results

    def get_device_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered device keys.

        :return: A list of unique device keys that were registered
        """
        return natsort.natsorted(self._devices.keys())  # Natural sort the list

    def get_device_parents(self) -> typing.Dict[str, DaxHasSystem]:
        """Return a dict with device keys and their corresponding parent.

        :return: A dict of unique device keys with their parent
        """
        return {k: parent for k, (_, parent) in self._devices.items()}

    def make_service_key(self, service_name: str) -> str:
        """Return the system key for a service name.

        This function can be used to generate a system key for a new service.

        :param service_name: The unique name of the service
        :return: The system key for this service
        :raises ValueError: Raised if the provided service name is not valid
        """

        # Check the given name
        assert isinstance(service_name, str), 'Service name must be a string'
        if not _is_valid_name(service_name):
            # Service name not valid
            raise ValueError(f'Invalid service name "{service_name}"')

        # Return assigned key
        return _KEY_SEPARATOR.join([self._sys_services_key, service_name])

    def add_service(self, service: __S_T) -> None:
        """Register a service.

        Services are added to the registry to ensure every service is only present once.
        Services can also be found using the registry.

        :param service: The service to register
        :raises NonUniqueRegistrationError: Raised if the service name was already registered
        """

        assert isinstance(service, DaxService), 'Service must be a DAX service'

        # Services get indexed by name (the name of a service is unique)
        key: str = service.get_name()

        # Get the service that registered with the service name (None if key is available)
        reg_service = self._services.get(key)

        if reg_service is not None:
            # Service name was already registered
            raise dax.base.exceptions.NonUniqueRegistrationError(f'Service with name "{key}" was already registered')

        # Add service to the registry
        self._services[key] = service

    def has_service(self, key: typing.Union[str, typing.Type[__S_T]]) -> bool:
        """Return if a service is available.

        Check if a service is available in this system.

        :param key: The key of the service, can be a string or the type of the service
        :return: True if the service is available
        """
        try:
            self.get_service(key)
        except KeyError:
            return False
        else:
            return True

    @typing.overload
    def get_service(self, key: str) -> DaxService:
        ...

    @typing.overload
    def get_service(self, key: typing.Type[__S_T]) -> __S_T:
        ...

    def get_service(self, key: typing.Union[str, typing.Type[__S_T]]) -> DaxService:
        """Get a service from the registry.

        Obtain a registered service.

        :param key: The key of the service, can be a string or the type of the service
        :return: The requested service
        :raises KeyError: Raised if the service is not available
        """

        assert isinstance(key, str) or issubclass(key, DaxService), 'Key must be a string or a DAX service type'

        # Obtain the key
        service_key: str = key if isinstance(key, str) else key.SERVICE_NAME

        if service_key in self._services:
            # Return the requested service
            return self._services[service_key]
        else:
            # Service was not found
            raise KeyError(f'Service "{service_key}" is not available')

    def get_service_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered service keys.

        :return: A list of service keys that were registered
        """
        return natsort.natsorted(self._services.keys())  # Natural sort the list

    def get_service_list(self) -> typing.List[DaxService]:
        """Return the list of registered services.

        :return: A list with service objects
        """
        return list(self._services.values())

    def find_interface(self, type_: typing.Type[__I_T]) -> __I_T:
        """Find a unique interface that matches the requested type.

        Note: mypy type checker does not handle pure abstract base classes correctly.
        A ``# type: ignore[misc]`` annotation on the line using this function is probably
        required to pass type checking.

        :param type_: The type of the interface
        :return: The unique interface of the requested type
        :raises KeyError: Raised if no interfaces of the desired type were found
        :raises LookupError: Raised if more than one interface of the desired type was found
        """

        # Search for all interfaces matching the type
        results = self.search_interfaces(type_)

        if not results:
            # No interfaces were found
            raise KeyError(f'Could not find interfaces with type "{type_.__name__}"')
        elif len(results) > 1:
            # More than one interface was found
            raise LookupError(f'Could not find a unique interface with type "{type_.__name__}"')

        # Return the only result
        _, interface = results.popitem()
        return interface

    def search_interfaces(self, type_: typing.Type[__I_T]) -> typing.Dict[str, __I_T]:
        """Search for interfaces that match the requested type and return results as a dict.

        Keys for services are returned as system keys to clearly distinguish modules from services.

        Note: mypy type checker does not handle pure abstract base classes correctly.
        A ``# type: ignore[misc]`` annotation on the line using this function is probably
        required to pass type checking.

        :param type_: The type of the interfaces
        :return: A dict with key-interface pairs
        """

        assert issubclass(type_, dax.base.interface.DaxInterface), 'Provided type must be a subclass of DaxInterface'

        # Search services matching the interface type and use system keys instead of service names
        results = {s.get_system_key(): typing.cast(DaxNameRegistry.__I_T, s)
                   for s in self._services.values() if isinstance(s, type_)}
        # Search modules matching the interface type
        results.update({k: typing.cast(DaxNameRegistry.__I_T, m)
                        for k, m in self._modules.items() if isinstance(m, type_)})

        # Return the dict with results
        return results

    def is_independent(self, *components: DaxHasSystem) -> bool:
        """Test if components are independent.

        Two components are independent if they can be controlled in parallel without the risk of
        conflicting device control. The following rules apply:

        - Zero or a single component is always independent.
        - For multiple components, only modules can be independent.
        - Modules are independent if none is a submodule of another

        :param components: DAX components part of this system such as modules and services
        :return: :const:`True` if the components are independent
        """
        if not all(isinstance(c, DaxHasSystem) for c in components):
            raise ValueError('One or more components are not DAX components that can be tested for independence')
        system_components = set(self._modules.values()) | set(self._services.values())
        if not all(c in system_components for c in components):
            raise ValueError('One or more components are not part of the system served by this registry')

        if len(components) <= 1:
            # A single component is always independent
            return True
        elif any(not isinstance(c, DaxModule) for c in components):
            # Everything that is not a module is by definition not independent
            return False
        else:
            # Get the keys
            keys: typing.List[str] = [c.get_system_key() for c in components]

            while keys:
                current_key = keys.pop()
                if any(k.startswith(current_key) or current_key.startswith(k) for k in keys):
                    # One module is a submodule of the other, so not independent
                    return False
            else:
                # Not matches were found, modules must be independent
                return True


class DaxDataStore:
    """Base class for the DAX data store.

    Data stores are used for long-term archiving of time series data and have
    methods that reflect the operations on ARTIQ datasets: set, mutate, and append.
    For system dataset methods, DAX automatically invokes the data store and
    no user code is required.

    The base DAX data store does not store anything and can be used
    as a placeholder object since it is not an abstract base class.
    Other DAX data store classes can inherit this class and
    override the :func:`set`, :func:`mutate`, and :func:`append` methods.
    """

    _logger: logging.Logger

    def __init__(self) -> None:
        """Construct a new DAX data store object."""
        # Create a logger object
        self._logger = logging.getLogger(f'{self.__module__}.{self.__class__.__name__}')

    @artiq.language.core.rpc(flags={'async'})
    def set(self, key, value):  # type: (str, typing.Any) -> None
        """Write a key-value into the data store.

        :param key: The key of the value
        :param value: The value to store
        """
        self._logger.debug(f'Set key "{key}" to value: "{value}"')

    @artiq.language.core.rpc(flags={'async'})
    def mutate(self, key, index, value):  # type: (str, typing.Any, typing.Any) -> None
        """Mutate a specific index of a key-value in the data store.

        :param key: The key of the value
        :param index: The index to mutate
        :param value: The value to store
        """
        self._logger.debug(f'Mutate key "{key}"[{index}] to value "{value}"')

    @artiq.language.core.rpc(flags={'async'})
    def append(self, key, value):  # type: (str, typing.Any) -> None
        """Append a value to a key-value in the data store.

        :param key: The key of the value
        :param value: The value to append
        """
        self._logger.debug(f'Append key "{key}" with value "{value}"')


class DaxDataStoreInfluxDb(DaxDataStore):
    """Influx DB DAX data store class.

    This data store connects to an Influx DB controller (see DAX comtools) to
    push data to an Influx database.
    """

    __F_T = typing.Union[bool, float, int, np.int32, np.int64, str]  # Field type variable for Influx DB supported types
    __FD_T = typing.Dict[str, __F_T]  # Field dict type variable
    __P_T = typing.Dict[str, typing.Union[str, __FD_T]]  # Point type variable

    _FIELD_TYPES: typing.ClassVar[typing.Tuple[type, ...]] = (bool, float, int, np.int32, np.int64, str)
    """Legal field types for Influx DB."""
    _NP_FIELD_TYPES: typing.ClassVar[typing.Tuple[type, ...]] = (np.int32, np.int64, np.floating, np.bool_,
                                                                 np.character)
    """Legal field types (Numpy types) for Influx DB."""

    _sys_id: str
    _index_table: typing.Dict[str, int]
    _base_tags: __FD_T
    _base_fields: __FD_T

    def __init__(self, environment: artiq.language.environment.HasEnvironment,
                 system_class: typing.Type[DaxSystem]):
        """Create a new DAX data store that uses an Influx DB backend.

        :param environment: An object which inherits ARTIQ :class:`HasEnvironment`
        :param system_class: The DAX system class this data store identifies itself with
        """

        assert isinstance(environment, artiq.language.environment.HasEnvironment), \
            'The environment parameter must be of type HasEnvironment'
        assert issubclass(system_class, DaxSystem), 'The system class must be a subclass of DaxSystem'
        assert isinstance(system_class.DAX_INFLUX_DB_KEY, str), 'The DAX Influx DB key must be of type str'
        _validate_system_class(system_class)

        # Call super
        super(DaxDataStoreInfluxDb, self).__init__()

        # Get the scheduler, which is a virtual device
        scheduler = environment.get_device('scheduler')
        if isinstance(scheduler, artiq.master.worker_db.DummyDevice):
            return  # ARTIQ is only discovering experiment classes, do not continue initialization

        # Get the Influx DB driver, this call can raise various exceptions
        self._get_driver(environment, system_class.DAX_INFLUX_DB_KEY)

        # Store values that will be used for data points later
        self._sys_id = system_class.SYS_ID
        # Initialize index table for the append function, required to emulate appending behavior
        self._index_table = {}

        # Prepare base tags
        self._base_tags = {
            'system_version': str(system_class.SYS_VER),  # Convert int version to str since tags are strings
        }

        # Prepare base fields
        self._base_fields = {
            'rid': int(scheduler.rid),
            'pipeline_name': str(scheduler.pipeline_name),
            'priority': int(scheduler.priority),
            'artiq_version': str(_artiq_version),
            'dax_version': str(_dax_version),
        }

        if isinstance(environment, DaxSystem):
            # Add DAX sim enabled flag to fields
            self._base_fields['dax_sim_enabled'] = bool(environment.dax_sim_enabled)

        if dax.util.git.in_repository():
            # Add commit hash and dirty flag to fields
            repo_info = dax.util.git.get_repository_info()
            self._base_fields['git_commit'] = repo_info.commit
            self._base_fields['git_dirty'] = repo_info.dirty

        # Add expid items to fields if keys do not exist yet and the types are appropriate
        self._base_fields.update((k, v) for k, v in scheduler.expid.items()
                                 if k not in self._base_fields and isinstance(v, self._FIELD_TYPES))

        # Debug message
        self._logger.debug(f'Initialized base fields: {self._base_fields}')

    @artiq.language.core.rpc(flags={'async'})
    def set(self, key, value):  # type: (str, typing.Any) -> None
        """Write a key-value into the Influx DB data store.

        Lists will be flattened to separate elements with an index since
        Influx DB does not support lists.

        :param key: The key of the value
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            # Write a single point
            self._write_points([self._make_point(key, value)])
        elif isinstance(value, np.ndarray) and any(np.issubdtype(value.dtype, t) for t in self._NP_FIELD_TYPES):
            # Numpy array
            if value.size:
                # If the array is not empty, write a list of points
                if value.ndim > 1:  # This if-else statement contains some redundant code, required to make mypy pass
                    points = [self._make_point(key, v, i) for i, v in np.ndenumerate(value)]
                else:
                    points = [self._make_point(key, v, i) for i, v in enumerate(value)]
                self._write_points(points)
        elif isinstance(value, collections.abc.Sequence) and all(isinstance(e, self._FIELD_TYPES) for e in value):
            # One-dimensional sequence
            if len(value):
                # If the list is not empty, write a list of points
                self._write_points([self._make_point(key, v, i) for i, v in enumerate(value)])
            if isinstance(value, collections.abc.MutableSequence):
                # Store the length of the mutable sequence for emulated appending later
                self._index_table[key] = len(value)
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not store value for key "{key}", unsupported value type for value "{value}"')

    @artiq.language.core.rpc(flags={'async'})
    def mutate(self, key, index, value):  # type: (str, typing.Any, typing.Any) -> None
        """Mutate a specified index of a key-value in the Influx DB data store.

        List structures are not supported by Influx DB and are emulated by using indices.
        The emulation only supports single-dimensional list structures.
        Hence, the index must be an integer.
        It is not checked if the key contains an actual list structure and if the index is in range.

        :param key: The key of the value
        :param index: The index to mutate
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            if isinstance(index, (int, np.int32, np.int64)):
                # One-dimensional index
                self._write_points([self._make_point(key, value, index)])
            else:
                # Non-integer index is not supported, do not raise but warn user instead
                self._logger.warning(f'Could not mutate value for key "{key}", index "{index}" not supported')
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not mutate value for key "{key}", unsupported value type for value "{value}"')

    @artiq.language.core.rpc(flags={'async'})
    def append(self, key, value):  # type: (str, typing.Any) -> None
        """Append a value to a key-value in the Influx DB data store.

        List structures are not supported by Influx DB and are emulated by using indices.
        The Influx DB data store caches the length of arrays when using :func:`set` to emulate appending.
        If the length of the array is not in the cache, the append operation can not be emulated.

        :param key: The key of the value
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            # Get the current index
            index = self._index_table.get(key)
            if index is not None:
                # Write a single point
                self._write_points([self._make_point(key, value, index)])
                # Update the index table
                self._index_table[key] += 1
            else:
                # Index unknown, can not emulate append operation
                self._logger.warning(f'Could not append value for key "{key}", no index was cached '
                                     f'and the append operation could not be emulated')
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not append value for key "{key}", unsupported value type for value "{value}"')

    def _make_point(self, key: str, value: __F_T,
                    index: typing.Union[None, int, np.int32, np.int64, typing.Tuple[int, ...]] = None) -> __P_T:
        """Make a point object from a key-value pair, optionally with an index.

        This function does not check the type of the value and the index, which should be checked before.
        Numpy integer values are automatically converted to Python int.
        """

        assert isinstance(key, str), 'Key should be of type str'

        if not _is_valid_key(key):
            # Invalid key
            raise ValueError(f'Influx DB data store received an invalid key "{key}"')

        if isinstance(value, (np.int32, np.int64)):
            # Convert Numpy int to Python int
            value = int(value)

        # Copy the base tags and fields
        tags = self._base_tags.copy()
        fields = self._base_fields.copy()

        # Split the key
        split_key = key.rsplit(_KEY_SEPARATOR, maxsplit=1)
        base: str = split_key[0] if len(split_key) == 2 else ''  # Base is empty if the key does not split

        if index is not None:
            # Add index if provided
            tags['index'] = str(index)  # Tags need to be of type str
        # Add base to tags
        tags['base'] = base
        # Add key-value to fields
        fields[key] = value  # The full key and the value are the actual field

        # Create the point object
        point: DaxDataStoreInfluxDb.__P_T = {
            'measurement': self._sys_id,
            'tags': tags,
            'fields': fields,
        }

        # Return point
        return point

    def _get_driver(self, environment: artiq.language.environment.HasEnvironment, key: str) -> None:
        """Get the required driver.

        This method was separated to allow testing without writing points.
        """
        self._influx = environment.get_device(key)  # Get the Influx DB driver, this call can raise various exceptions

    def _write_points(self, points: typing.Sequence[__P_T]) -> None:
        """Submit points to the Influx DB driver.

        This method was separated to allow testing without writing points.

        :param points: A list of points to write
        """
        self._influx.write_points(points)

    @classmethod
    def get_instance(cls, environment: DaxBase, system_class: typing.Type[DaxSystem]) -> DaxDataStore:
        """Get an instance of the Influx DB data store.

        In case of errors, this function will return a base :class:`DaxDataStore` object.

        :param environment: A :class:`DaxBase` object
        :param system_class: The DAX system class the Influx DB data store identifies itself with
        :return: A data store object
        """

        assert isinstance(environment, DaxBase), 'The given environment is not of type DaxBase'
        assert issubclass(system_class, DaxSystem), 'The given system class is not a subclass of DaxSystem'
        assert system_class.DAX_INFLUX_DB_KEY is not None, 'The given system class does not have a DAX Influx DB key'

        try:
            # Create an Influx DB data store
            return cls(environment, system_class)
        except KeyError:
            # Influx DB controller was not found in the device DB
            environment.logger.warning(f'Influx DB controller "{system_class.DAX_INFLUX_DB_KEY}" '
                                       f'not found in device DB')
            return DaxDataStore()
        except artiq.master.worker_db.DeviceError:
            # Failed to create Influx DB driver
            environment.logger.warning(f'Failed to create Influx DB driver "{system_class.DAX_INFLUX_DB_KEY}"',
                                       exc_info=True)
            return DaxDataStore()


# Note: These names should not alias with other type variable names!
__DCF_C_T = typing.TypeVar('__DCF_C_T', bound=DaxClient)  # Type variable for dax_client_factory() c (client) argument
__DCF_S_T = typing.TypeVar('__DCF_S_T', bound=DaxSystem)  # Type variable for dax_client_factory() system_type argument


def dax_client_factory(c: typing.Type[__DCF_C_T]) -> typing.Callable[[typing.Type[__DCF_S_T]], typing.Type[__DCF_C_T]]:
    """Decorator to convert a DaxClient class to a factory function for that class.

    Note that the factory function and the dynamically generated class can cause type checkers
    to raise issues. These errors and warnings can be ignored.

    :param c: The DAX client to create a factory function for
    :return: A factory for the client class that allows the client to be matched with a system
    """

    assert isinstance(c, type), 'The decorated object must be a type'
    if not issubclass(c, DaxClient):
        raise TypeError('The decorated class must be a subclass of DaxClient')
    if not issubclass(c, artiq.language.environment.Experiment):
        raise TypeError('The decorated class must be a subclass of Experiment')

    def wrapper(system_type: typing.Type[__DCF_S_T],
                *system_args: typing.Any, **system_kwargs: typing.Any) -> typing.Type[__DCF_C_T]:
        """Create a new DAX client class.

        This factory function will create a new client class for a given system type.

        :param system_type: The system type used by the client
        :param system_args: Positional arguments forwarded to the systems :func:`build` function
        :param system_kwargs: Keyword arguments forwarded to the systems :func:`build` function
        :return: A fusion of the client and system class
        :raises TypeError: Raised if the provided ``system_type`` parameter is not a subclass of :class:`DaxSystem`
        """

        # Check the system type
        assert isinstance(system_type, type), 'System type must be a type'
        if not issubclass(system_type, DaxSystem):
            raise TypeError('System type must be a subclass of DaxSystem')

        class WrapperClass(c):  # type: ignore[valid-type,misc]
            """The wrapper class that fuses the client class with the given system.

            The wrapper class extends the client class by constructing the system
            first and loading the client class afterwards using the system as the parent.
            """

            __system: DaxSystem

            def __init__(self, managers_or_parent: typing.Any,
                         *args: typing.Any, **kwargs: typing.Any):
                # Create the system
                self.__system = system_type(managers_or_parent, *system_args, **system_kwargs)

                if self.MANAGERS_KWARG is not None:
                    # Pass ARTIQ managers as a keyword argument
                    assert isinstance(self.MANAGERS_KWARG, str), 'MANAGERS_KWARG must be of type str or None'
                    kwargs[self.MANAGERS_KWARG] = managers_or_parent
                # Call constructor of the client class and give it the system as parent
                super(WrapperClass, self).__init__(self.__system, *args, **kwargs)

            def run(self) -> None:
                if self.DAX_INIT:
                    # Initialize the system
                    self.__system.dax_init()
                # Call the run method of the client class
                super(WrapperClass, self).run()

        # The factory function returns the newly constructed wrapper class
        return WrapperClass

    # Return the factory function
    return wrapper
