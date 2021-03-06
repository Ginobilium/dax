import typing
import numpy as np

from artiq.language.core import *

from dax.sim.device import DaxSimDevice


class CoreCache(DaxSimDevice):
    __V_T = typing.List[typing.Union[int, np.integer]]  # Cache value type
    _cache: typing.Dict[str, __V_T]

    def __init__(self, dmgr: typing.Any,
                 cache: typing.Optional[typing.Dict[str, __V_T]] = None,
                 **kwargs: typing.Any):
        """Simulation driver for :class:`artiq.coredevice.cache.CoreCache`.

        :param cache: Initial state of the cache
        """

        if isinstance(cache, dict):
            assert all(isinstance(k, str) and isinstance(v, list) for k, v in cache.items())
            assert all(all(isinstance(e, (int, np.integer)) for e in list_) for list_ in cache.values())
        else:
            assert cache is None, 'Cache must be of type dict or None'

        # Call super
        super(CoreCache, self).__init__(dmgr, **kwargs)

        # Cache
        self._cache = {} if cache is None else cache.copy()

    @kernel
    def get(self, key):  # type: (str) -> __V_T
        assert isinstance(key, str), 'Key must be of type str'

        # Return value
        # NOTE: ``get()`` mimics the fact that an empty cache key is not mutable and first has to be set using ``put()``
        return self._cache.get(key, [])

    @kernel
    def put(self, key, value):  # type: (str, __V_T) -> None
        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(value, list), 'Value must be of type list'
        assert all(isinstance(e, (int, np.integer)) for e in value), 'List elements must be of type int'

        # NOTE: we can not check if the value was extracted earlier in the same kernel
        if value:
            # Store value
            self._cache[key] = value
        else:
            # Erase key
            self._cache.pop(key, None)
