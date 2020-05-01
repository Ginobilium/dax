import unittest
import numpy as np
import logging

from dax.base.dax import *
import dax.base.dax

from dax.util.artiq_helpers import get_manager_or_parent

from artiq.coredevice.edge_counter import EdgeCounter  # type: ignore
from artiq.coredevice.ttl import TTLInOut, TTLOut  # type: ignore
from artiq.coredevice.core import Core  # type: ignore

"""Device DB for testing"""

device_db = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },

    # Generic TTL
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0},
        'comment': 'This is a fairly long comment, shown as tooltip.'
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 1},
        'comment': 'Hello World'
    },

    # Aliases
    'alias_0': 'ttl1',
    'alias_1': 'alias_0',
    'alias_2': 'alias_1',

    # Looped alias
    'loop_alias_0': 'loop_alias_0',
    'loop_alias_1': 'loop_alias_0',
    'loop_alias_2': 'loop_alias_4',
    'loop_alias_3': 'loop_alias_2',
    'loop_alias_4': 'loop_alias_3',

    # Dead aliases
    'dead_alias_0': 'this_key_does_not_exist_123',
    'dead_alias_1': 'dead_alias_0',
    'dead_alias_2': 'dead_alias_1',
}

"""Classes used for testing"""


class TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class TestModule(DaxModule):
    """Testing module."""

    def init(self):
        pass

    def post_init(self):
        pass


class TestModuleChild(TestModule):
    pass


class TestService(DaxService):
    SERVICE_NAME = 'test_service'

    def init(self):
        pass

    def post_init(self):
        pass


class TestInterface(DaxInterface):
    pass


class TestServiceChild(TestService, TestInterface):
    SERVICE_NAME = 'test_service_child'


"""Actual test cases"""


class DaxHelpersTestCase(unittest.TestCase):

    def test_valid_name(self):
        from dax.base.dax import _is_valid_name

        for n in ['foo', '_0foo', '_', '0', '_foo', 'FOO_', '0_foo']:
            # Test valid names
            self.assertTrue(_is_valid_name(n))

    def test_invalid_name(self):
        from dax.base.dax import _is_valid_name

        for n in ['', 'foo()', 'foo.bar', 'foo/', 'foo*', 'foo,', 'FOO+', 'foo-bar', 'foo/bar']:
            # Test illegal names
            self.assertFalse(_is_valid_name(n))

    def test_valid_key(self):
        from dax.base.dax import _is_valid_key

        for k in ['foo', '_0foo', '_', '0', 'foo.bar', 'foo.bar.baz', '_.0.A', 'foo0._bar']:
            # Test valid keys
            self.assertTrue(_is_valid_key(k))

    def test_invalid_key(self):
        from dax.base.dax import _is_valid_key

        for k in ['', 'foo()', 'foo,bar', 'foo/', '.foo', 'bar.', 'foo.bar.baz.']:
            # Test illegal keys
            self.assertFalse(_is_valid_key(k))

    def test_unique_device_key(self):
        from dax.base.dax import _get_unique_device_key

        # Test system and device DB
        s = TestSystem(get_manager_or_parent(device_db))
        d = s.get_device_db()

        # Test against various keys
        self.assertEqual(_get_unique_device_key(d, 'ttl0'), 'ttl0', 'Unique device key not returned correctly')
        self.assertEqual(_get_unique_device_key(d, 'alias_0'), 'ttl1',
                         'Alias key key does not return correct unique key')
        self.assertEqual(_get_unique_device_key(d, 'alias_1'), 'ttl1',
                         'Multi-alias key does not return correct unique key')
        self.assertEqual(_get_unique_device_key(d, 'alias_2'), 'ttl1',
                         'Multi-alias key does not return correct unique key')

    def test_looped_device_key(self):
        from dax.base.dax import _get_unique_device_key

        # Test system and device DB
        s = TestSystem(get_manager_or_parent(device_db))
        d = s.get_device_db()

        # Test looped alias
        loop_aliases = ['loop_alias_1', 'loop_alias_4']
        for key in loop_aliases:
            with self.assertRaises(LookupError, msg='Looped key alias did not raise'):
                _get_unique_device_key(d, key)

    def test_unavailable_device_key(self):
        from dax.base.dax import _get_unique_device_key

        # Test system and device DB
        s = TestSystem(get_manager_or_parent(device_db))
        d = s.get_device_db()

        # Test non-existing keys
        loop_aliases = ['not_existing_key_0', 'not_existing_key_1', 'dead_alias_2']
        for key in loop_aliases:
            with self.assertRaises(KeyError, msg='Non-existing key did not raise'):
                _get_unique_device_key(d, key)

    def test_virtual_device_key(self):
        from dax.base.dax import _get_unique_device_key

        # Test system and device DB
        s = TestSystem(get_manager_or_parent(device_db))
        d = s.get_device_db()
        # Test virtual devices
        virtual_devices = ['scheduler', 'ccb']
        for k in virtual_devices:
            self.assertEqual(_get_unique_device_key(d, k), k, 'Virtual device key not returned correctly')


class DaxNameRegistryTestCase(unittest.TestCase):

    def test_module(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent(device_db))
        # Registry
        r = s.registry

        # Test with no modules
        with self.assertRaises(KeyError, msg='Get non-existing module did not raise'):
            r.get_module('not_existing_key')
        with self.assertRaises(KeyError, msg='Find non-existing module did not raise'):
            r.find_module(TestModule)
        self.assertDictEqual(r.search_modules(TestModule), {}, 'Search result dict incorrect')

        # Test with one module
        t0 = TestModule(s, 'test_module')
        self.assertIs(r.get_module(t0.get_system_key()), t0, 'Returned module does not match expected module')
        with self.assertRaises(TypeError, msg='Type check in get_module() did not raise'):
            r.get_module(t0.get_system_key(), TestModuleChild)
        self.assertIs(r.find_module(TestModule), t0, 'Did not find the expected module')
        self.assertIs(r.find_module(DaxModule), t0, 'Did not find the expected module')
        with self.assertRaises(KeyError, msg='Search non-existing module did not raise'):
            r.find_module(TestModuleChild)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0]],
                             'Module key list incorrect')
        with self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError, msg='Adding module twice did not raise'):
            r.add_module(t0)
        with self.assertRaises(LookupError, msg='Adding module twice did not raise a LookupError'):
            r.add_module(t0)

        # Test with two modules
        t1 = TestModuleChild(s, 'test_module_child')
        self.assertIs(r.get_module(t1.get_system_key()), t1, 'Returned module does not match expected module')
        self.assertIs(r.get_module(t1.get_system_key(), TestModuleChild), t1,
                      'Type check in get_module() raised unexpectedly')
        self.assertIs(r.find_module(TestModuleChild), t1, 'Did not find expected module')
        with self.assertRaises(LookupError, msg='Non-unique search did not raise'):
            r.find_module(TestModule)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0, t1]],
                             'Module key list incorrect')
        self.assertDictEqual(r.search_modules(TestModule), {m.get_system_key(): m for m in [t0, t1]},
                             'Search result dict incorrect')

    def test_device(self):
        # Test system
        s = TestSystem(get_manager_or_parent(device_db))
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test core devices, which should be existing
        self.assertListEqual(r.get_device_key_list(), core_devices, 'Core devices were not found in device list')
        self.assertSetEqual(r.search_devices(Core), {'core'},
                            'Search devices did not returned the expected set of results')

    def test_service(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent(device_db))
        s0 = TestService(s)
        # Registry
        r = s.registry

        # Test adding the service again
        with self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError,
                               msg='Double service registration did not raise'):
            r.add_service(s0)

        # Test with one service
        self.assertFalse(r.has_service('foo'), 'Non-existing service did not returned false')
        self.assertFalse(r.has_service(TestServiceChild), 'Non-existing service did not returned false')
        self.assertTrue(r.has_service(TestService.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(TestService), 'Did not returned true for existing service')
        self.assertIs(r.get_service(s0.get_name()), s0, 'Did not returned expected service')
        self.assertIs(r.get_service(TestService), s0, 'Did not returned expected service')
        with self.assertRaises(KeyError, msg='Retrieving non-existing service did not raise'):
            r.get_service(TestServiceChild)
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0]],
                             'List of registered service keys incorrect')

        # Test with a second service
        s1 = TestServiceChild(s)
        self.assertTrue(r.has_service(TestServiceChild), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(TestServiceChild.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0, s1]],
                             'List of registered service keys incorrect')

    def test_interface(self):
        # Test system
        s = TestSystem(get_manager_or_parent(device_db))
        TestService(s)
        # Registry
        r = s.registry

        # Confirm that interface can not be found before adding
        with self.assertRaises(KeyError, msg='Interface not available did not raise'):
            r.find_interface(TestInterface)
        self.assertDictEqual(r.search_interfaces(TestInterface), {},
                             'Interface not available did not return an empty dict')

        # Add and test interface features
        itf = TestServiceChild(s)  # Class that implements the interface
        self.assertIs(r.find_interface(TestInterface), itf, 'Find interface did not return expected object')
        self.assertDictEqual(r.search_interfaces(TestInterface), {itf.get_system_key(): itf},
                             'Search interfaces did not return expected result')


class DaxDataStoreInfluxDbTestCase(unittest.TestCase):
    class NoWriteDataStore(dax.base.dax._DaxDataStoreInfluxDb):
        """Data store connector that does not write but a callback instead."""

        def __init__(self, callback, *args, **kwargs):
            assert callable(callback), 'Callback must be a callable function'
            self.callback = callback
            super(DaxDataStoreInfluxDbTestCase.NoWriteDataStore, self).__init__(*args, **kwargs)

        def _get_driver(self, system: DaxSystem, key: str) -> None:
            pass  # Do not obtain the driver

        def _write_points(self, points):
            # Do not write points but do a callback instead
            self.callback(points)

    def setUp(self) -> None:
        # Callback function
        def callback(points):
            for d in points:
                # Check if the types of all field values are valid
                for value in d['fields'].values():
                    self.assertIsInstance(value, (int, str, float, bool), 'Field in point has invalid type')
                # Check if the index is correct (if existing)
                self.assertIsInstance(d['tags'].get('index', ''), str, 'Index has invalid type (expected str)')

        # Test system
        self.s = TestSystem(get_manager_or_parent(device_db))
        # Special data store that skips actual writing
        self.ds = self.NoWriteDataStore(callback, self.s, 'dax_influx_db')

    def test_commit_hash(self):
        # Test if DAX commit hash was loaded, we can assume that the code was versioned (if not, the test fails)
        self.assertIsNotNone(self.ds._DAX_COMMIT, 'DAX commit hash was not loaded')
        self.assertIsInstance(self.ds._DAX_COMMIT, str, 'Unexpected type for DAX commit hash')
        # We are not sure if CWD is loaded, depends on where the test was initiated from
        self.assertIsInstance(self.ds._CWD_COMMIT, (str, type(None)), 'Unexpected type for cwd commit hash')

    def test_make_point(self):
        # Data to test against
        test_data = [
            ('k', 4),
            ('k', 0.1),
            ('k', True),
            ('k', 'value'),
            ('k.a', 7),
            ('k.b.c', 8),
            ('k.ddd', 9),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test making point
                d = self.ds._make_point(k, v)

                # Split key
                split_key = k.rsplit('.', maxsplit=1)
                base = split_key[0] if len(split_key) == 2 else ''

                # Verify point object
                self.assertEqual(base, d['tags']['base'], 'Base of key does not match tag in point object')
                self.assertIn(k, d['fields'], 'Key is not an available field in the point object')
                self.assertEqual(v, d['fields'][k], 'Field value in point object is not equal to inserted value')

    def test_make_point_index(self):
        # Data to test against
        test_data = [
            ('k', 4, None),
            ('k', 0.1, 1),
            ('k', True, np.int32(5)),
            ('k', 'value', np.int64(88)),
        ]

        for k, v, i in test_data:
            with self.subTest(k=k, v=v, i=i):
                # Test making point
                d = self.ds._make_point(k, v, i)

                if i is not None:
                    # Verify point object
                    self.assertIn('index', d['tags'], 'Index is not an available tag in the point object')
                    self.assertEqual(str(i), d['tags']['index'], 'Index of point object is not equal to inserted value')
                else:
                    # Confirm index does not exist
                    self.assertNotIn('index', d['tags'], 'Index found as tag in the point object while None was given')

    def test_set(self):
        # Data to test against
        test_data = [
            ('k', 5),
            ('k', 5),
            ('k', 7.65),
            ('k', 3.55),
            ('k', 'value'),
            ('k', False),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)

    def test_set_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad type resulted in unwanted write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', self),
            ('k', complex(3, 5)),
            ('k', complex(5)),
            ('k.a', self),
            ('kas', {1, 2, 6, 7}),
            ('kfd', {'i': 3}),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.set(k, v)

    def test_set_sequence(self):
        # Data to test against
        test_data = [
            ('k', [1, 2, 3]),
            ('k', list(range(9))),
            ('k', [str(i + 66) for i in range(5)]),
            ('k', [7.65 * i for i in range(4)]),
            ('k.a', np.arange(5)),
            ('k.a', np.empty(5)),
            ('k', [1, '2', True, 5.5]),
            ('k.a', range(7)),  # Ranges also work, though this is not specifically intended behavior
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)

    def test_set_sequence_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad sequence resulted in unwanted write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', {bool(i % 2) for i in range(5)}),  # Set should not work
            ('k', {i: float(i) for i in range(5)}),  # Dict should not work
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.set(k, v)

    def test_np_type_conversion(self):
        # Data to test against
        test_data = [
            ('k', np.int32(3)),
            ('k', np.int64(99999999)),
            ('k', np.float(4)),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)

    def test_mutate(self):
        # Data to test against
        test_data = [
            ('k', 3, 5),
            ('k', 44, 23),
            ('k', 'np.float(4)', -99),  # Negative indices are valid, though this is not specifically intended behavior
        ]

        for k, v, i in test_data:
            with self.subTest(k=k, v=v, i=i):
                # Test using the callback function
                self.ds.mutate(k, i, v)

    def test_mutate_index_np_type_conversion(self):
        # Data to test against
        test_data = [
            ('k', 3, np.int32(4)),
            ('k', 44, np.int64(-4)),
            ('k', True, np.int32(0)),
        ]

        for k, v, i in test_data:
            with self.subTest(k=k, v=v, i=i):
                # Test using the callback function
                self.ds.mutate(k, i, v)

    def test_append(self):
        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, 5),
            (key, 5),
            (key, 7.65),
            (key, 3.55),
            (key, 'value'),
            (key, False),
        ]

        # Initialize list for appending
        init_list = [4]
        self.ds.set(key, init_list)
        length = len(init_list)

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.append(k, v)
                # Test increment
                length += 1
                self.assertEqual(self.ds._index_table[key], length, 'Cached index was not updated correctly')

    def test_append_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad type resulted in unwanted write {} {}'.format(args, kwargs))

        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, self),
            (key, complex(3, 5)),
            (key, [2, 7, 4]),  # Can not append a list, only simple values
            (key, complex(5)),
        ]

        # Initialize list for appending
        self.ds.set(key, [4])

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.append(k, v)

    def test_append_not_cached(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad type resulted in unwanted write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, 1),
            (key, 'complex(3, 5)'),
            (key, 5.5),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.append(k, v)

    def test_append_cache(self):
        # Data to test against
        test_data = [
            ('k.b.c', []),
            ('k.a', list()),
            ('fds.aaa', np.zeros(0)),
            ('kfh', range(0)),
            ('ka.a', [5, 7, 3, 2]),
            ('kh.rt', np.zeros(4)),
            ('kee', range(6)),
        ]

        # Test empty cache
        self.assertDictEqual(self.ds._index_table, {}, 'Expected empty index cache table')

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Set
                self.ds.set(k, v)
                # Check if length is cached
                self.assertIn(k, self.ds._index_table, 'Expected entry in cache')
                self.assertEqual(len(v), self.ds._index_table[k], 'Cached length does not match actual list length')

    def test_empty_list(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Empty list of values resulted in unwanted write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', []),
            ('k.a', list()),
            ('k', np.zeros(0)),
            ('k', range(0)),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Store of empty sequence should never result in a write
                self.ds.set(k, v)
                # Check if length is cached
                self.assertIn(k, self.ds._index_table, 'Expected entry in cache')
                self.assertEqual(0, self.ds._index_table[k], 'Cached length does not match actual list length')


class DaxModuleBaseTestCase(unittest.TestCase):
    """Tests _DaxHasSystemBase, DaxModuleBase, DaxModule, and DaxSystem.

    The four mentioned modules are highly related and overlap mostly.
    Therefore they are all tested mutually.
    """

    def test_system_build(self):
        # A system that does not call super() in build()
        class BadTestSystem(TestSystem):
            def build(self):
                pass  # No call to super(), which is bad

        # Test if an error occurs when super() is not called in build()
        with self.assertRaises(AttributeError, msg='Not calling super.build() in user system did not raise'):
            BadTestSystem(get_manager_or_parent(device_db))

    def test_system_kernel_invariants(self):
        s = TestSystem(get_manager_or_parent(device_db))

        # No kernel invariants attribute yet
        self.assertTrue(hasattr(s, 'kernel_invariants'), 'Default kernel invariants not found')

        # Update kernel invariants
        invariant = 'foo'
        s.update_kernel_invariants(invariant)
        self.assertIn(invariant, s.kernel_invariants, 'Kernel invariants update not successful')

    def test_system_id(self):
        # Test if an error is raised when no ID is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system id did not raise'):
            DaxSystem(get_manager_or_parent(device_db))

        # Systems with bad ID
        class TestSystemBadId1(DaxSystem):
            SYS_ID = 'wrong.name'
            SYS_VER = 0

        # Systems with bad ID
        class TestSystemBadId2(DaxSystem):
            SYS_ID = ''
            SYS_VER = 0

        # Systems with bad ID
        class TestSystemBadId3(DaxSystem):
            SYS_ID = '+wrong'
            SYS_VER = 0

        for BadSystem in [TestSystemBadId1, TestSystemBadId2, TestSystemBadId3]:
            # Test if an error is raised when a bad ID is given to a system
            with self.assertRaises(AssertionError, msg='Providing bad system id did not raise'):
                BadSystem(get_manager_or_parent(device_db))

    def test_system_ver(self):
        class TestSystemNoVer(DaxSystem):
            SYS_ID = 'unittest_system'

        # Test if an error is raised when no version is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system version did not raise'):
            TestSystemNoVer(get_manager_or_parent(device_db))

        # System with bad version
        class TestSystemBadVer1(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = '1'

        # System with bad version
        class TestSystemBadVer2(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = -1

        # System with bad version
        class TestSystemBadVer3(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 1.1

        for BadSystem in [TestSystemBadVer1, TestSystemBadVer2, TestSystemBadVer3]:
            # Test if an error is raised when a bad version is given to a system
            with self.assertRaises(AssertionError, msg='Providing bad system version did not raise'):
                BadSystem(get_manager_or_parent(device_db))

        # System with version 0, which is fine
        class TestSystemVerZero(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 0

        # Test if it is possible to create a system with version 0
        TestSystemVerZero(get_manager_or_parent(device_db))

    def test_init(self):
        manager_or_parent = get_manager_or_parent(device_db)
        s = TestSystem(get_manager_or_parent(device_db))

        # Check constructor
        self.assertIsNotNone(s, 'Could not create DaxSystem')
        self.assertIsNotNone(TestModule(s, 'module_name'), 'Could not create a test module')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            TestModule(s, 'wrong!')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            TestModule(s, 'this.is.bad')
        with self.assertRaises(TypeError, msg='Providing non-DaxModuleBase parent to new module did not raise'):
            TestModule(manager_or_parent, 'module_name')

    def test_module_registration(self):
        # Check register
        s = TestSystem(get_manager_or_parent(device_db))
        t = TestModule(s, 'module_name')
        self.assertDictEqual(s.registry._modules, {m.get_system_key(): m for m in [s, t]},
                             'Dict with registered modules does not match expected content')

    def test_name(self):
        s = TestSystem(get_manager_or_parent(device_db))

        self.assertEqual(s.get_name(), TestSystem.SYS_NAME, 'Returned name did not match expected name')

    def test_system_key(self):
        s = TestSystem(get_manager_or_parent(device_db))

        self.assertEqual(s.get_system_key(), TestSystem.SYS_NAME, 'Returned key did not match expected key')

    def test_system_key_arguments(self):
        s = TestSystem(get_manager_or_parent(device_db))

        self.assertEqual(s.get_system_key('a', 'b'), '.'.join([TestSystem.SYS_NAME, 'a', 'b']),
                         'Returned key did not match expected key based on multiple components')
        k = 'string_as_key_list'
        self.assertEqual(s.get_system_key(*k), '.'.join([TestSystem.SYS_NAME, *k]),
                         'Returned key did not match expected key based on multiple components')

        n = 'test_module_name'
        t = TestModule(s, n)
        self.assertEqual(t.get_system_key(), '.'.join([TestSystem.SYS_NAME, n]),
                         'Key created for nested module did not match expected key')
        some_key = 'some_key'
        self.assertEqual(t.get_system_key(some_key), '.'.join([TestSystem.SYS_NAME, n, some_key]),
                         'System key creation derived from current module key failed')

    def test_bad_system_key_arguments(self):
        s = TestSystem(get_manager_or_parent(device_db))

        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('bad,key')
        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('good_key', 'bad,key')
        with self.assertRaises(AssertionError, msg='Creating system key with wrong key input did not raise'):
            # Intentionally wrong argument type, disabling inspection
            # noinspection PyTypeChecker
            s.get_system_key(1)

    def test_setattr_device(self):
        s = TestSystem(get_manager_or_parent(device_db))

        self.assertIsNone(s.setattr_device('ttl0'), 'setattr_device() did not return None')
        self.assertTrue(hasattr(s, 'ttl0'), 'setattr_device() did not set the attribute correctly')
        self.assertIsNone(s.setattr_device('alias_2', 'foo'), 'setattr_device() with attribute name failed')
        self.assertTrue(hasattr(s, 'foo'), 'setattr_device() with attribute name did not set attribute correctly')

    def test_get_device(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent(device_db))
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test getting devices
        self.assertIsNotNone(s.get_device('ttl0'), 'Device request with unique key failed')
        self.assertIsNotNone(s.get_device('alias_2'), 'Device request with alias failed')
        self.assertIn('ttl1', r.get_device_key_list(),
                      'Device registration did not found correct unique key for device alias')
        self.assertListEqual(r.get_device_key_list(), core_devices + ['ttl0', 'ttl1'], 'Device key list incorrect')
        with self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError,
                               msg='Double device registration did not raise when registered by unique name and alias'):
            s.get_device('alias_1')

    def test_get_device_type_check(self):
        s = TestSystem(get_manager_or_parent(device_db))

        with self.assertRaises(TypeError, msg='get_device() type check did not raise'):
            s.get_device('ttl1', EdgeCounter)  # EdgeCounter does not match the device type of ttl1

        # Correct type, should not raise
        self.assertIsNotNone(s.get_device('ttl1', TTLOut), 'get_device() type check raised unexpectedly')

    def test_search_devices(self):
        s = TestSystem(get_manager_or_parent(device_db))
        r = s.registry

        # Add devices
        self.assertIsNotNone(s.get_device('ttl0'), 'Device request with unique key failed')
        self.assertIsNotNone(s.get_device('alias_2'), 'Device request with alias failed')

        # Test if registry returns correct result
        self.assertSetEqual(r.search_devices(TTLInOut), {'ttl0'},
                            'Search devices did not returned expected result')
        self.assertSetEqual(r.search_devices(EdgeCounter), set(),
                            'Search devices did not returned expected result')
        self.assertSetEqual(r.search_devices((TTLInOut, TTLOut)), {'ttl0', 'ttl1'},
                            'Search devices did not returned expected result')

    def test_dataset(self):
        s = TestSystem(get_manager_or_parent(device_db))

        key = 'key1'
        value = [11, 12, 13]
        self.assertListEqual(s.get_dataset_sys(key, default=value), value,
                             'get_dataset_sys() did not returned the provided default value')
        self.assertListEqual(s.get_dataset_sys(key), value,
                             'get_dataset_sys() did not write the default value to the dataset')

    def test_setattr_dataset(self):
        s = TestSystem(get_manager_or_parent(device_db))

        key = 'key3'
        self.assertIsNone(s.setattr_dataset_sys(key, 10), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 10, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key5'
        s.set_dataset_sys(key, 5)
        self.assertIsNone(s.setattr_dataset_sys(key), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 5, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key4'
        self.assertIsNone(s.setattr_dataset_sys(key), 'setattr_dataset_sys() failed')
        self.assertFalse(hasattr(s, key), 'setattr_dataset_sys() set the attribute while it should not')
        self.assertNotIn(key, s.kernel_invariants,
                         'setattr_dataset_sys() did added the attribute to kernel_invariants while it should not')

    @unittest.expectedFailure
    def test_dataset_append(self):
        s = TestSystem(get_manager_or_parent(device_db))

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, []), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset_sys(key, 1), 'Appending to system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [1], 'Appending to system dataset has incorrect behavior')
        # NOTE: This test fails for unknown reasons (ARTIQ library) while real-life tests show correct behavior

    def test_dataset_append_nonempty(self):
        s = TestSystem(get_manager_or_parent(device_db))

        key = 'key4'
        self.assertIsNone(s.set_dataset(key, [0]), 'Setting new dataset failed')
        self.assertListEqual(s.get_dataset(key), [0], 'Returned dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset(key, 1), 'Appending to dataset failed')
        self.assertListEqual(s.get_dataset(key), [0, 1], 'Appending to dataset has incorrect behavior')

        key = 'key5'
        self.assertIsNone(s.set_dataset(s.get_system_key(key), [0]), 'Setting new dataset failed')
        self.assertListEqual(s.get_dataset(s.get_system_key(key)), [0],
                             'Returned dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset(s.get_system_key(key), 1), 'Appending to dataset failed')
        self.assertListEqual(s.get_dataset(s.get_system_key(key)), [0, 1],
                             'Appending to dataset has incorrect behavior')

    def test_dataset_mutate(self):
        s = TestSystem(get_manager_or_parent(device_db))

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, [0, 0, 0, 0]), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 0, 0, 0],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.mutate_dataset_sys(key, 1, 9), 'Mutating system dataset failed')
        self.assertIsNone(s.mutate_dataset_sys(key, 3, 99), 'Mutating system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 9, 0, 99], 'Mutating system dataset has incorrect behavior')

    def test_identifier(self):
        s = TestSystem(get_manager_or_parent(device_db))
        self.assertTrue(isinstance(s.get_identifier(), str), 'get_identifier() did not returned a string')


class DaxServiceTestCase(unittest.TestCase):

    def test_init(self):
        from dax.base.dax import _DaxNameRegistry

        s = TestSystem(get_manager_or_parent(device_db))

        class NoNameService(DaxService):
            def init(self) -> None:
                pass

            def post_init(self) -> None:
                pass

        with self.assertRaises(AssertionError, msg='Lack of class service name did not raise'):
            NoNameService(s)

        class WrongNameService(NoNameService):
            SERVICE_NAME = 3

        with self.assertRaises(AssertionError, msg='Wrong type service name did not raise'):
            WrongNameService(s)

        class GoodNameService(NoNameService):
            SERVICE_NAME = 'service_name'

        service = GoodNameService(s)
        self.assertIs(s.registry.get_service(GoodNameService), service,
                      'get_service() did not returned expected object')
        self.assertIn(GoodNameService.SERVICE_NAME, s.registry.get_service_key_list(),
                      'Could not find service name key in registry')

        class DuplicateNameService(NoNameService):
            SERVICE_NAME = 'service_name'

        with self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError,
                               msg='Duplicate service name registration did not raise'):
            DuplicateNameService(s)

        class GoodNameService2(NoNameService):
            SERVICE_NAME = 'service_name_2'

        self.assertTrue(GoodNameService2(service), 'Could not create new service with other service as parent')


class DaxClientTestCase(unittest.TestCase):

    def test_not_decorated(self):
        s = TestSystem(get_manager_or_parent(device_db))

        class Client(DaxClient):
            pass

        with self.assertRaises(TypeError, msg='Using client without client factory decorator did not raise'):
            Client(s)

    def test_load_super(self):
        class System(TestSystem):
            def init(self) -> None:
                self.is_initialized = True

        @dax_client_factory
        class Client(DaxClient):
            pass

        class ImplementableClient(Client(System)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(get_manager_or_parent(device_db))
        c.init()  # Is supposed to call the init() function of the system

        self.assertTrue(hasattr(c, 'is_initialized'), 'DAX system parent of client was not initialized correctly')


if __name__ == '__main__':
    unittest.main()
