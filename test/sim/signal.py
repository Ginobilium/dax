import unittest
import typing
import numpy as np

from artiq.language.core import now_mu, delay, delay_mu, parallel, sequential
from artiq.language.units import *
import artiq.coredevice.ttl  # type: ignore
import artiq.coredevice.edge_counter
import artiq.coredevice.ad9910  # type: ignore
import artiq.coredevice.ad9912  # type: ignore

from dax.experiment import DaxSystem
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager, PeekSignalManager, SignalNotSet
from dax.util.artiq_helpers import get_manager_or_parent


class PeekSignalManagerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # Create the system
        ddb = enable_dax_sim(True, _DEVICE_DB, output='peek')
        self.sys = _TestSystem(get_manager_or_parent(ddb))

        # Get the peek signal manager
        self.sm = typing.cast(PeekSignalManager, get_signal_manager())
        assert isinstance(self.sm, PeekSignalManager)

    def _test_all_not_set(self):
        for ttl in self.sys.ttls:
            for s in ['state', 'direction', 'sensitivity']:
                self.assertEqual(self.sm.peek(ttl, s), SignalNotSet)

    def test_not_set(self):
        # At zero time, all signals should be unset
        self._test_all_not_set()

    def test_peek_1(self):
        delay(1 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.input()

        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 0)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 0)
            self.assertEqual(self.sm.peek(ttl, 'state'), 'z')

    def test_peek_2(self):
        delay(1 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()

        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
            self.assertEqual(self.sm.peek(ttl, 'state'), 'x')

    def test_peek_after_delay(self):
        delay(1 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()

        delay(10 * us)

        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
            self.assertEqual(self.sm.peek(ttl, 'state'), 'x')

    def test_peek_negative_delay(self):
        delay(10 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()

        delay_mu(-1)
        self._test_all_not_set()

        delay_mu(1)
        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
            self.assertEqual(self.sm.peek(ttl, 'state'), 'x')

    def test_peek_negative_delay_arg(self):
        delay(10 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()

        for ttl in self.sys.ttls:
            for s in ['state', 'direction', 'sensitivity']:
                self.assertEqual(self.sm.peek(ttl, s, time=now_mu() - 1), SignalNotSet)

        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
            self.assertEqual(self.sm.peek(ttl, 'state'), 'x')

    def test_peek_overwrite(self):
        delay(10 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()
        for ttl in self.sys.ttls:
            ttl.input()

        for ttl in self.sys.ttls:
            self.assertEqual(self.sm.peek(ttl, 'direction'), 0)
            self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 0)
            self.assertEqual(self.sm.peek(ttl, 'state'), 'z')

    def test_peek_many_changes(self):
        delay(10 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()
        for ttl in self.sys.ttls:
            ttl.input()
        delay(3 * us)
        for ttl in self.sys.ttls:
            ttl.output()

        for ttl in self.sys.ttls:
            for i in range(10):
                delay(2 * us)
                ttl.set_o(i % 2)
                self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
                self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
                self.assertEqual(self.sm.peek(ttl, 'state'), i % 2)

    def test_peek_parallel(self):
        delay(10 * us)
        self._test_all_not_set()

        # Set direction
        for ttl in self.sys.ttls:
            ttl.output()

        for ttl in self.sys.ttls:
            with parallel:
                with sequential:
                    for i in range(10):
                        delay(2 * us)
                        ttl.set_o(i % 2)
                with sequential:
                    for i in range(10):
                        delay(2 * us)
                        self.assertEqual(self.sm.peek(ttl, 'direction'), 1)
                        self.assertEqual(self.sm.peek(ttl, 'sensitivity'), 'z')
                        self.assertEqual(self.sm.peek(ttl, 'state'), i % 2)

    def test_event(self):
        test_data = {
            self.sys.ttl0._state: [0, 1, 'x', 'X', 'z', 'Z', True, False, np.int32(0), np.int64(1)],  # bool
            self.sys.ec._count: [0, 1, 'x', 'X', 'z', 'Z', True, False, 99, -34, np.int32(655), np.int64(7)],  # int
            self.sys.ad9912._freq: [1.7, -8.2, 7.7, np.float(300)],  # float
            self.sys.core_dma._dma_record: ['foo', 'bar', None, ''],  # str
            self.sys.core_dma._dma_play: [True],  # object
        }

        for signal, values in test_data.items():
            with self.subTest(signal=signal):
                for v in values:
                    self.assertIsNone(self.sm.event(signal, v))

    def test_event_bad(self):
        test_data = {
            self.sys.ttl0._state: ['foo', np.int32(9), np.int64(-1), 0.4, None],  # bool
            self.sys.ec._count: ['foo', 0.3, object, complex(6, 7), None],  # int
            self.sys.ad9912._freq: [True, 1, object, complex(6, 7), None],  # float
            self.sys.core_dma._dma_record: [True, 1, object, complex(6, 7), 1.1],  # str
            self.sys.core_dma._dma_play: [3, 4.4, 'a', object],  # object
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    with self.assertRaises(ValueError, msg='Bad event value for signal did not raise'):
                        self.sm.event(signal, v)

    def test_bool_array(self):
        test_data = {
            self.sys.ad9910._phase_mode: [True, 1, 2, 3, 0, False, np.int64(2), 'x', 'z'],  # bool array
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    self.assertIsNone(self.sm.event(signal, v))

    def test_bool_array_bad(self):
        test_data = {
            self.sys.ad9910._phase_mode: ['foo', 0.3, object, complex(6, 7), None, 4, 9, -1, 1.0],  # bool array
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    with self.assertRaises(ValueError, msg='Bad event value for signal did not raise'):
                        self.sm.event(signal, v)


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, *args, **kwargs) -> None:
        super(_TestSystem, self).build(*args, **kwargs)

        self.ttl0 = self.get_device('ttl0', artiq.coredevice.ttl.TTLInOut)
        self.ttl1 = self.get_device('ttl1', artiq.coredevice.ttl.TTLInOut)
        self.ttls = [self.ttl0, self.ttl1]

        self.ec = self.get_device('ec', artiq.coredevice.edge_counter.EdgeCounter)

        self.ad9910 = self.get_device('ad9910', artiq.coredevice.ad9910.AD9910)
        self.ad9912 = self.get_device('ad9912', artiq.coredevice.ad9912.AD9912)


# Device DB
_DEVICE_DB = {
    # Core devices
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
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 1},
    },
    'ec': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {},
    },
    "cpld": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "spi_device": "spi_urukul1",
            "sync_device": None,
            "io_update_device": "ttl_urukul1_io_update",
            "refclk": 1e9,
            "clk_sel": 1,
            "clk_div": 3
        }
    },
    "ad9910": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "pll_en": 0,
            "chip_select": 4,
            "cpld_device": "cpld",
        }
    },
    "ad9912": {
        "type": "local",
        "module": "artiq.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "pll_n": 4,
            "chip_select": 5,
            "cpld_device": "cpld",
        }
    },
}

if __name__ == '__main__':
    unittest.main()