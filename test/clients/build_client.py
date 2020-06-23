import typing
import unittest

import artiq.coredevice.edge_counter

from dax.experiment import *
from dax.sim import enable_dax_sim
from dax.util.output import temp_dir
from dax.util.artiq_helpers import get_manager_or_parent

import dax.modules.rtio_benchmark
import dax.modules.rpc_benchmark
import dax.interfaces.detection

import dax.clients.introspect
import dax.clients.pmt_monitor
import dax.clients.rpc_benchmark
import dax.clients.rtio_benchmark
import dax.clients.system_benchmark


class _TestDetectionModule(DaxModule, dax.interfaces.detection.DetectionInterface):

    def build(self):
        self.pmt_array = [self.get_device('ec0', artiq.coredevice.edge_counter.EdgeCounter)]

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        return self.pmt_array

    def get_state_detection_threshold(self) -> int:
        return 2


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Call super
        super(_TestSystem, self).build(*args, **kwargs)
        # Create modules
        _TestDetectionModule(self, 'detection')
        dax.modules.rtio_benchmark.RtioLoopBenchmarkModule(self, 'rtio_bench', ttl_out='ttl0', ttl_in='ttl1')
        dax.modules.rpc_benchmark.RpcBenchmarkModule(self, 'rpc_bench')


class BuildClientTestCase(unittest.TestCase):
    """Test case that builds and initializes clients as a basic test."""

    _CLIENTS = [
        dax.clients.introspect.Introspect,
        dax.clients.pmt_monitor.PmtMonitor,
        dax.clients.rpc_benchmark.RpcBenchmarkLatency,
        dax.clients.rtio_benchmark.RtioBenchmarkEventThroughput,
        dax.clients.rtio_benchmark.RtioBenchmarkEventBurst,
        dax.clients.rtio_benchmark.RtioBenchmarkDmaThroughput,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyCoreRtio,
        dax.clients.rtio_benchmark.RtioBenchmarkInputBufferSize,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtioCore,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtt,
    ]
    """List of client types."""

    def test_build_client(self):
        with temp_dir():
            for client_type in self._CLIENTS:
                with self.subTest(client_type=client_type.__name__):
                    class _InstantiatedClient(client_type(_TestSystem)):
                        pass

                    # Create client
                    manager = get_manager_or_parent(enable_dax_sim(enable=True, ddb=_device_db, logging_level=30))
                    client = _InstantiatedClient(manager)
                    self.assertIsInstance(client, _InstantiatedClient)
                    # Get system
                    system = client.registry.find_module(DaxSystem)
                    self.assertIsInstance(system, _TestSystem)
                    self.assertIsNone(system.dax_init())


_device_db = {
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
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 1},
    },

    # Edge counters
    'ec0': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {'channel': 2},
    }
}

if __name__ == '__main__':
    unittest.main()