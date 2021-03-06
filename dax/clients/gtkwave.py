from dax.experiment import *
import dax.util.gtkwave

__all__ = ['GTKWaveSaveGenerator']


@dax_client_factory
class GTKWaveSaveGenerator(DaxClient, Experiment):
    """GTKWave save file generator."""

    DAX_INIT = False
    """Disable DAX init."""

    def prepare(self) -> None:
        # Get the system
        system = self.registry.find_module(DaxSystem)
        # Create the GTKWave save generator util
        dax.util.gtkwave.GTKWSaveGenerator(system)

    def run(self) -> None:
        pass
