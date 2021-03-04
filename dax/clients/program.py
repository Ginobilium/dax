# mypy: no_warn_unused_ignores

import typing
import types
import os.path
import shutil
import shlex
import argparse

import artiq.tools

# No wildcard import to prevent aliasing with ``types``
from dax.experiment import DaxClient, dax_client_factory, Experiment, StringValue

import dax.base.program
import dax.util.artiq
import dax.util.output
import dax.interfaces.operation

__all__ = ['ProgramClient']


def _import_file(file_name: str) -> types.ModuleType:
    return artiq.tools.file_import(file_name, prefix='dax_program_client_')


@dax_client_factory
class ProgramClient(DaxClient, Experiment):
    """Client for dynamically loading and running a DAX program."""

    MANAGERS_KWARG = 'managers'

    def build(self, managers: typing.Any) -> None:  # type: ignore
        # Store reference to ARTIQ managers
        self._managers: typing.Any = managers

        # Obtain arguments
        self._file: str = self.get_argument(
            'file', StringValue(), tooltip='File containing the program to run or an archive with a main.py file')
        self._class: str = self.get_argument(
            'class', StringValue(''), tooltip='Class name of the program to run (optional)')
        self._arguments: str = self.get_argument(
            'arguments', StringValue(''), tooltip='Command-line arguments (format: `[KEY=PYON_VALUE ...]`)')

    def prepare(self) -> None:
        # Archive input data
        self.set_dataset('file', self._file)
        self.set_dataset('class', self._class)
        self.set_dataset('arguments', self._arguments)

        # Load the module
        module: types.ModuleType = self._load_module()

        # Obtain class
        self.logger.debug('Loading program class%s', f' "{self._class}"' if self._class else '')
        program_cls = artiq.tools.get_experiment(module,
                                                 class_name=self._class if self._class else None)
        self.logger.info(f'Loaded program "{self._file}:{program_cls.__name__}"')

        # Test class
        if not issubclass(program_cls, dax.base.program.DaxProgram):
            raise TypeError(f'Class "{self._file}:{program_cls.__name__}" is not a DAX program')

        # Get interface
        self._interface: dax.interfaces.operation.OperationInterface
        self._interface = self.registry.find_interface(
            dax.interfaces.operation.OperationInterface)  # type: ignore[misc]

        # Parse arguments
        if self._arguments:
            self.logger.debug(f'Parsing arguments: {self._arguments}')
            parser = argparse.ArgumentParser()
            parser.add_argument('args', nargs='*')
            try:
                arguments: typing.Dict[str, typing.Any] = artiq.tools.parse_arguments(
                    parser.parse_args(shlex.split(self._arguments, posix=False)).args)
            except Exception as e:
                raise RuntimeError('Exception occurred while parsing arguments') from e
        else:
            arguments = {}

        # Build the program
        self.logger.info('Building program')
        self._program: Experiment = program_cls(
            dax.util.artiq.isolate_managers(self._managers, name='program', arguments=arguments),
            core=self.core,
            interface=self._interface
        )

        # Prepare the program
        self.logger.info('Preparing program')
        self._program.prepare()

    def run(self) -> None:
        # Validate the operation interface (should be done after DAX init)
        assert dax.interfaces.operation.validate_operation_interface(self._interface)
        # Run the program
        self.logger.info('Running program')
        self._program.run()

    def analyze(self) -> None:
        # Analyze the program
        self.logger.info('Analyzing program')
        self._program.analyze()

    def _load_module(self) -> types.ModuleType:
        # Expand and check path
        file_name = os.path.expanduser(self._file)
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'No such file or path is a directory: "{file_name}"')

        if self._file.endswith('.py'):
            # Load file/module
            self.logger.debug(f'Loading program file "{file_name}"')
            return _import_file(file_name)
        else:
            # We assume that we are dealing with an archive
            self.logger.debug(f'Unpacking and loading program archive "{file_name}"')
            with dax.util.output.temp_dir() as temp_dir:
                # Unpack archive
                shutil.unpack_archive(file_name, extract_dir=temp_dir)  # Raises exception of format is not recognized
                unpacked_file_name = os.path.join(temp_dir, 'main.py')
                if not os.path.isfile(unpacked_file_name):
                    raise FileNotFoundError(f'Archive "{file_name}" does not contain a main.py file')
                return _import_file(unpacked_file_name)