import unittest

from artiq.language.core import rpc, portable, kernel, host_only
import artiq.experiment
import artiq.master.worker_db

import dax.util.artiq
import dax.util.output


class ArtiqTestCase(unittest.TestCase):

    def test_get_managers(self):
        # Create an experiment object using the helper get_managers() function
        with dax.util.artiq.get_managers() as managers:
            self.assertIsInstance(artiq.experiment.EnvExperiment(managers), artiq.experiment.HasEnvironment)
            self.assertIsInstance(managers, tuple)

            num_managers = 4
            self.assertEqual(len(managers), num_managers)

            # Test unpacking
            _, _, _, _ = managers
            # Test indexing
            for i in range(num_managers):
                _ = managers[i]

    def test_get_managers_dataset_db(self):
        with dax.util.output.temp_dir():
            dataset_db = 'dataset_db.pyon'
            key = 'foo'
            value = 99

            with open(dataset_db, mode='x') as f:
                # Write pyon file
                f.write(f'{{\n    "{key}": {value}\n}}')

            # Create environment
            with dax.util.artiq.get_managers(dataset_db=dataset_db) as managers:
                env = artiq.experiment.EnvExperiment(managers)
                self.assertEqual(env.get_dataset(key), value, 'Retrieved dataset did not match earlier set value')

    def test_is_kernel(self):
        self.assertFalse(dax.util.artiq.is_kernel(self._undecorated_func),
                         'Undecorated function wrongly marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._rpc_func),
                         'RPC function wrongly marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._portable_func),
                         'Portable function wrongly marked as a kernel function')
        self.assertTrue(dax.util.artiq.is_kernel(self._kernel_func),
                        'Kernel function not marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._host_only_func),
                         'Host only function wrongly marked as a kernel function')

    def test_is_portable(self):
        self.assertFalse(dax.util.artiq.is_portable(self._undecorated_func),
                         'Undecorated function wrongly marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._rpc_func),
                         'RPC function wrongly marked as a portable function')
        self.assertTrue(dax.util.artiq.is_portable(self._portable_func),
                        'Portable function not marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._kernel_func),
                         'Kernel function wrongly marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._host_only_func),
                         'Host only function wrongly marked as a portable function')

    def test_is_host_only(self):
        self.assertFalse(dax.util.artiq.is_host_only(self._undecorated_func),
                         'Undecorated function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._rpc_func),
                         'RPC function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._portable_func),
                         'Portable function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._kernel_func),
                         'Kernel function wrongly marked as a host only function')
        self.assertTrue(dax.util.artiq.is_host_only(self._host_only_func),
                        'Host only function not marked as a host only function')

    def test_process_arguments(self):
        arguments = {'foo': 1,
                     'range': artiq.experiment.RangeScan(1, 10, 9),
                     'center': artiq.experiment.CenterScan(1, 10, 9),
                     'explicit': artiq.experiment.ExplicitScan([1, 10, 9]),
                     'no': artiq.experiment.NoScan(10)}

        processed_arguments = dax.util.artiq.process_arguments(arguments)
        self.assertEqual(len(arguments), len(processed_arguments))
        self.assertIsNot(arguments, processed_arguments)
        for v in processed_arguments.values():
            self.assertNotIsInstance(v, artiq.experiment.ScanObject)
        self.assertDictEqual(processed_arguments, {k: v.describe() if isinstance(v, artiq.experiment.ScanObject) else v
                                                   for k, v in arguments.items()})

    def test_cloned_dataset_manager(self):
        with dax.util.artiq.get_managers() as managers:
            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr)
            self.assertIs(clone.ddb, managers.dataset_mgr.ddb)
            self.assertIsInstance(clone, artiq.master.worker_db.DatasetManager)

    def test_cloned_dataset_manager_non_recursive(self):
        with dax.util.artiq.get_managers() as managers:
            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr)
            with self.assertRaises(TypeError, msg='Recursive clone did not raise'):
                dax.util.artiq.ClonedDatasetManager(clone)

    def test_cloned_dataset_manager_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar'

            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
            self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
            registered_clone_key, registered_clone = clone_dict.popitem()
            self.assertEqual(registered_clone_key, name, 'Dataset manager clone name was not passed correctly')
            self.assertIs(registered_clone, clone)

    def test_cloned_dataset_manager_name_index(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar_{index}'

            clone = dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
            self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
            registered_clone_key, registered_clone = clone_dict.popitem()
            self.assertEqual(registered_clone_key, name.format(index=0),
                             'Dataset manager clone name was not passed correctly')
            self.assertIs(registered_clone, clone)

    def test_cloned_dataset_manager_unique_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foobar'

            dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)
            with self.assertRaises(LookupError, msg='Non-unique name did not raise'):
                dax.util.artiq.ClonedDatasetManager(managers.dataset_mgr, name=name)

    def test_clone_managers(self):
        with dax.util.artiq.get_managers() as managers:
            device_mgr, dataset_mgr, argument_mgr, scheduler_defaults = managers
            write_hdf5_fn = dataset_mgr.write_hdf5
            cloned = dax.util.artiq.clone_managers(managers)

            self.assertIs(device_mgr, cloned[0], 'Device manager was modified unintentionally')
            self.assertIsNot(dataset_mgr, cloned[1], 'Dataset manager was not replaced')
            self.assertIsNot(dataset_mgr.write_hdf5, write_hdf5_fn, 'write_hdf5() function was not replaced')
            self.assertIsInstance(cloned[1], dax.util.artiq.ClonedDatasetManager)
            self.assertIsNot(argument_mgr, cloned[2], 'Argument manager was not replaced')
            self.assertIsNot(scheduler_defaults, cloned[3], 'Scheduler defaults were not replaced')

    def test_clone_managers_name(self):
        with dax.util.artiq.get_managers() as managers:
            name = 'foo'
            cloned = dax.util.artiq.clone_managers(managers, name=name)

            clone_dict = getattr(managers.dataset_mgr, dax.util.artiq.ClonedDatasetManager._CLONE_DICT_KEY)
            self.assertEqual(len(clone_dict), 1, 'Unexpected number of clones in dict')
            registered_clone_key, registered_clone = clone_dict.popitem()
            self.assertEqual(registered_clone_key, name, 'Dataset manager clone name was not passed correctly')
            self.assertIs(registered_clone, cloned[1])

    def test_clone_managers_arguments(self):
        with dax.util.artiq.get_managers() as managers:
            arguments = {'foo-bar': 1, 'bar-baz': 4, 'name': 'some_name'}
            kwargs = {'foo': 4.4, 'bar': 'bar'}
            ref = arguments.copy()  # Copy a reference for usage later

            cloned = dax.util.artiq.clone_managers(managers, arguments=arguments, **kwargs)

            # Check if we did not accidentally mutated the original arguments dict
            self.assertDictEqual(ref, arguments, 'The original given arguments were mutated')

            # Update reference to match expected outcome
            ref.update(kwargs)
            self.assertDictEqual(ref, cloned[2].unprocessed_arguments, 'Arguments were not passed correctly')

    """Functions used for tests"""

    def _undecorated_func(self):
        pass

    @rpc
    def _rpc_func(self):
        pass

    @portable
    def _portable_func(self):
        pass

    @kernel
    def _kernel_func(self):
        pass

    @host_only
    def _host_only_func(self):
        pass


if __name__ == '__main__':
    unittest.main()
