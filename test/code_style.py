import unittest
import io
import contextlib
import os


class TestCodeStyle(unittest.TestCase):

    def test_code_style(self):
        """Test that the code in the repository conforms to PEP-8."""

        try:
            import pycodestyle  # type: ignore
        except ImportError:
            self.skipTest('pycodestyle library not available')
        else:
            # Get DAX directory
            from dax import __dax_dir__ as dax_dir

            # Get a path to the configuration file
            config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'setup.cfg')
            if not os.path.isfile(config_file):
                self.skipTest('Could not find config file')

            # Create a style object using the config file
            style = pycodestyle.StyleGuide(config_file=config_file)
            # Buffer to store stdout output
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                # Check all files
                result = style.check_files([dax_dir])

            # Format message and assert
            msg = f'\n\nCode style report:\n{buf.getvalue()}'
            self.assertEqual(result.total_errors, 0, msg)


if __name__ == '__main__':
    unittest.main()
