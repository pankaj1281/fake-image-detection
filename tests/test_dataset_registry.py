import tempfile
import unittest

from src.dataset.loader import DatasetRegistry


class DatasetRegistryTests(unittest.TestCase):
    def test_supported_dataset_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DatasetRegistry.build("cifake", temp_dir)
            self.assertEqual(config.name, "cifake")


if __name__ == "__main__":
    unittest.main()
