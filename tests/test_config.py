import unittest

from src.utils.config import Settings


class SettingsTests(unittest.TestCase):
    def test_settings_are_binary_only(self) -> None:
        settings = Settings(dataset_mode="multiclass")
        self.assertEqual(settings.dataset_mode, "binary")
        self.assertEqual(settings.class_names, ["fake", "real"])


if __name__ == "__main__":
    unittest.main()
