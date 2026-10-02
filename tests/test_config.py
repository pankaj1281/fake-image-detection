import unittest

from src.utils.config import Settings


class SettingsTests(unittest.TestCase):
    def test_multiclass_class_order_matches_imagefolder_order(self) -> None:
        settings = Settings(dataset_mode="multiclass")
        self.assertEqual(settings.class_names, sorted(settings.class_names))
        self.assertEqual(
            settings.class_names,
            ["ai_generated", "deepfake", "diffusion_generated", "gan_generated", "manipulated", "real"],
        )


if __name__ == "__main__":
    unittest.main()
