from __future__ import annotations

from pathlib import Path

from src.utils.config import settings


SPLITS: tuple[str, ...] = ("train", "val")


def ensure_dataset_structure(base_dir: str | Path = "data") -> list[Path]:
    root = Path(base_dir)
    created: list[Path] = []
    for split in SPLITS:
        for class_name in settings.class_names:
            path = root / split / class_name
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(path)
    return created


def main() -> None:
    created = ensure_dataset_structure()
    if not created:
        print("Dataset structure already exists.")
        return

    print("Created class folders:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()
