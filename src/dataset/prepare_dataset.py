from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from src.dataset.prepare_structure import ensure_dataset_structure
from src.utils.config import settings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _clean_split(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for class_name in settings.class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.iterdir():
            if path.is_file():
                path.unlink()


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    reset_splits: bool,
    manifest_path: Path,
) -> dict[str, dict[str, int]]:
    ensure_dataset_structure(base_dir=output_dir)
    if reset_splits:
        _clean_split(output_dir / "train")
        _clean_split(output_dir / "val")

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}
    rows: list[dict[str, str]] = []

    for class_name in settings.class_names:
        class_source = source_dir / class_name
        if not class_source.exists():
            raise FileNotFoundError(f"Missing class folder in source data: {class_source}")

        all_images = _collect_images(class_source)
        rng.shuffle(all_images)
        total = len(all_images)
        val_count = int(total * val_ratio)
        if total > 1 and val_count == 0:
            val_count = 1
        val_images = all_images[:val_count]
        train_images = all_images[val_count:]

        for split, split_images in (("train", train_images), ("val", val_images)):
            target_class_dir = output_dir / split / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(split_images):
                target_name = f"{class_name}_{index:06d}{source_path.suffix.lower()}"
                destination = target_class_dir / target_name
                shutil.copy2(source_path, destination)
                rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "file_path": str(destination),
                        "source_path": str(source_path),
                    }
                )

        summary[class_name] = {"train": len(train_images), "val": len(val_images), "total": total}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "file_path", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare balanced train/val folders and dataset manifest")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-splits", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    summary = prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        reset_splits=args.reset_splits,
        manifest_path=args.manifest,
    )

    print(f"Dataset manifest written to: {args.manifest}")
    for class_name, counts in summary.items():
        print(f"{class_name}: train={counts['train']} val={counts['val']} total={counts['total']}")


if __name__ == "__main__":
    main()
