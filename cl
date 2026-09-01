#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the already-preprocessed COMBINED ViTop + ViBottom dataset into
train / validation / test folders, using an external YAML config.

Important:
- Original images are NEVER moved, deleted, or modified.
- Each original image is assigned to exactly ONE split.
- Augmentation is performed AFTER the split, so augmented versions of one
  image can never leak into validation/test.
- By default, rotations are created only for the training set.

Expected input example:
    classification_processed_combined/
        ViTop/
            Good/
            Bad/
        ViBottom/
            Good/
            Bad/

Output example:
    classification_split_combined/
        ViTop/
            train/
                Good/
                Bad/
            validation/
                Good/
                Bad/
            test/
                Good/
                Bad/
        ViBottom/
            ...

Run:
    python split_combined_config.py --config split_combined_config.yaml

Dependencies:
    pip install pyyaml opencv-python
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import yaml


# ============================================================
# CLI / CONFIG
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split combined ViTop/ViBottom dataset using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Tuple[dict, Path]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise SystemExit(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise SystemExit("The YAML top level must be a mapping/dictionary.")

    return config, path.parent


def resolve_path(value: str, config_dir: Path) -> Path:
    """
    Supports:
      ~
      environment variables
      relative paths (relative to the YAML file)
    """
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)

    if not path.is_absolute():
        path = config_dir / path

    return path.resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


# ============================================================
# VALIDATION
# ============================================================

def validate_config(config: dict, config_dir: Path) -> dict:
    try:
        input_root = resolve_path(config["paths"]["input_root"], config_dir)
        output_root = resolve_path(config["paths"]["output_root"], config_dir)

        datasets_raw = config["datasets"]
        classes = [str(x) for x in config["classes"]]

        split_cfg = config["splits"]
        split_ratios = {
            "train": float(split_cfg["train"]),
            "validation": float(split_cfg["validation"]),
            "test": float(split_cfg["test"]),
        }

        seed = int(config.get("random_seed", 42))

        image_extensions = {
            str(ext).lower()
            if str(ext).startswith(".")
            else "." + str(ext).lower()
            for ext in config["images"]["extensions"]
        }

        augmentation_cfg = config.get("augmentation", {})
        augmentation_enabled = bool(
            augmentation_cfg.get("enabled", False)
        )
        augmentation_splits = [
            str(x)
            for x in augmentation_cfg.get(
                "apply_to_splits",
                ["train"],
            )
        ]
        rotations = [
            int(x) % 360
            for x in augmentation_cfg.get(
                "rotations_degrees",
                [90, 180, 270],
            )
        ]

        output_cfg = config.get("output", {})
        clean_output = bool(
            output_cfg.get("clean_output_before_run", False)
        )
        existing_policy = str(
            output_cfg.get("existing_files", "error")
        ).lower()
        manifest_name = str(
            output_cfg.get("manifest_csv", "split_manifest.csv")
        )
        copy_metadata = bool(
            output_cfg.get("copy_file_metadata", True)
        )

    except KeyError as exc:
        raise SystemExit(f"Missing YAML setting: {exc}") from exc

    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")

    # Strong safety rule: source and destination must not contain each other.
    if (
        input_root == output_root
        or is_relative_to(output_root, input_root)
        or is_relative_to(input_root, output_root)
    ):
        raise SystemExit(
            "Safety stop: input_root and output_root must be separate "
            "directories and neither may be inside the other."
        )

    if not datasets_raw or not isinstance(datasets_raw, list):
        raise SystemExit("datasets must be a non-empty YAML list.")

    datasets = []
    for item in datasets_raw:
        if isinstance(item, str):
            datasets.append(
                {
                    "name": item,
                    "input_folder": item,
                    "augment": True,
                }
            )
            continue

        if not isinstance(item, dict):
            raise SystemExit(
                "Each datasets entry must be either a name or a mapping."
            )

        name = str(item["name"])
        input_folder = str(item.get("input_folder", name))
        augment = bool(item.get("augment", True))

        datasets.append(
            {
                "name": name,
                "input_folder": input_folder,
                "augment": augment,
            }
        )

    if not classes:
        raise SystemExit("classes must contain at least one class.")

    if any(r < 0.0 for r in split_ratios.values()):
        raise SystemExit("Split ratios cannot be negative.")

    ratio_sum = sum(split_ratios.values())
    if abs(ratio_sum - 1.0) > 1e-9:
        raise SystemExit(
            f"Split ratios must sum to 1.0; currently they sum to {ratio_sum:.12f}."
        )

    valid_split_names = {"train", "validation", "test"}
    unknown_aug_splits = set(augmentation_splits) - valid_split_names
    if unknown_aug_splits:
        raise SystemExit(
            "augmentation.apply_to_splits contains unknown split(s): "
            + ", ".join(sorted(unknown_aug_splits))
        )

    # Only exact 90-degree rotations are allowed here.
    invalid_rotations = [
        angle
        for angle in rotations
        if angle not in {0, 90, 180, 270}
    ]
    if invalid_rotations:
        raise SystemExit(
            "rotations_degrees must contain only multiples of 90 "
            "(0, 90, 180, 270)."
        )

    # 0° is the original image, so do not create a duplicate augmentation.
    rotations = [angle for angle in rotations if angle != 0]

    if existing_policy not in {"error", "skip", "overwrite"}:
        raise SystemExit(
            "output.existing_files must be one of: error, skip, overwrite."
        )

    if not manifest_name.lower().endswith(".csv"):
        raise SystemExit("output.manifest_csv must end in .csv")

    return {
        "input_root": input_root,
        "output_root": output_root,
        "datasets": datasets,
        "classes": classes,
        "split_ratios": split_ratios,
        "seed": seed,
        "image_extensions": image_extensions,
        "augmentation_enabled": augmentation_enabled,
        "augmentation_splits": augmentation_splits,
        "rotations": rotations,
        "clean_output": clean_output,
        "existing_policy": existing_policy,
        "manifest_name": manifest_name,
        "copy_metadata": copy_metadata,
    }


# ============================================================
# IMAGE ENUMERATION
# ============================================================

def list_images(folder: Path, extensions: set[str]) -> List[Path]:
    if not folder.is_dir():
        return []

    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ],
        key=lambda p: p.name.lower(),
    )


# ============================================================
# SPLIT COUNTS
# ============================================================

def allocate_counts(
    n: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    """
    Largest-remainder allocation.

    Example:
        n = 10
        train=0.70, validation=0.15, test=0.15

    Gives integer counts that always sum exactly to n.
    """
    order = ["train", "validation", "test"]

    raw = {
        name: n * ratios[name]
        for name in order
    }

    counts = {
        name: int(raw[name])
        for name in order
    }

    remainder = n - sum(counts.values())

    priority = sorted(
        order,
        key=lambda name: (
            -(raw[name] - counts[name]),
            order.index(name),
        ),
    )

    for name in priority[:remainder]:
        counts[name] += 1

    return counts


def deterministic_shuffle(
    paths: Sequence[Path],
    base_seed: int,
    dataset_name: str,
    class_name: str,
) -> List[Path]:
    """
    Every dataset/class group gets its own deterministic shuffle.
    Changing another group therefore does not change this group's order.
    """
    items = list(paths)

    rng = random.Random()
    rng.seed(
        f"{base_seed}|{dataset_name}|{class_name}",
        version=2,
    )
    rng.shuffle(items)

    return items


def split_paths(
    paths: Sequence[Path],
    ratios: Dict[str, float],
    base_seed: int,
    dataset_name: str,
    class_name: str,
) -> Dict[str, List[Path]]:
    shuffled = deterministic_shuffle(
        paths,
        base_seed,
        dataset_name,
        class_name,
    )

    counts = allocate_counts(len(shuffled), ratios)

    n_train = counts["train"]
    n_validation = counts["validation"]

    train_end = n_train
    validation_end = n_train + n_validation

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


# ============================================================
# SAFE OUTPUT
# ============================================================

def prepare_output(cfg: dict) -> None:
    output_root: Path = cfg["output_root"]

    if cfg["clean_output"] and output_root.exists():
        print(f"[CLEAN] Removing previous output: {output_root}")
        shutil.rmtree(output_root)

    if (
        cfg["existing_policy"] == "error"
        and output_root.exists()
        and any(output_root.iterdir())
    ):
        raise SystemExit(
            f"Output folder is not empty:\n{output_root}\n\n"
            "Either choose a new output folder, set "
            "output.clean_output_before_run: true, or set "
            "output.existing_files to skip/overwrite."
        )

    output_root.mkdir(parents=True, exist_ok=True)


def copy_file(
    source: Path,
    destination: Path,
    *,
    existing_policy: str,
    copy_metadata: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if existing_policy == "skip":
            return "skipped"
        if existing_policy == "error":
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

    if copy_metadata:
        shutil.copy2(source, destination)
    else:
        shutil.copyfile(source, destination)

    return "written"


# ============================================================
# 90-DEGREE AUGMENTATION
# ============================================================

def rotate_exact_90(image, angle: int):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unsupported rotation: {angle}")


def augmented_filename(source: Path, angle: int) -> str:
    return f"{source.stem}__rot{angle}{source.suffix}"


def write_rotated_copy(
    source: Path,
    destination: Path,
    angle: int,
    *,
    existing_policy: str,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if existing_policy == "skip":
            return "skipped"
        if existing_policy == "error":
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read image for augmentation: {source}"
        )

    rotated = rotate_exact_90(image, angle)

    ok = cv2.imwrite(str(destination), rotated)

    if not ok:
        raise RuntimeError(
            f"OpenCV could not write augmented image: {destination}"
        )

    return "written"


# ============================================================
# MANIFEST
# ============================================================

MANIFEST_COLUMNS = [
    "dataset",
    "class",
    "split",
    "is_augmented",
    "rotation_degrees",
    "source_relative_path",
    "output_relative_path",
]


def rel_string(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_manifest(
    rows: List[dict],
    output_root: Path,
    manifest_name: str,
) -> Path:
    path = output_root / manifest_name

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================
# MAIN SPLITTING
# ============================================================

def process_group(
    cfg: dict,
    dataset: dict,
    class_name: str,
    manifest_rows: List[dict],
) -> Dict[str, int]:
    input_root: Path = cfg["input_root"]
    output_root: Path = cfg["output_root"]

    source_folder = (
        input_root
        / dataset["input_folder"]
        / class_name
    )

    images = list_images(
        source_folder,
        cfg["image_extensions"],
    )

    if not source_folder.is_dir():
        print(f"[MISSING] {source_folder}")
        return {
            "source": 0,
            "train": 0,
            "validation": 0,
            "test": 0,
            "augmented": 0,
        }

    assignments = split_paths(
        images,
        cfg["split_ratios"],
        cfg["seed"],
        dataset["name"],
        class_name,
    )

    summary = {
        "source": len(images),
        "train": len(assignments["train"]),
        "validation": len(assignments["validation"]),
        "test": len(assignments["test"]),
        "augmented": 0,
    }

    for split_name in ("train", "validation", "test"):

        destination_folder = (
            output_root
            / dataset["name"]
            / split_name
            / class_name
        )

        for source in assignments[split_name]:
            destination = destination_folder / source.name

            copy_file(
                source,
                destination,
                existing_policy=cfg["existing_policy"],
                copy_metadata=cfg["copy_metadata"],
            )

            manifest_rows.append(
                {
                    "dataset": dataset["name"],
                    "class": class_name,
                    "split": split_name,
                    "is_augmented": 0,
                    "rotation_degrees": 0,
                    "source_relative_path": rel_string(
                        source,
                        input_root,
                    ),
                    "output_relative_path": rel_string(
                        destination,
                        output_root,
                    ),
                }
            )

            should_augment = (
                cfg["augmentation_enabled"]
                and dataset["augment"]
                and split_name in cfg["augmentation_splits"]
            )

            if not should_augment:
                continue

            for angle in cfg["rotations"]:
                aug_destination = (
                    destination_folder
                    / augmented_filename(source, angle)
                )

                write_rotated_copy(
                    source,
                    aug_destination,
                    angle,
                    existing_policy=cfg["existing_policy"],
                )

                summary["augmented"] += 1

                manifest_rows.append(
                    {
                        "dataset": dataset["name"],
                        "class": class_name,
                        "split": split_name,
                        "is_augmented": 1,
                        "rotation_degrees": angle,
                        "source_relative_path": rel_string(
                            source,
                            input_root,
                        ),
                        "output_relative_path": rel_string(
                            aug_destination,
                            output_root,
                        ),
                    }
                )

    return summary


def main() -> None:
    args = parse_args()

    raw_config, config_dir = load_yaml(args.config)
    cfg = validate_config(raw_config, config_dir)

    prepare_output(cfg)

    print()
    print("=" * 80)
    print("COMBINED DATASET SPLIT")
    print("=" * 80)
    print(f"CONFIG:      {args.config.expanduser().resolve()}")
    print(f"INPUT ROOT:  {cfg['input_root']}")
    print(f"OUTPUT ROOT: {cfg['output_root']}")
    print()
    print(
        "SPLITS: "
        f"train={cfg['split_ratios']['train']:.3f}, "
        f"validation={cfg['split_ratios']['validation']:.3f}, "
        f"test={cfg['split_ratios']['test']:.3f}"
    )
    print(f"RANDOM SEED: {cfg['seed']}")
    print(
        "AUGMENTATION: "
        + (
            f"ON | splits={cfg['augmentation_splits']} "
            f"| rotations={cfg['rotations']}"
            if cfg["augmentation_enabled"]
            else "OFF"
        )
    )
    print("ORIGINAL FILES: READ ONLY / COPIED, NEVER MOVED")
    print("=" * 80)

    manifest_rows: List[dict] = []

    grand_source = 0
    grand_augmented = 0

    for dataset in cfg["datasets"]:
        print()
        print("-" * 80)
        print(f"DATASET: {dataset['name']}")
        print("-" * 80)

        for class_name in cfg["classes"]:
            summary = process_group(
                cfg,
                dataset,
                class_name,
                manifest_rows,
            )

            grand_source += summary["source"]
            grand_augmented += summary["augmented"]

            print(
                f"{class_name:>12}: "
                f"source={summary['source']:4d} | "
                f"train={summary['train']:4d} | "
                f"validation={summary['validation']:4d} | "
                f"test={summary['test']:4d} | "
                f"augmented={summary['augmented']:4d}"
            )

    manifest_path = write_manifest(
        manifest_rows,
        cfg["output_root"],
        cfg["manifest_name"],
    )

    print()
    print("=" * 80)
    print("DONE")
    print(f"Original images assigned: {grand_source}")
    print(f"Augmented images created: {grand_augmented}")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {cfg['output_root']}")
    print()
    print(
        "Leakage protection: originals were split FIRST; "
        "rotations were created only AFTER assignment."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
