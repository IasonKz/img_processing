from pathlib import Path
import shutil
import random
import csv
import re


# ============================================================
# PATHS
# ============================================================

# Folder produced by the cropping script
INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_processed_final"
)

# New folder for train / validation / test
OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_split"
)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducible split
RANDOM_SEED = 42


# ============================================================
# DATASETS
# ============================================================

POSITIONS = [
    "ViTop",
    "ViBottom",
]

LABELS = [
    "Good",
    "Bad",
]


IMAGE_EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ============================================================
# GET UNIT ID
# ============================================================

def get_unit_id(filename):
    """
    Extract a unit/sample identifier from filename.

    Example:

    C00000084_Rc03Cc10_ViBottomPostTarget_30000us_....bmp

    becomes approximately:

    C00000084_Rc03Cc10

    This means images belonging to the same physical
    unit stay in the SAME split.

    This reduces train/test data leakage.
    """

    stem = Path(filename).stem

    # Remove everything beginning with _ViTop or _ViBottom
    match = re.split(
        r"_ViTop|_ViBottom",
        stem,
        maxsplit=1
    )

    return match[0]


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):
    """
    Return only real image files.

    _debug and Missing folders are not touched because
    we explicitly read only ViTop/Good, ViTop/Bad, etc.
    """

    images = []

    if not folder.exists():
        return images

    for path in folder.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

    return sorted(images)


# ============================================================
# GROUP IMAGES BY UNIT
# ============================================================

def group_by_unit(images):
    """
    Dictionary:

        unit_id -> [image1, image2, ...]

    All images from the same unit will remain together.
    """

    groups = {}

    for image_path in images:

        unit_id = get_unit_id(
            image_path.name
        )

        groups.setdefault(
            unit_id,
            []
        )

        groups[unit_id].append(
            image_path
        )

    return groups


# ============================================================
# SPLIT GROUPS
# ============================================================

def split_groups(groups, seed):
    """
    Split by unit, not individual image.

    Returns:
        train_groups
        val_groups
        test_groups
    """

    group_ids = list(
        groups.keys()
    )

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)

    if n == 0:
        return [], [], []

    # --------------------------------------------------------
    # Approximate split counts
    # --------------------------------------------------------

    n_train = int(
        round(
            n * TRAIN_RATIO
        )
    )

    n_val = int(
        round(
            n * VALIDATION_RATIO
        )
    )

    # Test receives remainder
    n_test = (
        n
        - n_train
        - n_val
    )

    # --------------------------------------------------------
    # Safety for small datasets
    # --------------------------------------------------------

    if n >= 3:

        if n_train < 1:
            n_train = 1

        if n_val < 1:
            n_val = 1

        n_test = (
            n
            - n_train
            - n_val
        )

        if n_test < 1:

            # Take one from train if possible
            if n_train > 1:
                n_train -= 1

            elif n_val > 1:
                n_val -= 1

            n_test = (
                n
                - n_train
                - n_val
            )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_ids = group_ids[
        :n_train
    ]

    val_ids = group_ids[
        n_train:
        n_train + n_val
    ]

    test_ids = group_ids[
        n_train + n_val:
    ]

    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# COPY GROUP
# ============================================================

def copy_groups(
    group_ids,
    groups,
    destination_folder,
    manifest_rows,
    position,
    label,
    split_name
):
    """
    COPY ONLY.

    Processed source images are never moved,
    renamed or deleted.
    """

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    copied = 0

    for group_id in group_ids:

        for source_path in groups[group_id]:

            destination_path = (
                destination_folder
                / source_path.name
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            manifest_rows.append(
                {
                    "position": position,
                    "label": label,
                    "split": split_name,
                    "unit_id": group_id,
                    "source_path": str(source_path),
                    "output_path": str(destination_path),
                }
            )

            copied += 1

    return copied


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(
    position,
    label,
    seed,
    manifest_rows
):

    input_folder = (
        INPUT_ROOT
        / position
        / label
    )

    print()
    print("=" * 75)
    print(
        f"{position} / {label}"
    )
    print(
        f"Input: {input_folder}"
    )
    print("=" * 75)

    images = get_images(
        input_folder
    )

    if not images:

        print(
            "[WARNING] No images found."
        )

        return

    # --------------------------------------------------------
    # Group images from same physical unit
    # --------------------------------------------------------

    groups = group_by_unit(
        images
    )

    print(
        f"Images: {len(images)}"
    )

    print(
        f"Unique units: {len(groups)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
        test_ids
    ) = split_groups(
        groups,
        seed
    )

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------

    train_folder = (
        OUTPUT_ROOT
        / position
        / "train"
        / label
    )

    val_folder = (
        OUTPUT_ROOT
        / position
        / "validation"
        / label
    )

    test_folder = (
        OUTPUT_ROOT
        / position
        / "test"
        / label
    )

    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------

    train_count = copy_groups(
        train_ids,
        groups,
        train_folder,
        manifest_rows,
        position,
        label,
        "train"
    )

    val_count = copy_groups(
        val_ids,
        groups,
        val_folder,
        manifest_rows,
        position,
        label,
        "validation"
    )

    test_count = copy_groups(
        test_ids,
        groups,
        test_folder,
        manifest_rows,
        position,
        label,
        "test"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = (
        train_count
        + val_count
        + test_count
    )

    print()
    print(
        f"TRAIN      : {train_count}"
    )

    print(
        f"VALIDATION : {val_count}"
    )

    print(
        f"TEST       : {test_count}"
    )

    print(
        f"TOTAL      : {total}"
    )

    if total > 0:

        print()

        print(
            f"Actual ratios:"
        )

        print(
            f"  train      = "
            f"{100 * train_count / total:.1f}%"
        )

        print(
            f"  validation = "
            f"{100 * val_count / total:.1f}%"
        )

        print(
            f"  test       = "
            f"{100 * test_count / total:.1f}%"
        )


# ============================================================
# SAVE MANIFEST CSV
# ============================================================

def save_manifest(
    rows
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        OUTPUT_ROOT
        / "dataset_split.csv"
    )

    fields = [
        "position",
        "label",
        "split",
        "unit_id",
        "source_path",
        "output_path",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Manifest saved:"
    )

    print(
        csv_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check ratios
    # --------------------------------------------------------

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(
        total_ratio - 1.0
    ) > 1e-8:

        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + "
            "TEST_RATIO must equal 1.0"
        )

    print()
    print(
        "DATASET SPLITTING"
    )

    print(
        "70% TRAIN / 15% VALIDATION / 15% TEST"
    )

    print(
        "COPY ONLY - source processed images "
        "will not be modified."
    )

    print(
        f"Input:\n{INPUT_ROOT}"
    )

    print(
        f"Output:\n{OUTPUT_ROOT}"
    )


    manifest_rows = []


    # --------------------------------------------------------
    # Split each classification task / label independently
    # --------------------------------------------------------

    for position in POSITIONS:

        for label in LABELS:

            # Different deterministic seed for each pair,
            # but reproducible every run.
            pair_seed = (
                RANDOM_SEED
                + POSITIONS.index(position) * 100
                + LABELS.index(label)
            )

            process_class(
                position=position,
                label=label,
                seed=pair_seed,
                manifest_rows=manifest_rows
            )


    # --------------------------------------------------------
    # Save CSV with exact assignment
    # --------------------------------------------------------

    save_manifest(
        manifest_rows
    )


    print()
    print("=" * 75)

    print(
        "DONE"
    )

    print(
        "Processed source images were NOT modified."
    )

    print(
        f"Split dataset:\n{OUTPUT_ROOT}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()