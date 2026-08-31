from pathlib import Path
import shutil
import random
import csv
import re


# ============================================================
# PATHS
# ============================================================

# Output from the combined cropping script
INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_processed_combined"
)

# New split dataset
OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_split"
)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Same split every time you run the script
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
# UNIT ID
# ============================================================

def get_unit_id(filename):
    """
    Example filename:

    C00000084_Rc03Cc10_ViBottomPostTarget_30000us_20260818T173126.bmp

    becomes:

    C00000084_Rc03Cc10

    This helps keep images from the same physical unit
    in the SAME dataset split.
    """

    stem = Path(filename).stem

    parts = re.split(
        r"_ViTop|_ViBottom",
        stem,
        maxsplit=1
    )

    return parts[0]


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_images(folder):
    """
    Reads only image files directly inside:

        ViTop/Good
        ViTop/Bad
        ViBottom/Good
        ViBottom/Bad

    Therefore Missing/debug folders are automatically ignored.
    """

    if not folder.exists():
        return []

    images = []

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
    Returns:

        {
            unit_id: [image1, image2, ...]
        }

    Images from the same unit stay together.
    """

    groups = {}

    for image_path in images:

        unit_id = get_unit_id(
            image_path.name
        )

        if unit_id not in groups:
            groups[unit_id] = []

        groups[unit_id].append(
            image_path
        )

    return groups


# ============================================================
# SPLIT UNIT GROUPS
# ============================================================

def split_groups(groups, seed):
    """
    Split physical units into:

        train
        validation
        test
    """

    unit_ids = list(
        groups.keys()
    )

    rng = random.Random(seed)

    rng.shuffle(
        unit_ids
    )

    n = len(unit_ids)

    if n == 0:
        return [], [], []


    # ========================================================
    # INITIAL COUNTS
    # ========================================================

    n_train = int(
        round(
            TRAIN_RATIO * n
        )
    )

    n_val = int(
        round(
            VALIDATION_RATIO * n
        )
    )


    # Test gets whatever remains
    n_test = (
        n
        - n_train
        - n_val
    )


    # ========================================================
    # SAFETY FOR SMALL DATASETS
    # ========================================================

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

            if n_train > 1:

                n_train -= 1

            elif n_val > 1:

                n_val -= 1


            n_test = (
                n
                - n_train
                - n_val
            )


    # ========================================================
    # SPLIT
    # ========================================================

    train_ids = unit_ids[
        :n_train
    ]

    val_ids = unit_ids[
        n_train:
        n_train + n_val
    ]

    test_ids = unit_ids[
        n_train + n_val:
    ]


    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# COPY GROUPS
# ============================================================

def copy_groups(
    unit_ids,
    groups,
    destination_folder,
    manifest_rows,
    position,
    label,
    split_name
):
    """
    COPY ONLY.

    Nothing is moved or deleted from the processed dataset.
    """

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0


    for unit_id in unit_ids:

        for source_path in groups[unit_id]:

            destination_path = (
                destination_folder
                / source_path.name
            )


            # =================================================
            # COPY ONLY
            # =================================================

            shutil.copy2(
                str(source_path),
                str(destination_path)
            )


            manifest_rows.append(
                {
                    "position": position,
                    "label": label,
                    "split": split_name,
                    "unit_id": unit_id,
                    "filename": source_path.name,
                    "source_path": str(source_path),
                    "output_path": str(destination_path),
                }
            )


            count += 1


    return count


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
    print("=" * 80)

    print(
        f"{position} / {label}"
    )

    print(
        f"Input: {input_folder}"
    )

    print("=" * 80)


    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = get_images(
        input_folder
    )


    if not images:

        print(
            "[WARNING] No images found."
        )

        return


    # ========================================================
    # GROUP SAME PHYSICAL UNIT
    # ========================================================

    groups = group_by_unit(
        images
    )


    print(
        f"Images       : {len(images)}"
    )

    print(
        f"Unique units : {len(groups)}"
    )


    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_ids,
        val_ids,
        test_ids
    ) = split_groups(
        groups,
        seed
    )


    # ========================================================
    # OUTPUT FOLDERS
    # ========================================================

    train_folder = (
        OUTPUT_ROOT
        / position
        / "train"
        / label
    )


    validation_folder = (
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


    # ========================================================
    # COPY
    # ========================================================

    train_count = copy_groups(
        train_ids,
        groups,
        train_folder,
        manifest_rows,
        position,
        label,
        "train"
    )


    validation_count = copy_groups(
        val_ids,
        groups,
        validation_folder,
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


    # ========================================================
    # STATISTICS
    # ========================================================

    total = (
        train_count
        + validation_count
        + test_count
    )


    print()

    print(
        f"TRAIN      : {train_count}"
    )

    print(
        f"VALIDATION : {validation_count}"
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
            "Actual image ratios:"
        )

        print(
            f"  train      = "
            f"{100.0 * train_count / total:.1f}%"
        )

        print(
            f"  validation = "
            f"{100.0 * validation_count / total:.1f}%"
        )

        print(
            f"  test       = "
            f"{100.0 * test_count / total:.1f}%"
        )


# ============================================================
# SAVE MANIFEST CSV
# ============================================================

def save_manifest(rows):

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
        "filename",
        "source_path",
        "output_path",
    ]


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()

    print(
        f"Split manifest saved:\n"
        f"{csv_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK RATIOS
    # ========================================================

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )


    if abs(
        total_ratio - 1.0
    ) > 1e-9:

        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + "
            "TEST_RATIO must equal 1.0"
        )


    print()
    print("=" * 80)

    print(
        "DATASET SPLITTING"
    )

    print(
        "TRAIN      = 70%"
    )

    print(
        "VALIDATION = 15%"
    )

    print(
        "TEST       = 15%"
    )

    print()

    print(
        "COPY ONLY"
    )

    print(
        "Missing/debug images are NOT included."
    )

    print()

    print(
        f"INPUT:\n{INPUT_ROOT}"
    )

    print()

    print(
        f"OUTPUT:\n{OUTPUT_ROOT}"
    )

    print("=" * 80)


    manifest_rows = []


    # ========================================================
    # TOP + BOTTOM
    # GOOD + BAD
    # ========================================================

    for position_index, position in enumerate(
        POSITIONS
    ):

        for label_index, label in enumerate(
            LABELS
        ):

            seed = (
                RANDOM_SEED
                + 100 * position_index
                + label_index
            )


            process_class(
                position=position,
                label=label,
                seed=seed,
                manifest_rows=manifest_rows
            )


    # ========================================================
    # SAVE EXACT ASSIGNMENT
    # ========================================================

    save_manifest(
        manifest_rows
    )


    print()
    print("=" * 80)

    print(
        "DONE"
    )

    print()

    print(
        f"Split dataset:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    print(
        "Source processed images were not modified."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()