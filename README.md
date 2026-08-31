from pathlib import Path
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka"
)

OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_processed"
)


INPUT_FOLDERS = {
    "ViTop_Good": INPUT_ROOT / "ViTop" / "Good",
    "ViTop_Bad": INPUT_ROOT / "ViTop" / "Bad",

    "ViBottom_Good": INPUT_ROOT / "ViBottom" / "Good",
    "ViBottom_Bad": INPUT_ROOT / "ViBottom" / "Bad",
}


OUTPUT_FOLDERS = {
    "ViTop_Good": OUTPUT_ROOT / "ViTop" / "Good",
    "ViTop_Bad": OUTPUT_ROOT / "ViTop" / "Bad",

    "ViBottom_Good": OUTPUT_ROOT / "ViBottom" / "Good",
    "ViBottom_Bad": OUTPUT_ROOT / "ViBottom" / "Bad",
}


# ============================================================
# GREEN DEBUG -> CROP ORIGINAL
# ============================================================

def crop_from_green_debug(original_img, debug_img):
    """
    Detect the green ROI in the debug image and crop
    the SAME region from the ORIGINAL image.

    The original image is NEVER modified.
    """

    hsv = cv2.cvtColor(debug_img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    points = cv2.findNonZero(green_mask)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    # Crop from ORIGINAL, not debug
    crop = original_img[
        y:y + h,
        x:x + w
    ].copy()

    return crop


# ============================================================
# PROCESS ONE FOLDER
# ============================================================

def process_folder(input_folder, output_folder, debug_folder):
    """
    Reads originals from input_folder.

    NEVER changes/deletes/moves the originals.

    Processed images are saved only inside output_folder.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    debug_folder = Path(debug_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_extensions = {
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    }

    for original_path in input_folder.iterdir():

        if (
            not original_path.is_file()
            or original_path.suffix.lower() not in image_extensions
        ):
            continue

        # ----------------------------------------------------
        # READ ORIGINAL
        # READ ONLY — nothing is written here
        # ----------------------------------------------------

        original_img = cv2.imread(
            str(original_path),
            cv2.IMREAD_UNCHANGED
        )

        if original_img is None:
            print(
                f"[ERROR] Cannot read original: "
                f"{original_path}"
            )
            continue

        # ----------------------------------------------------
        # FIND MATCHING DEBUG IMAGE
        # ----------------------------------------------------

        debug_path = debug_folder / original_path.name

        if not debug_path.exists():
            print(
                f"[WARNING] Debug not found: "
                f"{debug_path}"
            )
            continue

        debug_img = cv2.imread(
            str(debug_path),
            cv2.IMREAD_COLOR
        )

        if debug_img is None:
            print(
                f"[WARNING] Cannot read debug: "
                f"{debug_path}"
            )
            continue

        # ----------------------------------------------------
        # CROP ORIGINAL USING GREEN DEBUG AREA
        # ----------------------------------------------------

        crop = crop_from_green_debug(
            original_img,
            debug_img
        )

        if crop is None:
            print(
                f"[WARNING] Green ROI not found: "
                f"{original_path.name}"
            )
            continue

        # ----------------------------------------------------
        # SAVE ONLY TO NEW PROCESSED FOLDER
        # ----------------------------------------------------

        output_path = (
            output_folder
            / original_path.name
        )

        cv2.imwrite(
            str(output_path),
            crop
        )

        print(
            f"[OK] {original_path.name}"
            f" -> {output_path}"
        )


# ============================================================
# PROCESS ALL 4 DATASETS
# ============================================================

for name, input_folder in INPUT_FOLDERS.items():

    output_folder = OUTPUT_FOLDERS[name]

    # Change this only if your existing debug folder
    # has another exact location/name.
    debug_folder = (
        input_folder
        / "debug"
    )

    print()
    print("=" * 60)
    print(f"Processing: {name}")
    print(f"Input:      {input_folder}")
    print(f"Debug:      {debug_folder}")
    print(f"Output:     {output_folder}")
    print("=" * 60)

    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        debug_folder=debug_folder
    )


print()
print("DONE.")
print("Original images were NOT modified.")
print(f"Processed images are in: {OUTPUT_ROOT}")  