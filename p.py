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


DATASETS = [
    ("ViTop", "Good"),
    ("ViTop", "Bad"),
    ("ViBottom", "Good"),
    ("ViBottom", "Bad"),
]


# ============================================================
# SETTINGS
# ============================================================

# Small background around detected target
PADDING_RATIO = 0.05

# Target is expected roughly around the center
CENTER_WEIGHT = 3.0

# Approximate acceptable target dimensions relative to image
MIN_SIZE_RATIO = 0.15
MAX_SIZE_RATIO = 0.65

IMAGE_EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPER: convert image to 8 bit ONLY for detection/debug
# Original image remains untouched.
# ============================================================

def to_uint8(img):

    if img.dtype == np.uint8:
        return img.copy()

    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)

    if max_val <= min_val:
        max_val = min_val + 1

    normalized = np.clip(
        (img.astype(np.float32) - min_val)
        / (max_val - min_val),
        0,
        1
    )

    return (normalized * 255).astype(np.uint8)


# ============================================================
# DETECT CENTRAL TARGET
# ============================================================

def detect_target(original_img):
    """
    Detect central approximately-square target.

    Returns:
        x1, y1, x2, y2

    These SAME coordinates are used for:
        1. green debug rectangle
        2. final crop from ORIGINAL image
    """

    # --------------------------------------------------------
    # Detection copy
    # --------------------------------------------------------

    work = to_uint8(original_img)

    if work.ndim == 3:
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    else:
        gray = work

    H, W = gray.shape[:2]

    image_cx = W / 2
    image_cy = H / 2

    # --------------------------------------------------------
    # Slight denoising
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Improve local contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    # --------------------------------------------------------
    # Edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        enhanced,
        30,
        100
    )

    # --------------------------------------------------------
    # Connect broken target boundaries
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -1e9

    # ========================================================
    # SCORE CONTOURS
    # ========================================================

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        wr = w / W
        hr = h / H

        # Reject tiny contours
        if (
            wr < MIN_SIZE_RATIO
            or hr < MIN_SIZE_RATIO
        ):
            continue

        # Reject enormous contours / image borders
        if (
            wr > MAX_SIZE_RATIO
            or hr > MAX_SIZE_RATIO
        ):
            continue

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        aspect = w / float(h)

        if aspect < 0.55 or aspect > 1.8:
            continue

        # ----------------------------------------------------
        # Center position
        # ----------------------------------------------------

        cx = x + w / 2
        cy = y + h / 2

        dx = (cx - image_cx) / W
        dy = (cy - image_cy) / H

        center_distance = np.sqrt(
            dx * dx + dy * dy
        )

        center_score = 1.0 - center_distance

        # ----------------------------------------------------
        # Squareness
        # ----------------------------------------------------

        square_score = 1.0 - abs(
            np.log(aspect)
        )

        # ----------------------------------------------------
        # Rectangularity
        # ----------------------------------------------------

        rect_area = w * h

        rectangularity = (
            area / rect_area
            if rect_area > 0
            else 0
        )

        # ----------------------------------------------------
        # Prefer reasonable target size
        # Target roughly ~1/3 of image width/height
        # ----------------------------------------------------

        target_ratio = 0.33

        size_error = (
            abs(wr - target_ratio)
            +
            abs(hr - target_ratio)
        )

        size_score = 1.0 - size_error

        # ----------------------------------------------------
        # Total score
        # ----------------------------------------------------

        score = (
            CENTER_WEIGHT * center_score
            + 1.5 * square_score
            + 1.0 * rectangularity
            + 1.5 * size_score
        )

        if score > best_score:
            best_score = score
            best_contour = contour


    # ========================================================
    # FALLBACK
    # ========================================================

    if best_contour is None:

        print("    No valid contour found")

        return None


    # ========================================================
    # BOUNDING BOX
    # ========================================================

    x, y, w, h = cv2.boundingRect(
        best_contour
    )

    # --------------------------------------------------------
    # Make square around target
    # --------------------------------------------------------

    cx = x + w / 2
    cy = y + h / 2

    side = max(w, h)

    # --------------------------------------------------------
    # Small padding
    # --------------------------------------------------------

    pad = int(
        side * PADDING_RATIO
    )

    side = side + 2 * pad

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

    x2 = x1 + side
    y2 = y1 + side

    # --------------------------------------------------------
    # Keep crop inside image
    # --------------------------------------------------------

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(W, x2)
    y2 = min(H, y2)

    return x1, y1, x2, y2


# ============================================================
# CREATE DEBUG IMAGE
# ============================================================

def create_debug(original_img, bbox):

    debug = to_uint8(original_img)

    if debug.ndim == 2:
        debug = cv2.cvtColor(
            debug,
            cv2.COLOR_GRAY2BGR
        )

    x1, y1, x2, y2 = bbox

    # EXACT SAME box that will be cropped
    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        4
    )

    return debug


# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(
    image_path,
    output_folder,
    debug_folder
):

    # ========================================================
    # READ ORIGINAL
    # ========================================================

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if original is None:

        print(
            f"[ERROR] Cannot read: "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # DETECT TARGET FROM ORIGINAL
    # ========================================================

    bbox = detect_target(original)

    if bbox is None:

        print(
            f"[REJECT] {image_path.name}"
        )

        return False


    x1, y1, x2, y2 = bbox


    # ========================================================
    # CREATE NEW DEBUG
    # ========================================================

    debug = create_debug(
        original,
        bbox
    )

    debug_path = (
        debug_folder
        / f"{image_path.stem}_debug.png"
    )

    cv2.imwrite(
        str(debug_path),
        debug
    )


    # ========================================================
    # FINAL CROP
    #
    # IMPORTANT:
    # Crop is taken DIRECTLY FROM ORIGINAL.
    #
    # EXACTLY the area inside green box.
    # ========================================================

    final_crop = original[
        y1:y2,
        x1:x2
    ].copy()


    # ========================================================
    # SAVE PROCESSED IMAGE
    # ========================================================

    output_path = (
        output_folder
        / image_path.name
    )

    success = cv2.imwrite(
        str(output_path),
        final_crop
    )

    if not success:

        print(
            f"[ERROR SAVE] "
            f"{output_path}"
        )

        return False


    print(
        f"[OK] {image_path.name} "
        f"-> crop {final_crop.shape[1]}x"
        f"{final_crop.shape[0]}"
    )

    return True


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder(
    input_folder,
    output_folder
):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # --------------------------------------------------------
    # Separate debug folder inside OUTPUT
    # --------------------------------------------------------

    debug_folder = (
        output_folder
        / "debug"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print("=" * 70)
    print(f"INPUT : {input_folder}")
    print(f"OUTPUT: {output_folder}")
    print("=" * 70)


    count = 0
    success = 0


    for image_path in sorted(
        input_folder.iterdir()
    ):

        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        count += 1

        if process_image(
            image_path,
            output_folder,
            debug_folder
        ):
            success += 1


    print()
    print(
        f"Finished: {success}/{count}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "STARTING IMAGE PREPROCESSING"
    )

    print(
        "ORIGINAL DATA WILL NOT BE MODIFIED"
    )

    print()


    for position, label in DATASETS:

        input_folder = (
            INPUT_ROOT
            / position
            / label
        )

        output_folder = (
            OUTPUT_ROOT
            / position
            / label
        )


        if not input_folder.exists():

            print(
                f"[MISSING] "
                f"{input_folder}"
            )

            continue


        process_folder(
            input_folder,
            output_folder
        )


    print()
    print("=" * 70)
    print("DONE")
    print()
    print(
        "Original folders were READ ONLY."
    )
    print(
        f"Results saved in:\n"
        f"{OUTPUT_ROOT}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()