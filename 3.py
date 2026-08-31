from pathlib import Path
import shutil
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka"
)

OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka_processed_final"
)


DATASETS = [
    ("ViTop", "Good"),
    ("ViTop", "Bad"),
    ("ViBottom", "Good"),
    ("ViBottom", "Bad"),
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
# SETTINGS
# ============================================================

# Final classifier image size
FINAL_SIZE = 600


# ------------------------------------------------------------
# CROP OFFSET
# ------------------------------------------------------------
#
# 1.05 means only ~5% extra total around the detected target.
#
# Previous version was 1.12 -> too much background.
#
# Try:
# 1.03 = very tight
# 1.05 = recommended
# 1.08 = a little safer
#
CROP_EXPAND_FACTOR = 1.05


# ------------------------------------------------------------
# CENTRAL TARGET DETECTION
# ------------------------------------------------------------

SEARCH_RATIO = 0.60

MAX_CENTER_OFFSET_X = 0.19
MAX_CENTER_OFFSET_Y = 0.19

MIN_SIZE_RATIO = 0.08
MAX_SIZE_RATIO = 0.42

EXPECTED_SIZE_RATIO = 0.18


# If confidence is too low -> Missing
DETECTION_SCORE_MIN = 7.0


# ============================================================
# IMAGE HELPERS
# ============================================================

def gray8(img):
    """
    Create an 8-bit grayscale COPY for detection/debug.

    The original image is never modified.
    """

    if img.ndim == 3:

        if img.shape[2] == 4:
            g = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2GRAY
            )

        else:
            g = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

    else:
        g = img.copy()


    if g.dtype == np.uint8:
        return g.copy()


    x = g.astype(np.float32)

    low = float(
        np.percentile(x, 1)
    )

    high = float(
        np.percentile(x, 99)
    )


    if high <= low:
        high = low + 1.0


    x = np.clip(
        (x - low) / (high - low),
        0.0,
        1.0
    )


    return (
        x * 255.0
    ).astype(np.uint8)


# ============================================================
# SLIGHT BRIGHTENING
# ============================================================

def lighten_if_dark(img):
    """
    Slightly brightens only genuinely dark images.

    No CLAHE is applied to the final image.
    No aggressive normalization.

    Gamma:
        1.0 = unchanged
        ~0.90 = slightly brighter
    """

    g = gray8(img)

    mean_brightness = float(
        np.mean(g)
    )


    # Already bright enough
    if mean_brightness >= 65:
        return img.copy()


    # --------------------------------------------------------
    # Mild brightness correction only
    # --------------------------------------------------------

    darkness = np.clip(
        (65.0 - mean_brightness) / 65.0,
        0.0,
        1.0
    )


    gamma = (
        1.0
        - 0.10 * darkness
    )


    # --------------------------------------------------------
    # Preserve original dtype
    # --------------------------------------------------------

    if np.issubdtype(
        img.dtype,
        np.integer
    ):

        max_value = float(
            np.iinfo(img.dtype).max
        )

        x = (
            img.astype(np.float32)
            / max_value
        )

        x = np.power(
            np.clip(
                x,
                0.0,
                1.0
            ),
            gamma
        )

        result = np.clip(
            x * max_value,
            0,
            max_value
        )

        return result.astype(
            img.dtype
        )


    return img.copy()


# ============================================================
# DETECT CENTRAL TARGET
# ============================================================

def detect_central_target(original):
    """
    Detect approximately square CENTRAL target.

    Important:
    Side targets are hard-rejected.

    Returns:
        {
            "cx": center x,
            "cy": center y,
            "side": detected target size,
            "score": confidence
        }

    or None.
    """

    g = gray8(original)

    H, W = g.shape[:2]


    image_cx = W / 2.0
    image_cy = H / 2.0


    # ========================================================
    # CENTRAL SEARCH REGION
    # ========================================================

    search_w = int(
        W * SEARCH_RATIO
    )

    search_h = int(
        H * SEARCH_RATIO
    )


    sx1 = max(
        0,
        int(
            image_cx
            - search_w / 2
        )
    )

    sy1 = max(
        0,
        int(
            image_cy
            - search_h / 2
        )
    )


    sx2 = min(
        W,
        sx1 + search_w
    )

    sy2 = min(
        H,
        sy1 + search_h
    )


    roi = g[
        sy1:sy2,
        sx1:sx2
    ].copy()


    # ========================================================
    # PREPROCESS ONLY DETECTION COPY
    # ========================================================

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )


    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )


    enhanced = clahe.apply(
        blurred
    )


    # ========================================================
    # BRIGHT MASK
    # ========================================================

    _, bright = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    # ========================================================
    # EDGE MASK
    # ========================================================

    median = float(
        np.median(enhanced)
    )


    low = int(
        max(
            10,
            median * 0.60
        )
    )


    high = int(
        min(
            255,
            max(
                low + 30,
                median * 1.40
            )
        )
    )


    edges = cv2.Canny(
        enhanced,
        low,
        high
    )


    # ========================================================
    # MORPHOLOGY
    # ========================================================

    kernel9 = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )


    kernel5 = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )


    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        kernel9,
        iterations=2
    )


    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        kernel5,
        iterations=1
    )


    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel9,
        iterations=3
    )


    edges = cv2.dilate(
        edges,
        kernel5,
        iterations=1
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours_bright, _ = cv2.findContours(
        bright,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    contours_edges, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    contours = (
        contours_bright
        +
        contours_edges
    )


    best = None
    best_score = -1e30


    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )


        if area <= 0:
            continue


        rect = cv2.minAreaRect(
            contour
        )


        (
            (cx_roi, cy_roi),
            (rw, rh),
            angle
        ) = rect


        if rw < 5 or rh < 5:
            continue


        # Coordinates in full image
        cx = cx_roi + sx1
        cy = cy_roi + sy1


        long_side = max(
            rw,
            rh
        )


        short_side = min(
            rw,
            rh
        )


        if short_side <= 1:
            continue


        # ====================================================
        # SHAPE
        # ====================================================

        aspect = (
            long_side
            / short_side
        )


        if aspect > 1.70:
            continue


        # ====================================================
        # SIZE
        # ====================================================

        size_ratio = 0.5 * (

            long_side / W

            +

            short_side / H
        )


        if (
            size_ratio < MIN_SIZE_RATIO
            or
            size_ratio > MAX_SIZE_RATIO
        ):
            continue


        # ====================================================
        # HARD CENTER FILTER
        # ====================================================

        dx = (
            abs(
                cx - image_cx
            )
            / W
        )


        dy = (
            abs(
                cy - image_cy
            )
            / H
        )


        if dx > MAX_CENTER_OFFSET_X:
            continue


        if dy > MAX_CENTER_OFFSET_Y:
            continue


        # ====================================================
        # CENTER SCORE
        # ====================================================

        center_distance = np.sqrt(
            dx * dx
            + dy * dy
        )


        max_distance = np.sqrt(

            MAX_CENTER_OFFSET_X ** 2

            +

            MAX_CENTER_OFFSET_Y ** 2
        )


        center_score = (

            1.0

            - center_distance
            / max_distance
        )


        center_score = float(
            np.clip(
                center_score,
                0.0,
                1.0
            )
        )


        # ====================================================
        # SQUARE SCORE
        # ====================================================

        square_score = (
            1.0 / aspect
        )


        # ====================================================
        # FILL SCORE
        # ====================================================

        rect_area = (
            rw * rh
        )


        fill_score = min(

            area
            / max(
                rect_area,
                1.0
            ),

            1.0
        )


        # ====================================================
        # SIZE SCORE
        # ====================================================

        size_score = (

            1.0

            - abs(
                size_ratio
                - EXPECTED_SIZE_RATIO
            )
            / EXPECTED_SIZE_RATIO
        )


        size_score = float(
            np.clip(
                size_score,
                0.0,
                1.0
            )
        )


        # ====================================================
        # TOTAL SCORE
        # ====================================================

        score = (

            10.0
            * center_score

            +

            2.0
            * square_score

            +

            1.5
            * fill_score

            +

            1.5
            * size_score
        )


        if score > best_score:

            best_score = score

            best = {

                "cx":
                    float(cx),

                "cy":
                    float(cy),

                "side":
                    float(long_side),

                "score":
                    float(score),
            }


    return best


# ============================================================
# TIGHT CROP SIZE
# ============================================================

def compute_crop_size(
    detected_side
):
    """
    IMPORTANT CHANGE:

    We DO NOT force crop size >= 600 anymore.

    Example:
        target = 470 px
        crop   = ~494 px

    Then:
        494x494 -> resize -> 600x600
    """

    crop_size = int(
        round(
            detected_side
            * CROP_EXPAND_FACTOR
        )
    )


    # Safety minimum only
    crop_size = max(
        crop_size,
        100
    )


    return crop_size


# ============================================================
# SQUARE CROP
# ============================================================

def crop_square(
    original,
    cx,
    cy,
    crop_size
):

    H, W = original.shape[:2]


    if (
        crop_size > W
        or crop_size > H
    ):

        return (
            None,
            None,
            True
        )


    x1 = int(
        round(
            cx
            - crop_size / 2
        )
    )


    y1 = int(
        round(
            cy
            - crop_size / 2
        )
    )


    # Check if target is outside reasonable image region
    outside = (

        x1 < 0
        or
        y1 < 0
        or
        x1 + crop_size > W
        or
        y1 + crop_size > H
    )


    if outside:

        return (
            None,
            None,
            True
        )


    x2 = (
        x1
        + crop_size
    )


    y2 = (
        y1
        + crop_size
    )


    crop = original[
        y1:y2,
        x1:x2
    ].copy()


    bbox = (
        x1,
        y1,
        x2,
        y2
    )


    return (
        crop,
        bbox,
        False
    )


# ============================================================
# DEBUG
# ============================================================

def make_debug(
    original,
    bbox,
    color=(0, 255, 0),
    label="ROI"
):

    debug_gray = gray8(
        original
    )


    debug = cv2.cvtColor(
        debug_gray,
        cv2.COLOR_GRAY2BGR
    )


    (
        x1,
        y1,
        x2,
        y2
    ) = bbox


    cv2.rectangle(
        debug,
        (x1, y1),
        (x2 - 1, y2 - 1),
        color,
        4
    )


    cv2.putText(
        debug,
        label,
        (
            x1,
            max(
                25,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
    )


    return debug


# ============================================================
# RESIZE TO EXACT 600x600
# ============================================================

def resize_to_600(
    crop
):

    H, W = crop.shape[:2]


    if (
        H == FINAL_SIZE
        and
        W == FINAL_SIZE
    ):

        return crop.copy()


    # --------------------------------------------------------
    # If crop is larger -> INTER_AREA
    # If crop is smaller -> INTER_CUBIC
    # --------------------------------------------------------

    if (
        H > FINAL_SIZE
        or
        W > FINAL_SIZE
    ):

        interpolation = (
            cv2.INTER_AREA
        )

    else:

        interpolation = (
            cv2.INTER_CUBIC
        )


    return cv2.resize(
        crop,
        (
            FINAL_SIZE,
            FINAL_SIZE
        ),
        interpolation=interpolation
    )


# ============================================================
# MISSING
# ============================================================

def save_missing(
    image_path,
    missing_folder
):

    missing_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    destination = (

        missing_folder

        / image_path.name
    )


    # COPY ONLY.
    # Original remains untouched.
    shutil.copy2(
        str(image_path),
        str(destination)
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    output_folder,
    debug_folder,
    missing_folder
):

    # ========================================================
    # READ ORIGINAL ONLY
    # ========================================================

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )


    if original is None:

        print(
            f"[ERROR READ] "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # DETECT
    # ========================================================

    detection = detect_central_target(
        original
    )


    if detection is None:

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| target not found"
        )

        return False


    if (
        detection["score"]
        < DETECTION_SCORE_MIN
    ):

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| low score "
            f"{detection['score']:.2f}"
        )

        return False


    # ========================================================
    # TIGHT CROP
    # ========================================================

    crop_size = compute_crop_size(
        detection["side"]
    )


    (
        crop,
        bbox,
        outside
    ) = crop_square(

        original,

        detection["cx"],
        detection["cy"],

        crop_size
    )


    if (
        outside
        or
        crop is None
    ):

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| target outside image"
        )

        return False


    # ========================================================
    # DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_debug(
        original,
        bbox
    )


    debug_path = (

        debug_folder

        / (
            f"{image_path.stem}"
            f"_debug.png"
        )
    )


    cv2.imwrite(
        str(debug_path),
        debug
    )


    # ========================================================
    # NO ROTATION
    # NO ECC
    # NO REFERENCE
    # NO WARP
    #
    # JUST RESIZE THE DETECTED CROP
    # ========================================================

    final_img = resize_to_600(
        crop
    )


    # ========================================================
    # ONLY SLIGHTLY BRIGHTEN IF DARK
    # ========================================================

    final_img = lighten_if_dark(
        final_img
    )


    # ========================================================
    # SAVE PROCESSED COPY
    # ========================================================

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (

        output_folder

        / image_path.name
    )


    success = cv2.imwrite(
        str(output_path),
        final_img
    )


    if not success:

        print(
            f"[ERROR SAVE] "
            f"{output_path}"
        )

        return False


    print(

        f"[OK] "
        f"{image_path.name} "

        f"| crop="
        f"{crop_size}x{crop_size} "

        f"-> 600x600 "

        f"| score="
        f"{detection['score']:.2f}"
    )


    return True


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder(
    input_folder,
    output_folder,
    debug_folder,
    missing_folder
):

    total = 0
    processed = 0
    missing = 0


    print()

    print(
        "=" * 75
    )

    print(
        f"INPUT:  "
        f"{input_folder}"
    )

    print(
        f"OUTPUT: "
        f"{output_folder}"
    )

    print(
        "=" * 75
    )


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


        total += 1


        success = process_image(

            image_path,

            output_folder,

            debug_folder,

            missing_folder
        )


        if success:
            processed += 1

        else:
            missing += 1


    print()

    print(
        f"Processed: "
        f"{processed}/{total}"
    )

    print(
        f"Missing: "
        f"{missing}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "START PREPROCESSING"
    )

    print(
        "NO ALIGNMENT / NO ROTATION / NO ECC"
    )

    print(
        "FINAL SIZE = 600 x 600"
    )

    print(
        "ORIGINAL DATA = READ ONLY"
    )


    for (
        position,
        label
    ) in DATASETS:


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


        debug_folder = (

            OUTPUT_ROOT

            / "_debug"

            / position

            / label
        )


        missing_folder = (

            OUTPUT_ROOT

            / "Missing"

            / position

            / label
        )


        if not input_folder.exists():

            print(
                f"[MISSING INPUT FOLDER] "
                f"{input_folder}"
            )

            continue


        process_folder(

            input_folder,

            output_folder,

            debug_folder,

            missing_folder
        )


    print()

    print(
        "=" * 75
    )

    print(
        "DONE"
    )

    print(
        "All processed images are 600x600."
    )

    print(
        "No original image was modified."
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()