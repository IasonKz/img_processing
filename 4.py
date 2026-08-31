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
# OFFSET
#
# Extra margin on EACH side of the fitted die.
#
# 0.04 = about 4% extra per side
#
# resulting square side:
#
#     detected_side * 1.08
#
# This is relatively tight but still safer to include
# slightly more instead of cutting the die.
# ------------------------------------------------------------

OFFSET_RATIO = 0.04


# ------------------------------------------------------------
# DETECTION REGION
# ------------------------------------------------------------

SEARCH_RATIO = 0.62


# Candidate center cannot be too far from image center
MAX_CENTER_OFFSET_X = 0.20
MAX_CENTER_OFFSET_Y = 0.20


# Approximate die size relative to full image
MIN_SIZE_RATIO = 0.08
MAX_SIZE_RATIO = 0.42

EXPECTED_SIZE_RATIO = 0.18


# Detection confidence
DETECTION_SCORE_MIN = 7.0


# ------------------------------------------------------------
# ROTATION
#
# You said only small rotations are expected.
#
# If fitted die requires > 15 degrees correction,
# treat it as suspicious / Missing.
# ------------------------------------------------------------

MAX_ROTATION_DEG = 15.0


# ============================================================
# GRAYSCALE COPY FOR DETECTION
# ============================================================

def gray8(img):
    """
    Create an 8-bit grayscale COPY.

    Used ONLY for:
        - target detection
        - debug images

    Original image is never modified.
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

    lo = float(
        np.percentile(x, 1)
    )

    hi = float(
        np.percentile(x, 99)
    )


    if hi <= lo:
        hi = lo + 1.0


    x = np.clip(
        (x - lo) / (hi - lo),
        0.0,
        1.0
    )


    return (
        255.0 * x
    ).astype(np.uint8)


# ============================================================
# ORDER 4 RECTANGLE CORNERS
# ============================================================

def order_points(pts):
    """
    Order points:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    pts = np.asarray(
        pts,
        dtype=np.float32
    )


    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )


    s = pts.sum(axis=1)

    diff = np.diff(
        pts,
        axis=1
    ).reshape(-1)


    ordered[0] = pts[
        np.argmin(s)
    ]  # TL


    ordered[2] = pts[
        np.argmax(s)
    ]  # BR


    ordered[1] = pts[
        np.argmin(diff)
    ]  # TR


    ordered[3] = pts[
        np.argmax(diff)
    ]  # BL


    return ordered


# ============================================================
# NORMALIZE SMALL ROTATION ANGLE
# ============================================================

def normalize_square_angle(angle):
    """
    Square orientation is equivalent every 90 degrees.

    Convert to approximately:
        [-45°, +45°)
    """

    while angle >= 45.0:
        angle -= 90.0

    while angle < -45.0:
        angle += 90.0

    return angle


# ============================================================
# FIT CENTRAL ROTATED DIE
# ============================================================

def detect_central_die(original):
    """
    Find central approximately-square die.

    Returns information about the best fitted
    rotated rectangle.

    No crop / rotation is applied here.
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
            image_cx - search_w / 2
        )
    )


    sy1 = max(
        0,
        int(
            image_cy - search_h / 2
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
    # PREPROCESS DETECTION COPY
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
            0.60 * median
        )
    )


    high = int(
        min(
            255,
            max(
                low + 30,
                1.40 * median
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
    # CONTOURS
    # ========================================================

    contours1, _ = cv2.findContours(
        bright,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    contours2, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    contours = (
        contours1
        +
        contours2
    )


    best = None
    best_score = -1e30


    # ========================================================
    # TEST CANDIDATES
    # ========================================================

    for contour in contours:

        contour_area = float(
            cv2.contourArea(contour)
        )


        if contour_area <= 0:
            continue


        # ----------------------------------------------------
        # FIT ROTATED RECTANGLE
        # ----------------------------------------------------

        rect = cv2.minAreaRect(
            contour
        )


        box = cv2.boxPoints(
            rect
        )


        # Convert ROI coordinates to full image coordinates
        box[:, 0] += sx1
        box[:, 1] += sy1


        box = order_points(
            box
        )


        tl, tr, br, bl = box


        # ----------------------------------------------------
        # Edge lengths
        # ----------------------------------------------------

        top = np.linalg.norm(
            tr - tl
        )

        bottom = np.linalg.norm(
            br - bl
        )

        left = np.linalg.norm(
            bl - tl
        )

        right = np.linalg.norm(
            br - tr
        )


        width = (
            top + bottom
        ) / 2.0


        height = (
            left + right
        ) / 2.0


        if (
            width < 5
            or
            height < 5
        ):
            continue


        long_side = max(
            width,
            height
        )


        short_side = min(
            width,
            height
        )


        # ----------------------------------------------------
        # Should be approximately square
        # ----------------------------------------------------

        aspect = (
            long_side
            / short_side
        )


        if aspect > 1.70:
            continue


        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        center = np.mean(
            box,
            axis=0
        )


        cx = float(
            center[0]
        )


        cy = float(
            center[1]
        )


        # ====================================================
        # HARD CENTRAL REJECTION
        # ====================================================

        dx = abs(
            cx - image_cx
        ) / W


        dy = abs(
            cy - image_cy
        ) / H


        if dx > MAX_CENTER_OFFSET_X:
            continue


        if dy > MAX_CENTER_OFFSET_Y:
            continue


        # ====================================================
        # SIZE
        # ====================================================

        size_ratio = 0.5 * (

            width / W

            +

            height / H
        )


        if (
            size_ratio < MIN_SIZE_RATIO
            or
            size_ratio > MAX_SIZE_RATIO
        ):
            continue


        # ====================================================
        # ROTATION ANGLE
        # ====================================================

        top_vec = (
            tr - tl
        )


        raw_angle = np.degrees(
            np.arctan2(
                top_vec[1],
                top_vec[0]
            )
        )


        angle = normalize_square_angle(
            float(raw_angle)
        )


        # ====================================================
        # SCORES
        # ====================================================

        center_distance = np.sqrt(
            dx * dx
            + dy * dy
        )


        max_center_distance = np.sqrt(

            MAX_CENTER_OFFSET_X ** 2

            +

            MAX_CENTER_OFFSET_Y ** 2
        )


        center_score = (

            1.0

            - center_distance
            / max_center_distance
        )


        center_score = float(
            np.clip(
                center_score,
                0.0,
                1.0
            )
        )


        square_score = (
            1.0 / aspect
        )


        rectangle_area = (
            width * height
        )


        fill_score = min(

            contour_area
            / max(
                rectangle_area,
                1.0
            ),

            1.0
        )


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

                "center":
                    np.array(
                        [cx, cy],
                        dtype=np.float32
                    ),

                "box":
                    box.copy(),

                "width":
                    float(width),

                "height":
                    float(height),

                "side":
                    float(long_side),

                "angle":
                    float(angle),

                "score":
                    float(score),
            }


    return best


# ============================================================
# BUILD EXPANDED ROTATED SQUARE
# ============================================================

def build_rotated_square(
    detection
):
    """
    Build a TRUE square around the detected die.

    It has:
        - same center
        - same small rotation
        - side = largest fitted side + offset

    This intentionally prefers a little MORE
    rather than cutting into the die.
    """

    box = detection["box"]

    tl, tr, br, bl = box


    center = detection[
        "center"
    ]


    # ========================================================
    # DEFINE HORIZONTAL DIRECTION OF DIE
    # ========================================================

    top_vec = (
        tr - tl
    )


    bottom_vec = (
        br - bl
    )


    u = (
        top_vec
        +
        bottom_vec
    ) / 2.0


    norm_u = np.linalg.norm(
        u
    )


    if norm_u <= 1e-6:
        return None


    u = (
        u / norm_u
    )


    # Make direction approximately left -> right
    if u[0] < 0:
        u = -u


    # Perpendicular direction
    v = np.array(
        [
            -u[1],
            u[0]
        ],
        dtype=np.float32
    )


    # Make v point approximately downward
    if v[1] < 0:
        v = -v


    # ========================================================
    # SQUARE SIDE + OFFSET
    # ========================================================

    die_side = max(

        detection["width"],

        detection["height"]
    )


    square_side = (

        die_side

        * (
            1.0
            +
            2.0
            * OFFSET_RATIO
        )
    )


    half = (
        square_side / 2.0
    )


    # ========================================================
    # FOUR ROTATED SQUARE CORNERS
    # ========================================================

    tl2 = (
        center
        - u * half
        - v * half
    )


    tr2 = (
        center
        + u * half
        - v * half
    )


    br2 = (
        center
        + u * half
        + v * half
    )


    bl2 = (
        center
        - u * half
        + v * half
    )


    square = np.array(
        [
            tl2,
            tr2,
            br2,
            bl2
        ],
        dtype=np.float32
    )


    return (
        square,
        square_side
    )


# ============================================================
# CHECK SQUARE IS INSIDE IMAGE
# ============================================================

def square_inside_image(
    square,
    image
):

    H, W = image.shape[:2]


    xs = square[:, 0]
    ys = square[:, 1]


    return (

        np.all(xs >= 0)

        and

        np.all(xs < W)

        and

        np.all(ys >= 0)

        and

        np.all(ys < H)
    )


# ============================================================
# RECTIFY / STRAIGHTEN ROTATED SQUARE
# ============================================================

def rectify_die(
    original,
    square,
    square_side
):
    """
    Perspective-transform the slightly rotated square
    into an upright square.

    No ECC.
    No reference image.
    No arbitrary rotation.

    Only geometry from the detected die is used.
    """

    native_size = max(
        100,
        int(
            round(
                square_side
            )
        )
    )


    destination = np.array(
        [
            [0, 0],

            [
                native_size - 1,
                0
            ],

            [
                native_size - 1,
                native_size - 1
            ],

            [
                0,
                native_size - 1
            ],
        ],
        dtype=np.float32
    )


    transform = cv2.getPerspectiveTransform(
        square.astype(
            np.float32
        ),
        destination
    )


    rectified = cv2.warpPerspective(

        original,

        transform,

        (
            native_size,
            native_size
        ),

        flags=cv2.INTER_LINEAR,

        borderMode=cv2.BORDER_REPLICATE
    )


    return rectified


# ============================================================
# RESIZE FINAL TO EXACTLY 600x600
# ============================================================

def resize_to_600(
    image
):

    H, W = image.shape[:2]


    if (
        H == FINAL_SIZE
        and
        W == FINAL_SIZE
    ):

        return image.copy()


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

        image,

        (
            FINAL_SIZE,
            FINAL_SIZE
        ),

        interpolation=interpolation
    )


# ============================================================
# SLIGHT LIGHTING CORRECTION
# ============================================================

def mean_brightness_ratio(
    img
):
    """
    Approximate absolute brightness in [0,1].
    """

    if img.ndim == 3:

        values = img.astype(
            np.float32
        ).mean(axis=2)

    else:

        values = img.astype(
            np.float32
        )


    if np.issubdtype(
        img.dtype,
        np.integer
    ):

        max_value = float(
            np.iinfo(
                img.dtype
            ).max
        )

    else:

        max_value = 1.0


    return float(
        np.mean(values)
        / max_value
    )


def lighten_if_dark(
    img
):
    """
    Only mild brightening.

    No histogram manipulation on final dataset.
    No strong CLAHE.

    Maximum intensity increase:
        about 12%.
    """

    brightness = mean_brightness_ratio(
        img
    )


    DARK_THRESHOLD = 0.22


    if brightness >= DARK_THRESHOLD:
        return img.copy()


    darkness = (

        DARK_THRESHOLD
        - brightness

    ) / DARK_THRESHOLD


    darkness = float(
        np.clip(
            darkness,
            0.0,
            1.0
        )
    )


    # Maximum 12% brighter
    factor = (

        1.0

        +

        0.12
        * darkness
    )


    if np.issubdtype(
        img.dtype,
        np.integer
    ):

        max_value = float(
            np.iinfo(
                img.dtype
            ).max
        )


        result = (

            img.astype(
                np.float32
            )

            * factor
        )


        result = np.clip(
            result,
            0,
            max_value
        )


        return result.astype(
            img.dtype
        )


    return img.copy()


# ============================================================
# DEBUG IMAGE WITH ROTATED SQUARE
# ============================================================

def make_debug(
    original,
    square,
    detection,
    color=(0, 255, 0)
):

    g = gray8(
        original
    )


    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    points = np.round(
        square
    ).astype(
        np.int32
    )


    # ========================================================
    # GREEN ROTATED CROP BOX
    # ========================================================

    cv2.polylines(

        debug,

        [
            points.reshape(
                (-1, 1, 2)
            )
        ],

        True,

        color,

        4
    )


    # Center
    center = np.round(
        detection["center"]
    ).astype(int)


    cv2.circle(
        debug,
        tuple(center),
        5,
        (0, 0, 255),
        -1
    )


    text = (

        f"score="
        f"{detection['score']:.2f} "

        f"angle="
        f"{detection['angle']:.2f} deg"
    )


    cv2.putText(

        debug,

        text,

        (
            30,
            50
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        color,

        2,

        cv2.LINE_AA
    )


    return debug


# ============================================================
# MISSING DEBUG
# ============================================================

def make_missing_debug(
    original,
    text,
    square=None
):

    g = gray8(
        original
    )


    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    if square is not None:

        pts = np.round(
            square
        ).astype(
            np.int32
        )


        cv2.polylines(

            debug,

            [
                pts.reshape(
                    (-1, 1, 2)
                )
            ],

            True,

            (0, 0, 255),

            4
        )


    cv2.putText(

        debug,

        text,

        (
            30,
            60
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (0, 0, 255),

        3,

        cv2.LINE_AA
    )


    return debug


# ============================================================
# COPY ORIGINAL TO MISSING
# ============================================================

def save_missing(
    image_path,
    original,
    missing_folder,
    missing_debug_folder,
    reason,
    square=None
):

    missing_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    missing_debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # COPY ORIGINAL
    #
    # Never move/delete/modify source.
    # ========================================================

    destination = (

        missing_folder

        / image_path.name
    )


    shutil.copy2(

        str(image_path),

        str(destination)
    )


    debug = make_missing_debug(

        original,

        reason,

        square
    )


    debug_path = (

        missing_debug_folder

        / (
            image_path.stem
            + "_missing.png"
        )
    )


    cv2.imwrite(
        str(debug_path),
        debug
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    output_folder,
    debug_folder,
    missing_folder,
    missing_debug_folder
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

            f"[ERROR READ] "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # DETECT CENTRAL DIE
    # ========================================================

    detection = detect_central_die(
        original
    )


    if detection is None:

        save_missing(

            image_path,
            original,

            missing_folder,
            missing_debug_folder,

            "MISSING - NO DIE"
        )


        print(

            f"[MISSING] "
            f"{image_path.name} "

            f"| die not found"
        )

        return False


    # ========================================================
    # CHECK SCORE
    # ========================================================

    if (
        detection["score"]
        < DETECTION_SCORE_MIN
    ):

        save_missing(

            image_path,
            original,

            missing_folder,
            missing_debug_folder,

            (
                "MISSING - "
                "LOW CONFIDENCE"
            )
        )


        print(

            f"[MISSING] "
            f"{image_path.name} "

            f"| score="
            f"{detection['score']:.2f}"
        )

        return False


    # ========================================================
    # CHECK ROTATION
    # ========================================================

    if abs(
        detection["angle"]
    ) > MAX_ROTATION_DEG:

        save_missing(

            image_path,
            original,

            missing_folder,
            missing_debug_folder,

            (
                "MISSING - "
                "TOO MUCH ROTATION"
            )
        )


        print(

            f"[MISSING] "
            f"{image_path.name} "

            f"| suspicious angle="
            f"{detection['angle']:.2f}"
        )

        return False


    # ========================================================
    # BUILD ROTATED SQUARE + OFFSET
    # ========================================================

    result = build_rotated_square(
        detection
    )


    if result is None:

        save_missing(

            image_path,
            original,

            missing_folder,
            missing_debug_folder,

            "MISSING - BAD GEOMETRY"
        )

        return False


    square, square_side = result


    # ========================================================
    # CHECK SQUARE INSIDE IMAGE
    # ========================================================

    if not square_inside_image(
        square,
        original
    ):

        save_missing(

            image_path,
            original,

            missing_folder,
            missing_debug_folder,

            (
                "MISSING - "
                "DIE OUTSIDE IMAGE"
            ),

            square
        )


        print(

            f"[MISSING] "
            f"{image_path.name} "

            f"| crop outside image"
        )

        return False


    # ========================================================
    # SAVE DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_debug(

        original,

        square,

        detection
    )


    debug_path = (

        debug_folder

        / (
            image_path.stem
            + "_debug.png"
        )
    )


    cv2.imwrite(
        str(debug_path),
        debug
    )


    # ========================================================
    # STRAIGHTEN ROTATED DIE
    #
    # This is the ONLY geometric alignment.
    #
    # No reference.
    # No ECC.
    # No 90/180/270 search.
    # ========================================================

    rectified = rectify_die(

        original,

        square,

        square_side
    )


    # ========================================================
    # EXACTLY 600 x 600
    # ========================================================

    final_img = resize_to_600(
        rectified
    )


    # ========================================================
    # SLIGHT BRIGHTEN IF NECESSARY
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

        f"| angle="
        f"{detection['angle']:.2f} deg "

        f"| crop="
        f"{square_side:.0f}px "

        f"| score="
        f"{detection['score']:.2f} "

        f"| final=600x600"
    )


    return True


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder(
    input_folder,
    output_folder,
    debug_folder,
    missing_folder,
    missing_debug_folder
):

    total = 0
    good = 0
    missing = 0


    print()

    print(
        "=" * 80
    )


    print(
        f"INPUT  : "
        f"{input_folder}"
    )


    print(
        f"OUTPUT : "
        f"{output_folder}"
    )


    print(
        "=" * 80
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

            missing_folder,

            missing_debug_folder
        )


        if success:
            good += 1

        else:
            missing += 1


    print()

    print(

        f"Processed: "
        f"{good}/{total}"
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
        "ROTATED DIE CROP PREPROCESSING"
    )

    print(
        "Final output = 600 x 600"
    )

    print(
        "No ECC / No reference alignment"
    )

    print(
        "Original data is READ ONLY"
    )


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


        missing_debug_folder = (

            OUTPUT_ROOT

            / "_debug_missing"

            / position

            / label
        )


        if not input_folder.exists():

            print(

                f"[MISSING INPUT] "
                f"{input_folder}"
            )

            continue


        process_folder(

            input_folder,

            output_folder,

            debug_folder,

            missing_folder,

            missing_debug_folder
        )


    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "Final processed images: 600x600"
    )

    print(
        "Original files were not modified."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()