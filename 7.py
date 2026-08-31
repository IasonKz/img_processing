from pathlib import Path
import shutil
import cv2
import numpy as np


# ============================================================
# PATHS - ViBottom TEST ONLY
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka\ViBottom"
)

OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\ViBottom_crop_test_v2"
)

LABELS = [
    "Good",
    "Bad"
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

FINAL_SIZE = 600


# Small margin around the fitted die.
#
# 0.03 = 3% extra PER SIDE
OFFSET_RATIO = 0.03


# Allow die to be substantially displaced.
# This is deliberately less strict than ViTop.
MAX_CENTER_OFFSET_X = 0.32
MAX_CENTER_OFFSET_Y = 0.32


# Approximate allowed size of die
MIN_SIDE_RATIO = 0.14
MAX_SIDE_RATIO = 0.52


# Only small rotations should normally occur.
MAX_ROTATION_DEG = 18.0


# Detection confidence
MIN_DETECTION_SCORE = 3.0


# ============================================================
# 8-BIT GRAYSCALE COPY
# ============================================================

def gray8(img):

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
        x * 255.0
    ).astype(np.uint8)


# ============================================================
# ORDER RECTANGLE POINTS
# ============================================================

def order_points(points):

    pts = np.asarray(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )


    sums = pts.sum(axis=1)

    diffs = np.diff(
        pts,
        axis=1
    ).reshape(-1)


    # Top left
    ordered[0] = pts[
        np.argmin(sums)
    ]

    # Top right
    ordered[1] = pts[
        np.argmin(diffs)
    ]

    # Bottom right
    ordered[2] = pts[
        np.argmax(sums)
    ]

    # Bottom left
    ordered[3] = pts[
        np.argmax(diffs)
    ]


    return ordered


# ============================================================
# ANGLE OF FITTED DIE
# ============================================================

def rectangle_angle(box):

    box = order_points(box)

    tl, tr, br, bl = box

    top_vector = tr - tl

    angle = np.degrees(
        np.arctan2(
            top_vector[1],
            top_vector[0]
        )
    )


    # Equivalent every 90 degrees
    while angle >= 45:
        angle -= 90

    while angle < -45:
        angle += 90


    return float(angle)


# ============================================================
# BRIGHTNESS CONTRAST SCORE
# ============================================================

def rectangle_contrast(
    gray,
    box
):

    H, W = gray.shape[:2]

    box = np.asarray(
        box,
        dtype=np.float32
    )


    center = np.mean(
        box,
        axis=0
    )


    # --------------------------------------------------------
    # Inner rectangle
    # --------------------------------------------------------

    inner = np.round(
        box
    ).astype(np.int32)


    inner_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        inner_mask,
        inner,
        255
    )


    # --------------------------------------------------------
    # Larger rectangle surrounding die
    # --------------------------------------------------------

    outer_box = (

        center

        +

        1.30
        * (
            box - center
        )
    )


    outer = np.round(
        outer_box
    ).astype(np.int32)


    outer_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        outer_mask,
        outer,
        255
    )


    # Ring only
    ring_mask = cv2.subtract(
        outer_mask,
        inner_mask
    )


    inside_pixels = gray[
        inner_mask > 0
    ]


    outside_pixels = gray[
        ring_mask > 0
    ]


    if (
        len(inside_pixels) == 0
        or
        len(outside_pixels) == 0
    ):
        return 0.0


    inside_mean = float(
        np.mean(inside_pixels)
    )


    outside_mean = float(
        np.mean(outside_pixels)
    )


    return (
        inside_mean
        - outside_mean
    ) / 255.0


# ============================================================
# FIND THE WHOLE BOTTOM DIE
# ============================================================

def detect_bottom_die(original):
    """
    Detect the OUTER bright body of ViBottom.

    Important idea:

    We heavily blur the image first.

    This suppresses:
        - 16 circular spots
        - labels
        - local texture

    while keeping:
        - the large bright square die
        - dark surrounding background
    """

    gray = gray8(
        original
    )

    H, W = gray.shape[:2]

    D = min(H, W)


    image_center = np.array(
        [
            W / 2.0,
            H / 2.0
        ],
        dtype=np.float32
    )


    # ========================================================
    # HEAVY LOW-FREQUENCY BLUR
    # ========================================================

    sigma = max(
        12.0,
        D * 0.018
    )


    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )


    # ========================================================
    # TRY SEVERAL THRESHOLDS
    #
    # Much more robust than relying on only one Otsu value.
    # ========================================================

    otsu_value, otsu_mask = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    threshold_values = [
        float(otsu_value),
        float(np.percentile(smooth, 62)),
        float(np.percentile(smooth, 68)),
        float(np.percentile(smooth, 72)),
    ]


    all_candidates = []


    # ========================================================
    # MORPHOLOGY SIZE
    # ========================================================

    close_size = max(
        15,
        int(D * 0.025)
    )


    if close_size % 2 == 0:
        close_size += 1


    open_size = max(
        5,
        int(D * 0.008)
    )


    if open_size % 2 == 0:
        open_size += 1


    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            close_size,
            close_size
        )
    )


    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            open_size,
            open_size
        )
    )


    # ========================================================
    # TEST THRESHOLDS
    # ========================================================

    for threshold in threshold_values:

        _, mask = cv2.threshold(
            smooth,
            threshold,
            255,
            cv2.THRESH_BINARY
        )


        # ----------------------------------------------------
        # Remove internal holes / spots
        # ----------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=1
        )


        # ----------------------------------------------------
        # External contours only
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        for contour in contours:

            area = float(
                cv2.contourArea(
                    contour
                )
            )


            if area <= 0:
                continue


            # =================================================
            # CONVEX HULL
            #
            # Helps if one die edge was slightly fragmented.
            # =================================================

            hull = cv2.convexHull(
                contour
            )


            rect = cv2.minAreaRect(
                hull
            )


            (
                (cx, cy),
                (rw, rh),
                _
            ) = rect


            if rw < 10 or rh < 10:
                continue


            long_side = max(
                rw,
                rh
            )


            short_side = min(
                rw,
                rh
            )


            # =================================================
            # SIZE FILTER
            # =================================================

            side_ratio = (
                long_side / D
            )


            if (
                side_ratio < MIN_SIDE_RATIO
                or
                side_ratio > MAX_SIDE_RATIO
            ):
                continue


            # =================================================
            # SHOULD BE APPROXIMATELY SQUARE
            # =================================================

            aspect = (
                long_side
                / max(
                    short_side,
                    1.0
                )
            )


            if aspect > 1.55:
                continue


            # =================================================
            # CENTER FILTER
            # =================================================

            dx = abs(
                cx - image_center[0]
            ) / W


            dy = abs(
                cy - image_center[1]
            ) / H


            # fairly permissive!
            if dx > MAX_CENTER_OFFSET_X:
                continue


            if dy > MAX_CENTER_OFFSET_Y:
                continue


            # =================================================
            # RECTANGULAR FILL
            # =================================================

            rect_area = (
                rw * rh
            )


            fill = (
                area
                / max(
                    rect_area,
                    1.0
                )
            )


            fill = float(
                np.clip(
                    fill,
                    0.0,
                    1.0
                )
            )


            # =================================================
            # BOX POINTS
            # =================================================

            box = cv2.boxPoints(
                rect
            )


            # =================================================
            # BRIGHTNESS CONTRAST
            #
            # Very important for Bottom:
            #
            # die should be brighter than surrounding area.
            # =================================================

            contrast = rectangle_contrast(
                gray,
                box
            )


            # =================================================
            # SCORES
            # =================================================

            square_score = (
                1.0 / aspect
            )


            center_distance = np.hypot(
                dx,
                dy
            )


            max_center_distance = np.hypot(
                MAX_CENTER_OFFSET_X,
                MAX_CENTER_OFFSET_Y
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


            # Expected Bottom die around 25-35% image
            expected_ratio = 0.28


            size_score = (

                1.0

                - abs(
                    side_ratio
                    - expected_ratio
                )
                / expected_ratio
            )


            size_score = float(
                np.clip(
                    size_score,
                    0.0,
                    1.0
                )
            )


            # =================================================
            # FINAL SCORE
            #
            # Center is NOT dominant.
            #
            # Shape + brightness matter more because
            # Bottom may genuinely be displaced.
            # =================================================

            score = (

                3.5
                * contrast

                +

                2.5
                * square_score

                +

                2.0
                * fill

                +

                1.0
                * size_score

                +

                0.5
                * center_score
            )


            all_candidates.append(
                {
                    "score":
                        float(score),

                    "center":
                        np.array(
                            [cx, cy],
                            dtype=np.float32
                        ),

                    "width":
                        float(rw),

                    "height":
                        float(rh),

                    "box":
                        box.astype(
                            np.float32
                        ),

                    "contrast":
                        float(contrast),

                    "fill":
                        float(fill),
                }
            )


    # ========================================================
    # NOTHING FOUND
    # ========================================================

    if not all_candidates:
        return None


    # ========================================================
    # SELECT BEST DIE
    # ========================================================

    all_candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )


    return all_candidates[0]


# ============================================================
# BUILD A ROTATED SQUARE AROUND DIE
# ============================================================

def make_square_from_die(
    detection
):
    """
    Convert fitted rotated rectangle into rotated SQUARE.

    Prefer slightly more rather than less:
    side = max(width,height) + small offset.
    """

    box = order_points(
        detection["box"]
    )


    tl, tr, br, bl = box


    # ========================================================
    # ORIENTATION
    # ========================================================

    horizontal = (
        (tr - tl)
        +
        (br - bl)
    ) / 2.0


    hnorm = np.linalg.norm(
        horizontal
    )


    if hnorm < 1e-6:
        return None


    horizontal = (
        horizontal
        / hnorm
    )


    # ensure it points approximately right
    if horizontal[0] < 0:
        horizontal = -horizontal


    vertical = np.array(
        [
            -horizontal[1],
            horizontal[0]
        ],
        dtype=np.float32
    )


    if vertical[1] < 0:
        vertical = -vertical


    # ========================================================
    # SQUARE SIDE
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


    center = detection[
        "center"
    ]


    # ========================================================
    # ROTATED SQUARE
    # ========================================================

    tl2 = (
        center
        - horizontal * half
        - vertical * half
    )


    tr2 = (
        center
        + horizontal * half
        - vertical * half
    )


    br2 = (
        center
        + horizontal * half
        + vertical * half
    )


    bl2 = (
        center
        - horizontal * half
        + vertical * half
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
# CHECK CROP INSIDE IMAGE
# ============================================================

def inside_image(
    square,
    image
):

    H, W = image.shape[:2]


    xs = square[:, 0]
    ys = square[:, 1]


    return bool(

        np.all(xs >= 0)

        and

        np.all(xs < W)

        and

        np.all(ys >= 0)

        and

        np.all(ys < H)
    )


# ============================================================
# STRAIGHTEN ONLY THE DETECTED SMALL ANGLE
# ============================================================

def rectify_square(
    original,
    square,
    native_size
):

    source = order_points(
        square
    )


    native_size = max(
        100,
        int(
            round(
                native_size
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
        source,
        destination
    )


    crop = cv2.warpPerspective(
        original,
        transform,
        (
            native_size,
            native_size
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


    return crop


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


    if H > FINAL_SIZE:
        interpolation = cv2.INTER_AREA

    else:
        interpolation = cv2.INTER_CUBIC


    return cv2.resize(
        crop,
        (
            FINAL_SIZE,
            FINAL_SIZE
        ),
        interpolation=interpolation
    )


# ============================================================
# MILD LIGHTING
# ============================================================

def lighten_if_dark(
    image
):

    if image.ndim == 3:

        values = image.astype(
            np.float32
        ).mean(axis=2)

    else:

        values = image.astype(
            np.float32
        )


    if not np.issubdtype(
        image.dtype,
        np.integer
    ):
        return image.copy()


    max_value = float(
        np.iinfo(
            image.dtype
        ).max
    )


    brightness = float(
        np.mean(values)
        / max_value
    )


    DARK_THRESHOLD = 0.22


    if brightness >= DARK_THRESHOLD:
        return image.copy()


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


    # Only maximum 10% increase
    factor = (
        1.0
        +
        0.10 * darkness
    )


    result = (

        image.astype(
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
        image.dtype
    )


# ============================================================
# DEBUG
# ============================================================

def make_debug(
    original,
    square,
    detection,
    angle
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
    # GREEN = EXACT CROP
    # ========================================================

    cv2.polylines(
        debug,
        [
            points.reshape(
                (-1, 1, 2)
            )
        ],
        True,
        (0, 255, 0),
        4
    )


    center = np.round(
        detection["center"]
    ).astype(int)


    cv2.circle(
        debug,
        tuple(center),
        6,
        (0, 0, 255),
        -1
    )


    text = (

        f"score={detection['score']:.2f} "

        f"contrast={detection['contrast']:.2f} "

        f"angle={angle:.2f}"
    )


    cv2.putText(
        debug,
        text,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    return debug


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


    shutil.copy2(
        str(image_path),
        str(
            missing_folder
            / image_path.name
        )
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

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )


    if original is None:

        print(
            f"[ERROR] "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # DETECT WHOLE DIE
    # ========================================================

    detection = detect_bottom_die(
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
            f"| die not found"
        )

        return False


    if (
        detection["score"]
        < MIN_DETECTION_SCORE
    ):

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| score="
            f"{detection['score']:.2f}"
        )

        return False


    # ========================================================
    # ROTATION
    # ========================================================

    angle = rectangle_angle(
        detection["box"]
    )


    if abs(angle) > MAX_ROTATION_DEG:

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| suspicious angle="
            f"{angle:.2f}"
        )

        return False


    # ========================================================
    # SQUARE + SMALL OFFSET
    # ========================================================

    result = make_square_from_die(
        detection
    )


    if result is None:

        save_missing(
            image_path,
            missing_folder
        )

        return False


    square, square_side = result


    if not inside_image(
        square,
        original
    ):

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| crop outside image"
        )

        return False


    # ========================================================
    # DEBUG BEFORE CROP
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_debug(
        original,
        square,
        detection,
        angle
    )


    cv2.imwrite(
        str(
            debug_folder
            / (
                image_path.stem
                + "_debug.png"
            )
        ),
        debug
    )


    # ========================================================
    # STRAIGHTEN / CROP
    # ========================================================

    crop = rectify_square(
        original,
        square,
        square_side
    )


    # ========================================================
    # FINAL 600x600
    # ========================================================

    final = resize_to_600(
        crop
    )


    final = lighten_if_dark(
        final
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    cv2.imwrite(
        str(
            output_folder
            / image_path.name
        ),
        final
    )


    print(
        f"[OK] "
        f"{image_path.name} "
        f"| angle={angle:.2f}° "
        f"| side={square_side:.0f}px "
        f"| contrast={detection['contrast']:.3f} "
        f"| score={detection['score']:.2f} "
        f"| final=600x600"
    )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "ViBottom OUTER-DIE DETECTION TEST"
    )

    print(
        "No circles / No ECC / No reference"
    )

    print(
        "Final = 600x600"
    )

    print(
        "Originals remain untouched"
    )


    for label in LABELS:

        input_folder = (
            INPUT_ROOT
            / label
        )


        output_folder = (
            OUTPUT_ROOT
            / label
        )


        debug_folder = (
            OUTPUT_ROOT
            / "_debug"
            / label
        )


        missing_folder = (
            OUTPUT_ROOT
            / "Missing"
            / label
        )


        if not input_folder.exists():

            print(
                f"[MISSING FOLDER] "
                f"{input_folder}"
            )

            continue


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


            process_image(
                image_path,
                output_folder,
                debug_folder,
                missing_folder
            )


    print()
    print("DONE")


if __name__ == "__main__":
    main()