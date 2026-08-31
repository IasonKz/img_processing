from pathlib import Path
import shutil
import cv2
import numpy as np


# ============================================================
# PATHS - ViBottom ONLY
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka\ViBottom"
)

OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\ViBottom_crop_test"
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


# ------------------------------------------------------------
# OFFSET AROUND DIE
#
# Based on the 4x4 grid:
# outer die edge should be roughly half a grid-spacing
# outside the outer spot centers.
#
# EXTRA_OFFSET gives a little additional safety.
#
# 0.08 = 8% of one grid spacing on each side.
# Very small offset.
# ------------------------------------------------------------

EXTRA_OFFSET = 0.08


# ------------------------------------------------------------
# CIRCLE DETECTION
# ------------------------------------------------------------

MIN_CIRCLES_REQUIRED = 10

# Normally we expect ~16
MAX_CIRCLES_USED = 16


# ------------------------------------------------------------
# Missing / sanity
# ------------------------------------------------------------

MAX_ROTATION_DEG = 15.0

MIN_GRID_SCORE = 0.50


# ============================================================
# GRAYSCALE 8 BIT COPY
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
# DETECT CIRCULAR SPOTS
# ============================================================

def detect_spots(original):

    g = gray8(original)

    H, W = g.shape[:2]

    D = min(H, W)


    # --------------------------------------------------------
    # Improve contrast ONLY for detection
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    work = clahe.apply(g)

    work = cv2.medianBlur(
        work,
        5
    )


    # --------------------------------------------------------
    # Expected circle sizes
    #
    # For ~1600x1600 Bottom images this means roughly:
    # radius 13 ... 48 px
    # --------------------------------------------------------

    min_radius = max(
        6,
        int(D * 0.008)
    )

    max_radius = max(
        min_radius + 3,
        int(D * 0.030)
    )

    min_distance = max(
        20,
        int(D * 0.045)
    )


    circles = cv2.HoughCircles(
        work,
        cv2.HOUGH_GRADIENT,

        dp=1.2,

        minDist=min_distance,

        param1=100,

        param2=25,

        minRadius=min_radius,

        maxRadius=max_radius
    )


    if circles is None:
        return None


    circles = circles[0].astype(
        np.float32
    )


    # ========================================================
    # REMOVE EXTREME IMAGE-EDGE DETECTIONS
    # ========================================================

    margin_x = W * 0.08
    margin_y = H * 0.08


    valid = []

    for x, y, r in circles:

        if x < margin_x:
            continue

        if x > W - margin_x:
            continue

        if y < margin_y:
            continue

        if y > H - margin_y:
            continue

        valid.append(
            [x, y, r]
        )


    if len(valid) < MIN_CIRCLES_REQUIRED:
        return None


    return np.array(
        valid,
        dtype=np.float32
    )


# ============================================================
# NEAREST NEIGHBOUR SPACING
# ============================================================

def estimate_spacing(points):

    if len(points) < 2:
        return None


    distances = []


    for i in range(len(points)):

        d = np.linalg.norm(
            points - points[i],
            axis=1
        )

        d = d[
            d > 1e-6
        ]

        if len(d) == 0:
            continue


        distances.append(
            float(np.min(d))
        )


    if not distances:
        return None


    return float(
        np.median(distances)
    )


# ============================================================
# SELECT THE 4x4 GRID
# ============================================================

def select_grid(circles, image_shape):

    H, W = image_shape[:2]

    points = circles[:, :2]

    spacing = estimate_spacing(
        points
    )


    if spacing is None:
        return None


    # ========================================================
    # CHECK HOW MANY "GRID NEIGHBOURS"
    # EACH POINT HAS
    # ========================================================

    candidates = []


    image_center = np.array(
        [
            W / 2.0,
            H / 2.0
        ],
        dtype=np.float32
    )


    for i, p in enumerate(points):

        distances = np.linalg.norm(
            points - p,
            axis=1
        )


        # Expected nearest horizontal/vertical neighbours
        neighbour_mask = (

            (distances > spacing * 0.65)

            &

            (distances < spacing * 1.35)
        )


        neighbour_count = int(
            np.sum(neighbour_mask)
        )


        center_distance = float(
            np.linalg.norm(
                p - image_center
            )
        )


        candidates.append(
            (
                neighbour_count,
                -center_distance,
                i
            )
        )


    # Points that belong to a grid tend to have
    # 2, 3 or 4 neighbours.
    candidates.sort(
        reverse=True
    )


    selected_indices = [
        item[2]
        for item in candidates[
            :MAX_CIRCLES_USED
        ]
    ]


    selected = circles[
        selected_indices
    ]


    if len(selected) < MIN_CIRCLES_REQUIRED:
        return None


    return selected


# ============================================================
# ESTIMATE GRID ROTATION
# ============================================================

def estimate_grid_angle(points):

    spacing = estimate_spacing(
        points
    )


    if spacing is None:
        return None


    angles = []


    for i in range(len(points)):

        for j in range(
            i + 1,
            len(points)
        ):

            vector = (
                points[j]
                - points[i]
            )


            distance = float(
                np.linalg.norm(vector)
            )


            # Only nearest-neighbour-like vectors
            if not (
                spacing * 0.70
                <
                distance
                <
                spacing * 1.30
            ):
                continue


            angle = np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )


            # Grid is equivalent every 90 degrees.
            # Convert to roughly [-45,+45]
            angle = (
                (angle + 45.0)
                % 90.0
            ) - 45.0


            angles.append(
                angle
            )


    if len(angles) < 4:
        return 0.0


    return float(
        np.median(angles)
    )


# ============================================================
# GRID QUALITY
# ============================================================

def grid_quality(points):

    spacing = estimate_spacing(
        points
    )


    if spacing is None:
        return 0.0


    good = 0


    for i in range(len(points)):

        distances = np.linalg.norm(
            points - points[i],
            axis=1
        )


        neighbours = np.sum(

            (distances > spacing * 0.65)

            &

            (distances < spacing * 1.35)
        )


        if neighbours >= 2:
            good += 1


    return (
        good / float(len(points))
    )


# ============================================================
# ROTATE IMAGE AROUND DETECTED GRID CENTER
# ============================================================

def rotate_image_and_points(
    original,
    points,
    center,
    angle
):

    H, W = original.shape[:2]


    # IMPORTANT:
    # Using +angle here straightens the grid
    # because coordinates use image Y direction.
    M = cv2.getRotationMatrix2D(
        tuple(center),
        angle,
        1.0
    )


    rotated = cv2.warpAffine(
        original,
        M,
        (W, H),

        flags=cv2.INTER_LINEAR,

        borderMode=cv2.BORDER_REFLECT_101
    )


    ones = np.ones(
        (len(points), 1),
        dtype=np.float32
    )


    homogeneous = np.hstack(
        [
            points.astype(np.float32),
            ones
        ]
    )


    rotated_points = (

        M
        @ homogeneous.T

    ).T


    return (
        rotated,
        rotated_points,
        M
    )


# ============================================================
# CALCULATE TIGHT DIE CROP FROM 4x4 GRID
# ============================================================

def calculate_die_crop(
    rotated_points,
    image_shape
):

    H, W = image_shape[:2]


    xs = rotated_points[:, 0]
    ys = rotated_points[:, 1]


    # ========================================================
    # GRID SPACING AFTER ROTATION
    # ========================================================

    spacing = estimate_spacing(
        rotated_points
    )


    if spacing is None:
        return None


    # ========================================================
    # CENTER OF GRID
    #
    # Use bounding midpoint rather than mean.
    # More robust if 1-2 circles were missed.
    # ========================================================

    cx = (
        np.min(xs)
        +
        np.max(xs)
    ) / 2.0


    cy = (
        np.min(ys)
        +
        np.max(ys)
    ) / 2.0


    # ========================================================
    # OUTER SPOT SPAN
    # ========================================================

    span_x = (
        np.max(xs)
        -
        np.min(xs)
    )


    span_y = (
        np.max(ys)
        -
        np.min(ys)
    )


    # ========================================================
    # DIE SIDE
    #
    # The 4 outer spot centers span about 3 spacings.
    #
    # Die extends roughly half a spacing outside
    # the first and last spot:
    #
    #        3*d + d = 4*d
    #
    # We calculate using both observed span and spacing.
    # Prefer slightly MORE rather than less.
    # ========================================================

    observed_span = max(
        span_x,
        span_y
    )


    die_side = max(

        observed_span
        + spacing,

        4.0 * spacing
    )


    # Very small extra offset
    crop_side = (

        die_side

        +

        2.0
        * EXTRA_OFFSET
        * spacing
    )


    crop_side = int(
        round(crop_side)
    )


    # ========================================================
    # CROP COORDINATES
    # ========================================================

    x1 = int(
        round(
            cx
            - crop_side / 2
        )
    )

    y1 = int(
        round(
            cy
            - crop_side / 2
        )
    )


    x2 = (
        x1
        + crop_side
    )

    y2 = (
        y1
        + crop_side
    )


    # ========================================================
    # CHECK IMAGE BOUNDS
    # ========================================================

    if (
        x1 < 0
        or
        y1 < 0
        or
        x2 > W
        or
        y2 > H
    ):
        return None


    return (
        x1,
        y1,
        x2,
        y2,
        spacing
    )


# ============================================================
# FINAL RESIZE 600x600
# ============================================================

def resize_to_600(img):

    H, W = img.shape[:2]


    if (
        H == FINAL_SIZE
        and
        W == FINAL_SIZE
    ):
        return img.copy()


    if (
        H > FINAL_SIZE
        or
        W > FINAL_SIZE
    ):
        interpolation = cv2.INTER_AREA

    else:
        interpolation = cv2.INTER_CUBIC


    return cv2.resize(
        img,
        (
            FINAL_SIZE,
            FINAL_SIZE
        ),
        interpolation=interpolation
    )


# ============================================================
# MILD LIGHTING
# ============================================================

def lighten_if_dark(img):

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
        return img.copy()


    brightness = float(
        np.mean(values)
        / max_value
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


    # Max ~10% brightening
    factor = (
        1.0
        +
        0.10 * darkness
    )


    result = (

        img.astype(np.float32)
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


# ============================================================
# DEBUG IMAGE
# ============================================================

def make_debug(
    original,
    circles,
    bbox,
    angle
):

    g = gray8(original)

    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    # --------------------------------------------------------
    # Draw detected grid circles
    # --------------------------------------------------------

    for x, y, r in circles:

        cv2.circle(
            debug,
            (
                int(round(x)),
                int(round(y))
            ),
            int(round(r)),
            (255, 0, 0),
            2
        )


        cv2.circle(
            debug,
            (
                int(round(x)),
                int(round(y))
            ),
            3,
            (0, 0, 255),
            -1
        )


    # --------------------------------------------------------
    # Final crop rectangle
    #
    # Note:
    # bbox belongs to ROTATED image,
    # so this debug is mainly used for spot detection.
    # --------------------------------------------------------

    cv2.putText(
        debug,
        f"spots={len(circles)} angle={angle:.2f} deg",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    return debug


def make_rotated_debug(
    rotated,
    rotated_points,
    bbox
):

    g = gray8(rotated)

    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    for p in rotated_points:

        cv2.circle(
            debug,
            (
                int(round(p[0])),
                int(round(p[1]))
            ),
            6,
            (255, 0, 0),
            -1
        )


    x1, y1, x2, y2, spacing = bbox


    # THIS green box is exactly what is cropped
    cv2.rectangle(
        debug,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        4
    )


    cv2.putText(
        debug,
        f"spacing={spacing:.1f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
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


    destination = (
        missing_folder
        / image_path.name
    )


    # COPY ONLY
    shutil.copy2(
        str(image_path),
        str(destination)
    )


# ============================================================
# PROCESS ONE BOTTOM IMAGE
# ============================================================

def process_bottom(
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
            f"[ERROR] {image_path.name}"
        )

        return False


    # ========================================================
    # 1. CIRCLES
    # ========================================================

    circles = detect_spots(
        original
    )


    if circles is None:

        save_missing(
            image_path,
            missing_folder
        )

        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| not enough spots"
        )

        return False


    # ========================================================
    # 2. SELECT 4x4 GRID
    # ========================================================

    grid = select_grid(
        circles,
        original.shape
    )


    if grid is None:

        save_missing(
            image_path,
            missing_folder
        )

        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| grid not found"
        )

        return False


    points = grid[:, :2]


    # ========================================================
    # 3. VALIDATE GRID
    # ========================================================

    quality = grid_quality(
        points
    )


    if quality < MIN_GRID_SCORE:

        save_missing(
            image_path,
            missing_folder
        )

        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| bad grid quality "
            f"{quality:.2f}"
        )

        return False


    # ========================================================
    # 4. GRID CENTER
    # ========================================================

    rect = cv2.minAreaRect(
        points.astype(
            np.float32
        )
    )


    grid_center = np.array(
        rect[0],
        dtype=np.float32
    )


    # ========================================================
    # 5. SMALL ROTATION ANGLE
    # ========================================================

    angle = estimate_grid_angle(
        points
    )


    if angle is None:
        angle = 0.0


    if abs(angle) > MAX_ROTATION_DEG:

        save_missing(
            image_path,
            missing_folder
        )

        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| angle={angle:.2f}"
        )

        return False


    # ========================================================
    # 6. ROTATE AROUND ACTUAL GRID CENTER
    # ========================================================

    (
        rotated,
        rotated_points,
        M
    ) = rotate_image_and_points(

        original,
        points,
        grid_center,
        angle
    )


    # ========================================================
    # 7. CALCULATE CROP DIRECTLY FROM 4x4 GRID
    # ========================================================

    bbox = calculate_die_crop(
        rotated_points,
        rotated.shape
    )


    if bbox is None:

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


    x1, y1, x2, y2, spacing = bbox


    # ========================================================
    # 8. EXACT CROP
    # ========================================================

    crop = rotated[
        y1:y2,
        x1:x2
    ].copy()


    # ========================================================
    # 9. 600x600
    # ========================================================

    final = resize_to_600(
        crop
    )


    # ========================================================
    # 10. MILD LIGHTING
    # ========================================================

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


    output_path = (
        output_folder
        / image_path.name
    )


    cv2.imwrite(
        str(output_path),
        final
    )


    # ========================================================
    # DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    before_debug = make_debug(
        original,
        grid,
        bbox,
        angle
    )


    after_debug = make_rotated_debug(
        rotated,
        rotated_points,
        bbox
    )


    cv2.imwrite(
        str(
            debug_folder
            / f"{image_path.stem}_01_spots.png"
        ),
        before_debug
    )


    cv2.imwrite(
        str(
            debug_folder
            / f"{image_path.stem}_02_crop.png"
        ),
        after_debug
    )


    print(
        f"[OK] "
        f"{image_path.name} "
        f"| spots={len(grid)} "
        f"| grid={quality:.2f} "
        f"| angle={angle:.2f}° "
        f"| spacing={spacing:.1f} "
        f"| crop={x2-x1}px "
        f"| final=600x600"
    )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "ViBottom GRID-BASED CROP TEST"
    )

    print(
        "Uses 4x4 circular spot pattern"
    )

    print(
        "Final output = 600x600"
    )

    print(
        "Original images remain untouched"
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
                f"[NO FOLDER] "
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


            process_bottom(
                image_path,
                output_folder,
                debug_folder,
                missing_folder
            )


    print()
    print("DONE")


if __name__ == "__main__":
    main()