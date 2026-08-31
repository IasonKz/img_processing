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
    r"C:\Users\laka\software\classification_laka_processed_combined"
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
# COMMON SETTINGS
# ============================================================

FINAL_SIZE = 600


# ViTop offset
TOP_OFFSET_RATIO = 0.04


# ViBottom offset
BOTTOM_OFFSET_RATIO = 0.03


# Lighting
DARK_THRESHOLD = 0.22

# Maximum brightness increase = 10%
MAX_BRIGHTEN = 0.10


# ============================================================
# TOP SETTINGS
# ============================================================

TOP_SEARCH_RATIO = 0.62

TOP_MAX_CENTER_OFFSET_X = 0.20
TOP_MAX_CENTER_OFFSET_Y = 0.20

TOP_MIN_SIZE_RATIO = 0.08
TOP_MAX_SIZE_RATIO = 0.42

TOP_EXPECTED_SIZE_RATIO = 0.18

TOP_MIN_DETECTION_SCORE = 7.0

TOP_MAX_ROTATION_DEG = 15.0


# ============================================================
# BOTTOM SETTINGS
# ============================================================

BOTTOM_MAX_CENTER_OFFSET_X = 0.34
BOTTOM_MAX_CENTER_OFFSET_Y = 0.34

BOTTOM_MIN_SIDE_RATIO = 0.14
BOTTOM_MAX_SIDE_RATIO = 0.52

BOTTOM_MAX_ROTATION_DEG = 18.0

BOTTOM_MIN_DETECTION_SCORE = 3.0

# Search around expected edge
BOTTOM_EDGE_SEARCH_RATIO = 0.28


# ============================================================
# COMMON HELPERS
# ============================================================

def gray8(img):
    """
    Create an 8-bit grayscale COPY.

    Used only for:
        detection
        debug

    ORIGINAL IS NEVER MODIFIED.
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


    x = g.astype(
        np.float32
    )


    lo = float(
        np.percentile(
            x,
            1
        )
    )


    hi = float(
        np.percentile(
            x,
            99
        )
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
    ).astype(
        np.uint8
    )


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


    sums = pts.sum(
        axis=1
    )


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
# ROTATION ANGLE
# ============================================================

def rectangle_angle(box):

    box = order_points(
        box
    )


    tl, tr, br, bl = box


    vector = (
        tr - tl
    )


    angle = np.degrees(
        np.arctan2(
            vector[1],
            vector[0]
        )
    )


    # Square repeats every 90°
    while angle >= 45:

        angle -= 90


    while angle < -45:

        angle += 90


    return float(
        angle
    )


# ============================================================
# FINAL RESIZE
# ============================================================

def resize_to_600(image):

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
# MILD LIGHTING
# ============================================================

def lighten_if_dark(image):
    """
    Very mild lighting correction.

    Maximum brightness increase:
        10%

    Bright images remain unchanged.
    """

    if not np.issubdtype(
        image.dtype,
        np.integer
    ):

        return image.copy()


    if image.ndim == 3:

        values = image.astype(
            np.float32
        ).mean(
            axis=2
        )

    else:

        values = image.astype(
            np.float32
        )


    max_value = float(
        np.iinfo(
            image.dtype
        ).max
    )


    brightness = float(
        np.mean(values)
        /
        max_value
    )


    if brightness >= DARK_THRESHOLD:

        return image.copy()


    darkness = (
        DARK_THRESHOLD
        -
        brightness
    ) / DARK_THRESHOLD


    darkness = float(
        np.clip(
            darkness,
            0.0,
            1.0
        )
    )


    factor = (
        1.0
        +
        MAX_BRIGHTEN
        * darkness
    )


    result = (
        image.astype(
            np.float32
        )
        *
        factor
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
# MISSING
# ============================================================

def save_missing(
    image_path,
    original,
    missing_folder,
    missing_debug_folder,
    reason
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
    # COPY ONLY
    # ========================================================

    destination = (
        missing_folder
        /
        image_path.name
    )


    shutil.copy2(
        str(image_path),
        str(destination)
    )


    # ========================================================
    # DEBUG MISSING
    # ========================================================

    g = gray8(
        original
    )


    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    cv2.putText(
        debug,
        reason,
        (
            30,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (
            0,
            0,
            255
        ),
        3,
        cv2.LINE_AA
    )


    debug_path = (
        missing_debug_folder
        /
        (
            image_path.stem
            +
            "_missing.png"
        )
    )


    cv2.imwrite(
        str(debug_path),
        debug
    )


# ============================================================
#
#                 ViTop METHOD
#
# ============================================================


# ============================================================
# DETECT TOP DIE
# ============================================================

def detect_top_die(original):

    g = gray8(
        original
    )


    H, W = g.shape[:2]


    image_cx = W / 2.0
    image_cy = H / 2.0


    # ========================================================
    # CENTRAL SEARCH
    # ========================================================

    search_w = int(
        W * TOP_SEARCH_RATIO
    )


    search_h = int(
        H * TOP_SEARCH_RATIO
    )


    sx1 = max(
        0,
        int(
            image_cx
            -
            search_w / 2
        )
    )


    sy1 = max(
        0,
        int(
            image_cy
            -
            search_h / 2
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
    # PREPROCESS
    # ========================================================

    blurred = cv2.GaussianBlur(
        roi,
        (
            5,
            5
        ),
        0
    )


    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(
            8,
            8
        )
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
        +
        cv2.THRESH_OTSU
    )


    # ========================================================
    # EDGES
    # ========================================================

    median = float(
        np.median(
            enhanced
        )
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


    kernel9 = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            9,
            9
        )
    )


    kernel5 = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            5,
            5
        )
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
    # SCORE CANDIDATES
    # ========================================================

    for contour in contours:


        area = float(
            cv2.contourArea(
                contour
            )
        )


        if area <= 0:

            continue


        rect = cv2.minAreaRect(
            contour
        )


        box = cv2.boxPoints(
            rect
        )


        # ROI coordinates -> full image
        box[:, 0] += sx1
        box[:, 1] += sy1


        box = order_points(
            box
        )


        tl, tr, br, bl = box


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
            top
            +
            bottom
        ) / 2.0


        height = (
            left
            +
            right
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


        aspect = (
            long_side
            /
            short_side
        )


        if aspect > 1.70:

            continue


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


        dx = (
            abs(
                cx
                -
                image_cx
            )
            /
            W
        )


        dy = (
            abs(
                cy
                -
                image_cy
            )
            /
            H
        )


        # ====================================================
        # HARD REJECT SIDE TARGETS
        # ====================================================

        if dx > TOP_MAX_CENTER_OFFSET_X:

            continue


        if dy > TOP_MAX_CENTER_OFFSET_Y:

            continue


        size_ratio = 0.5 * (

            width / W

            +

            height / H
        )


        if (
            size_ratio < TOP_MIN_SIZE_RATIO
            or
            size_ratio > TOP_MAX_SIZE_RATIO
        ):

            continue


        center_distance = np.sqrt(
            dx * dx
            +
            dy * dy
        )


        max_distance = np.sqrt(

            TOP_MAX_CENTER_OFFSET_X ** 2

            +

            TOP_MAX_CENTER_OFFSET_Y ** 2
        )


        center_score = (

            1.0

            -

            center_distance
            /
            max_distance
        )


        center_score = float(
            np.clip(
                center_score,
                0.0,
                1.0
            )
        )


        square_score = (
            1.0
            /
            aspect
        )


        rect_area = (
            width
            *
            height
        )


        fill_score = min(

            area
            /
            max(
                rect_area,
                1.0
            ),

            1.0
        )


        size_score = (

            1.0

            -

            abs(
                size_ratio
                -
                TOP_EXPECTED_SIZE_RATIO
            )
            /
            TOP_EXPECTED_SIZE_RATIO
        )


        size_score = float(
            np.clip(
                size_score,
                0.0,
                1.0
            )
        )


        score = (

            10.0
            *
            center_score

            +

            2.0
            *
            square_score

            +

            1.5
            *
            fill_score

            +

            1.5
            *
            size_score
        )


        if score > best_score:


            best_score = score


            best = {

                "center":
                    np.array(
                        [
                            cx,
                            cy
                        ],
                        dtype=np.float32
                    ),

                "box":
                    box.copy(),

                "width":
                    float(width),

                "height":
                    float(height),

                "angle":
                    rectangle_angle(
                        box
                    ),

                "score":
                    float(score),
            }


    return best


# ============================================================
# BUILD TOP ROTATED SQUARE
# ============================================================

def build_top_square(detection):

    box = order_points(
        detection["box"]
    )


    tl, tr, br, bl = box


    center = detection[
        "center"
    ]


    # Average horizontal direction
    u = (

        (tr - tl)

        +

        (br - bl)

    ) / 2.0


    norm = np.linalg.norm(
        u
    )


    if norm <= 1e-6:

        return None


    u = (
        u / norm
    )


    if u[0] < 0:

        u = -u


    v = np.array(
        [
            -u[1],
            u[0]
        ],
        dtype=np.float32
    )


    if v[1] < 0:

        v = -v


    die_side = max(
        detection["width"],
        detection["height"]
    )


    square_side = (

        die_side

        *

        (
            1.0
            +
            2.0
            *
            TOP_OFFSET_RATIO
        )
    )


    half = (
        square_side
        /
        2.0
    )


    tl2 = (
        center
        -
        u * half
        -
        v * half
    )


    tr2 = (
        center
        +
        u * half
        -
        v * half
    )


    br2 = (
        center
        +
        u * half
        +
        v * half
    )


    bl2 = (
        center
        -
        u * half
        +
        v * half
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
# CHECK ROTATED SQUARE
# ============================================================

def square_inside_image(
    square,
    image
):

    H, W = image.shape[:2]


    xs = square[:, 0]

    ys = square[:, 1]


    return bool(

        np.all(
            xs >= 0
        )

        and

        np.all(
            xs < W
        )

        and

        np.all(
            ys >= 0
        )

        and

        np.all(
            ys < H
        )
    )


# ============================================================
# RECTIFY TOP
# ============================================================

def rectify_top(
    original,
    square,
    square_side
):

    source = order_points(
        square
    )


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
            [
                0,
                0
            ],

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
# TOP DEBUG
# ============================================================

def make_top_debug(
    original,
    square,
    detection
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


    # Green = exact crop
    cv2.polylines(
        debug,
        [
            points.reshape(
                (
                    -1,
                    1,
                    2
                )
            )
        ],
        True,
        (
            0,
            255,
            0
        ),
        4
    )


    center = tuple(
        np.round(
            detection["center"]
        ).astype(int)
    )


    cv2.circle(
        debug,
        center,
        6,
        (
            0,
            0,
            255
        ),
        -1
    )


    cv2.putText(
        debug,
        (
            f"TOP score="
            f"{detection['score']:.2f} "
            f"angle="
            f"{detection['angle']:.2f}"
        ),
        (
            30,
            50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (
            0,
            255,
            0
        ),
        2,
        cv2.LINE_AA
    )


    return debug


# ============================================================
# PROCESS TOP
# ============================================================

def process_top(
    image_path,
    output_folder,
    debug_folder,
    missing_folder,
    missing_debug_folder
):

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )


    if original is None:

        print(
            f"[TOP ERROR] "
            f"{image_path.name}"
        )

        return False


    detection = detect_top_die(
        original
    )


    if detection is None:

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "TOP MISSING - NO DIE"
        )


        print(
            f"[TOP MISSING] "
            f"{image_path.name} "
            f"| no die"
        )

        return False


    if (
        detection["score"]
        <
        TOP_MIN_DETECTION_SCORE
    ):

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "TOP MISSING - LOW SCORE"
        )


        return False


    if (
        abs(
            detection["angle"]
        )
        >
        TOP_MAX_ROTATION_DEG
    ):

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "TOP MISSING - ROTATION"
        )


        return False


    result = build_top_square(
        detection
    )


    if result is None:

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "TOP MISSING - GEOMETRY"
        )

        return False


    square, square_side = result


    if not square_inside_image(
        square,
        original
    ):

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "TOP MISSING - OUTSIDE"
        )

        return False


    # ========================================================
    # DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_top_debug(
        original,
        square,
        detection
    )


    cv2.imwrite(
        str(
            debug_folder
            /
            (
                image_path.stem
                +
                "_debug.png"
            )
        ),
        debug
    )


    # ========================================================
    # CROP + STRAIGHTEN
    # ========================================================

    crop = rectify_top(
        original,
        square,
        square_side
    )


    final = resize_to_600(
        crop
    )


    final = lighten_if_dark(
        final
    )


    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        output_folder
        /
        image_path.name
    )


    success = cv2.imwrite(
        str(output_path),
        final
    )


    if not success:

        return False


    print(
        f"[TOP OK] "
        f"{image_path.name} "
        f"| angle="
        f"{detection['angle']:.2f}° "
        f"| crop="
        f"{square_side:.0f}px "
        f"| 600x600"
    )


    return True


# ============================================================
#
#               ViBottom METHOD
#
# ============================================================


# ============================================================
# CONTRAST OF BOTTOM CANDIDATE
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


    inner_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        inner_mask,
        np.round(
            box
        ).astype(
            np.int32
        ),
        255
    )


    outer_box = (

        center

        +

        1.28
        *
        (
            box
            -
            center
        )
    )


    outer_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        outer_mask,
        np.round(
            outer_box
        ).astype(
            np.int32
        ),
        255
    )


    ring_mask = cv2.subtract(
        outer_mask,
        inner_mask
    )


    inside = gray[
        inner_mask > 0
    ]


    outside = gray[
        ring_mask > 0
    ]


    if (
        len(inside) == 0
        or
        len(outside) == 0
    ):

        return 0.0


    return (

        float(
            np.mean(
                inside
            )
        )

        -

        float(
            np.mean(
                outside
            )
        )

    ) / 255.0


# ============================================================
# ROUGH BOTTOM DETECTION
# ============================================================

def detect_bottom_die(original):

    gray = gray8(
        original
    )


    H, W = gray.shape[:2]


    D = min(
        H,
        W
    )


    image_center = np.array(
        [
            W / 2.0,
            H / 2.0
        ],
        dtype=np.float32
    )


    # ========================================================
    # HEAVY BLUR
    # ========================================================

    sigma = max(
        12.0,
        D * 0.018
    )


    smooth = cv2.GaussianBlur(
        gray,
        (
            0,
            0
        ),
        sigmaX=sigma,
        sigmaY=sigma
    )


    # ========================================================
    # MULTIPLE THRESHOLDS
    # ========================================================

    otsu_value, _ = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )


    thresholds = [

        float(
            otsu_value
        ),

        float(
            np.percentile(
                smooth,
                60
            )
        ),

        float(
            np.percentile(
                smooth,
                66
            )
        ),

        float(
            np.percentile(
                smooth,
                72
            )
        ),
    ]


    close_size = max(
        15,
        int(
            D * 0.025
        )
    )


    if close_size % 2 == 0:

        close_size += 1


    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            close_size,
            close_size
        )
    )


    candidates = []


    for threshold in thresholds:


        _, mask = cv2.threshold(
            smooth,
            threshold,
            255,
            cv2.THRESH_BINARY
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )


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


            hull = cv2.convexHull(
                contour
            )


            rect = cv2.minAreaRect(
                hull
            )


            (
                (
                    cx,
                    cy
                ),
                (
                    rw,
                    rh
                ),
                _
            ) = rect


            if (
                rw < 10
                or
                rh < 10
            ):

                continue


            long_side = max(
                rw,
                rh
            )


            short_side = min(
                rw,
                rh
            )


            side_ratio = (
                long_side
                /
                D
            )


            if not (
                BOTTOM_MIN_SIDE_RATIO
                <=
                side_ratio
                <=
                BOTTOM_MAX_SIDE_RATIO
            ):

                continue


            aspect = (
                long_side
                /
                max(
                    short_side,
                    1.0
                )
            )


            if aspect > 1.55:

                continue


            dx = (
                abs(
                    cx
                    -
                    image_center[0]
                )
                /
                W
            )


            dy = (
                abs(
                    cy
                    -
                    image_center[1]
                )
                /
                H
            )


            # Bottom can be displaced
            if dx > BOTTOM_MAX_CENTER_OFFSET_X:

                continue


            if dy > BOTTOM_MAX_CENTER_OFFSET_Y:

                continue


            box = cv2.boxPoints(
                rect
            ).astype(
                np.float32
            )


            contrast = rectangle_contrast(
                gray,
                box
            )


            rect_area = (
                rw
                *
                rh
            )


            fill = float(
                np.clip(

                    area
                    /
                    max(
                        rect_area,
                        1.0
                    ),

                    0.0,
                    1.0
                )
            )


            square_score = (
                1.0
                /
                aspect
            )


            expected_ratio = 0.28


            size_score = (

                1.0

                -

                abs(
                    side_ratio
                    -
                    expected_ratio
                )
                /
                expected_ratio
            )


            size_score = float(
                np.clip(
                    size_score,
                    0.0,
                    1.0
                )
            )


            score = (

                3.5
                *
                contrast

                +

                2.5
                *
                square_score

                +

                2.0
                *
                fill

                +

                1.0
                *
                size_score
            )


            candidates.append(
                {

                    "center":
                        np.array(
                            [
                                cx,
                                cy
                            ],
                            dtype=np.float32
                        ),

                    "width":
                        float(
                            rw
                        ),

                    "height":
                        float(
                            rh
                        ),

                    "box":
                        box,

                    "score":
                        float(
                            score
                        ),
                }
            )


    if not candidates:

        return None


    candidates.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )


    return candidates[0]


# ============================================================
# ROTATE BOTTOM
# ============================================================

def rotate_bottom_image(
    original,
    center,
    angle
):

    H, W = original.shape[:2]


    matrix = cv2.getRotationMatrix2D(
        (
            float(
                center[0]
            ),
            float(
                center[1]
            )
        ),
        angle,
        1.0
    )


    rotated = cv2.warpAffine(
        original,
        matrix,
        (
            W,
            H
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


    return rotated


# ============================================================
# FIND PROFILE EDGE
# ============================================================

def find_profile_edge(
    profile,
    expected_position,
    search_radius,
    positive
):

    profile = profile.astype(
        np.float32
    )


    size = max(
        9,
        int(
            len(profile)
            *
            0.01
        )
    )


    if size % 2 == 0:

        size += 1


    smooth = cv2.GaussianBlur(
        profile.reshape(
            1,
            -1
        ),
        (
            size,
            1
        ),
        0
    ).reshape(
        -1
    )


    gradient = np.gradient(
        smooth
    )


    low = max(
        1,
        int(
            expected_position
            -
            search_radius
        )
    )


    high = min(
        len(profile) - 2,
        int(
            expected_position
            +
            search_radius
        )
    )


    if high <= low:

        return None


    local = gradient[
        low:
        high + 1
    ]


    if positive:

        index = (

            low

            +

            int(
                np.argmax(
                    local
                )
            )
        )


    else:

        index = (

            low

            +

            int(
                np.argmin(
                    local
                )
            )
        )


    return index


# ============================================================
# REFINE BOTTOM EDGES + RECENTER
# ============================================================

def refine_bottom_edges(
    rotated,
    approx_center,
    approx_side
):

    gray = gray8(
        rotated
    )


    H, W = gray.shape[:2]


    cx = float(
        approx_center[0]
    )


    cy = float(
        approx_center[1]
    )


    # ========================================================
    # BLUR INNER DETAILS
    # ========================================================

    sigma = max(
        8.0,
        approx_side * 0.035
    )


    smooth = cv2.GaussianBlur(
        gray,
        (
            0,
            0
        ),
        sigmaX=sigma,
        sigmaY=sigma
    )


    # ========================================================
    # CENTRAL STRIPS ONLY
    # ========================================================

    band_half = int(
        approx_side
        *
        0.30
    )


    y1 = max(
        0,
        int(
            cy
            -
            band_half
        )
    )


    y2 = min(
        H,
        int(
            cy
            +
            band_half
        )
    )


    x1 = max(
        0,
        int(
            cx
            -
            band_half
        )
    )


    x2 = min(
        W,
        int(
            cx
            +
            band_half
        )
    )


    if (
        y2 <= y1
        or
        x2 <= x1
    ):

        return None


    # ========================================================
    # HORIZONTAL PROFILE
    # ========================================================

    x_profile = np.mean(
        smooth[
            y1:y2,
            :
        ],
        axis=0
    )


    # ========================================================
    # VERTICAL PROFILE
    # ========================================================

    y_profile = np.mean(
        smooth[
            :,
            x1:x2
        ],
        axis=1
    )


    expected_left = (
        cx
        -
        approx_side / 2
    )


    expected_right = (
        cx
        +
        approx_side / 2
    )


    expected_top = (
        cy
        -
        approx_side / 2
    )


    expected_bottom = (
        cy
        +
        approx_side / 2
    )


    search_radius = (
        approx_side
        *
        BOTTOM_EDGE_SEARCH_RATIO
    )


    # ========================================================
    # FOUR REAL EDGES
    # ========================================================

    left = find_profile_edge(
        x_profile,
        expected_left,
        search_radius,
        positive=True
    )


    right = find_profile_edge(
        x_profile,
        expected_right,
        search_radius,
        positive=False
    )


    top = find_profile_edge(
        y_profile,
        expected_top,
        search_radius,
        positive=True
    )


    bottom = find_profile_edge(
        y_profile,
        expected_bottom,
        search_radius,
        positive=False
    )


    if (
        left is None
        or
        right is None
        or
        top is None
        or
        bottom is None
    ):

        return None


    span_x = (
        right
        -
        left
    )


    span_y = (
        bottom
        -
        top
    )


    # ========================================================
    # SANITY CHECK
    # ========================================================

    if (
        span_x < approx_side * 0.65
        or
        span_x > approx_side * 1.35
    ):

        return None


    if (
        span_y < approx_side * 0.65
        or
        span_y > approx_side * 1.35
    ):

        return None


    # ========================================================
    # TRUE CENTER
    # ========================================================

    refined_cx = (
        left
        +
        right
    ) / 2.0


    refined_cy = (
        top
        +
        bottom
    ) / 2.0


    die_side = max(
        span_x,
        span_y
    )


    return {

        "center":
            np.array(
                [
                    refined_cx,
                    refined_cy
                ],
                dtype=np.float32
            ),

        "side":
            float(
                die_side
            ),

        "edges":
            (
                int(
                    left
                ),
                int(
                    top
                ),
                int(
                    right
                ),
                int(
                    bottom
                )
            ),
    }


# ============================================================
# CROP BOTTOM
# ============================================================

def crop_bottom(
    rotated,
    refined
):

    H, W = rotated.shape[:2]


    cx = float(
        refined["center"][0]
    )


    cy = float(
        refined["center"][1]
    )


    side = int(
        round(

            refined["side"]

            *

            (
                1.0
                +
                2.0
                *
                BOTTOM_OFFSET_RATIO
            )
        )
    )


    x1 = int(
        round(
            cx
            -
            side / 2
        )
    )


    y1 = int(
        round(
            cy
            -
            side / 2
        )
    )


    x2 = (
        x1
        +
        side
    )


    y2 = (
        y1
        +
        side
    )


    if (
        x1 < 0
        or
        y1 < 0
        or
        x2 > W
        or
        y2 > H
    ):

        return (
            None,
            None
        )


    crop = rotated[
        y1:y2,
        x1:x2
    ].copy()


    return (
        crop,
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# BOTTOM DEBUG
# ============================================================

def make_bottom_debug(
    rotated,
    refined,
    crop_bbox
):

    g = gray8(
        rotated
    )


    debug = cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


    (
        left,
        top,
        right,
        bottom
    ) = refined[
        "edges"
    ]


    # ========================================================
    # YELLOW = ACTUAL DETECTED DIE
    # ========================================================

    cv2.rectangle(
        debug,
        (
            left,
            top
        ),
        (
            right,
            bottom
        ),
        (
            0,
            255,
            255
        ),
        3
    )


    (
        x1,
        y1,
        x2,
        y2
    ) = crop_bbox


    # ========================================================
    # GREEN = EXACT FINAL CROP
    # ========================================================

    cv2.rectangle(
        debug,
        (
            x1,
            y1
        ),
        (
            x2 - 1,
            y2 - 1
        ),
        (
            0,
            255,
            0
        ),
        4
    )


    center = tuple(
        np.round(
            refined["center"]
        ).astype(int)
    )


    # ========================================================
    # RED = REFINED CENTER
    # ========================================================

    cv2.circle(
        debug,
        center,
        6,
        (
            0,
            0,
            255
        ),
        -1
    )


    return debug


# ============================================================
# PROCESS BOTTOM
# ============================================================

def process_bottom(
    image_path,
    output_folder,
    debug_folder,
    missing_folder,
    missing_debug_folder
):

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )


    if original is None:

        print(
            f"[BOTTOM ERROR] "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # 1. ROUGH DETECTION
    # ========================================================

    detection = detect_bottom_die(
        original
    )


    if detection is None:

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "BOTTOM MISSING - NO DIE"
        )


        print(
            f"[BOTTOM MISSING] "
            f"{image_path.name} "
            f"| no die"
        )

        return False


    if (
        detection["score"]
        <
        BOTTOM_MIN_DETECTION_SCORE
    ):

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "BOTTOM MISSING - LOW SCORE"
        )

        return False


    # ========================================================
    # 2. SMALL ROTATION
    # ========================================================

    angle = rectangle_angle(
        detection["box"]
    )


    if (
        abs(angle)
        >
        BOTTOM_MAX_ROTATION_DEG
    ):

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "BOTTOM MISSING - ROTATION"
        )

        return False


    rotated = rotate_bottom_image(
        original,
        detection["center"],
        angle
    )


    approx_center = (
        detection["center"]
    )


    approx_side = max(
        detection["width"],
        detection["height"]
    )


    # ========================================================
    # 3. TRUE EDGE DETECTION / RECENTER
    # ========================================================

    refined = refine_bottom_edges(
        rotated,
        approx_center,
        approx_side
    )


    if refined is None:

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "BOTTOM MISSING - EDGE REFINE"
        )


        print(
            f"[BOTTOM MISSING] "
            f"{image_path.name} "
            f"| edge refinement failed"
        )

        return False


    # ========================================================
    # 4. FINAL TIGHT CROP
    # ========================================================

    (
        crop,
        crop_bbox
    ) = crop_bottom(
        rotated,
        refined
    )


    if crop is None:

        save_missing(
            image_path,
            original,
            missing_folder,
            missing_debug_folder,
            "BOTTOM MISSING - OUTSIDE"
        )

        return False


    # ========================================================
    # DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_bottom_debug(
        rotated,
        refined,
        crop_bbox
    )


    cv2.imwrite(
        str(
            debug_folder
            /
            (
                image_path.stem
                +
                "_debug.png"
            )
        ),
        debug
    )


    # ========================================================
    # 600 x 600
    # ========================================================

    final = resize_to_600(
        crop
    )


    final = lighten_if_dark(
        final
    )


    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        output_folder
        /
        image_path.name
    )


    success = cv2.imwrite(
        str(output_path),
        final
    )


    if not success:

        return False


    print(
        f"[BOTTOM OK] "
        f"{image_path.name} "
        f"| angle="
        f"{angle:.2f}° "
        f"| die="
        f"{refined['side']:.0f}px "
        f"| 600x600"
    )


    return True


# ============================================================
# PROCESS ONE FOLDER
# ============================================================

def process_folder(
    position,
    label
):

    input_folder = (
        INPUT_ROOT
        /
        position
        /
        label
    )


    output_folder = (
        OUTPUT_ROOT
        /
        position
        /
        label
    )


    debug_folder = (
        OUTPUT_ROOT
        /
        "_debug"
        /
        position
        /
        label
    )


    missing_folder = (
        OUTPUT_ROOT
        /
        "Missing"
        /
        position
        /
        label
    )


    missing_debug_folder = (
        OUTPUT_ROOT
        /
        "_debug_missing"
        /
        position
        /
        label
    )


    if not input_folder.exists():

        print(
            f"[MISSING INPUT] "
            f"{input_folder}"
        )

        return


    total = 0
    processed = 0
    missing = 0


    print()

    print(
        "=" * 80
    )


    print(
        f"PROCESSING: "
        f"{position} / {label}"
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


        # ====================================================
        # DIFFERENT METHOD FOR TOP / BOTTOM
        # ====================================================

        if position == "ViTop":


            success = process_top(

                image_path,

                output_folder,

                debug_folder,

                missing_folder,

                missing_debug_folder
            )


        else:


            success = process_bottom(

                image_path,

                output_folder,

                debug_folder,

                missing_folder,

                missing_debug_folder
            )


        if success:

            processed += 1


        else:

            missing += 1


    print()

    print(
        f"{position}/{label}: "
        f"{processed}/{total} processed "
        f"| Missing={missing}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 80
    )

    print(
        "COMBINED PREPROCESSING"
    )

    print()

    print(
        "ViTop:"
    )

    print(
        "  rotated central die detection"
    )

    print()

    print(
        "ViBottom:"
    )

    print(
        "  rough detection"
    )

    print(
        "  small rotation correction"
    )

    print(
        "  left/right/top/bottom recenter"
    )

    print()

    print(
        "FINAL SIZE = 600 x 600"
    )

    print(
        "NO ECC / NO REFERENCE"
    )

    print(
        "ORIGINAL DATA = READ ONLY"
    )

    print(
        "=" * 80
    )


    for (
        position,
        label
    ) in DATASETS:


        process_folder(
            position,
            label
        )


    print()

    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print()

    print(
        f"Processed:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    print(
        f"Debug:\n"
        f"{OUTPUT_ROOT / '_debug'}"
    )

    print()

    print(
        f"Missing:\n"
        f"{OUTPUT_ROOT / 'Missing'}"
    )

    print()

    print(
        "No original image was overwritten, "
        "moved or deleted."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()