from pathlib import Path
import shutil

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\laka\software\classification_laka\ViBottom"
)

OUTPUT_ROOT = Path(
    r"C:\Users\laka\software\ViBottom_crop_test_v3"
)

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
# SETTINGS
# ============================================================

FINAL_SIZE = 600


# Small offset around actual die
OFFSET_RATIO = 0.03


# Bottom can genuinely be displaced,
# so don't force it too close to image center.
MAX_CENTER_OFFSET_X = 0.34
MAX_CENTER_OFFSET_Y = 0.34


MIN_SIDE_RATIO = 0.14
MAX_SIDE_RATIO = 0.52


MAX_ROTATION_DEG = 18.0

MIN_DETECTION_SCORE = 3.0


# How far around the approximate edge we search
EDGE_SEARCH_RATIO = 0.28


# ============================================================
# 8-BIT COPY FOR DETECTION
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


    x = img.astype(np.float32)

    if x.ndim == 3:
        x = np.mean(x, axis=2)


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
# ORDER RECTANGLE POINTS
# ============================================================

def order_points(points):

    pts = np.asarray(
        points,
        dtype=np.float32
    )


    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )


    sums = pts.sum(axis=1)

    diffs = np.diff(
        pts,
        axis=1
    ).reshape(-1)


    result[0] = pts[
        np.argmin(sums)
    ]  # top-left


    result[1] = pts[
        np.argmin(diffs)
    ]  # top-right


    result[2] = pts[
        np.argmax(sums)
    ]  # bottom-right


    result[3] = pts[
        np.argmax(diffs)
    ]  # bottom-left


    return result


# ============================================================
# ANGLE
# ============================================================

def rectangle_angle(box):

    box = order_points(box)

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


    # Square orientation repeats every 90 degrees
    while angle >= 45:
        angle -= 90

    while angle < -45:
        angle += 90


    return float(angle)


# ============================================================
# CONTRAST OF CANDIDATE DIE
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
    # Inside die
    # --------------------------------------------------------

    inner_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        inner_mask,
        np.round(box).astype(np.int32),
        255
    )


    # --------------------------------------------------------
    # Slightly larger surrounding area
    # --------------------------------------------------------

    outer_box = (
        center
        +
        1.28
        * (box - center)
    )


    outer_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )


    cv2.fillConvexPoly(
        outer_mask,
        np.round(
            outer_box
        ).astype(np.int32),
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

        float(np.mean(inside))
        -
        float(np.mean(outside))

    ) / 255.0


# ============================================================
# FIRST / ROUGH DIE DETECTION
# ============================================================

def detect_bottom_die(original):
    """
    First detection is intentionally only approximate.

    We later RE-CENTER using the four actual outer edges.
    """

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
    #
    # Remove spots, labels, texture etc.
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
    # TRY MULTIPLE THRESHOLDS
    # ========================================================

    otsu_value, _ = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    thresholds = [

        float(otsu_value),

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
        int(D * 0.025)
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


            # ------------------------------------------------
            # SIZE
            # ------------------------------------------------

            ratio = (
                long_side / D
            )


            if not (
                MIN_SIDE_RATIO
                <= ratio
                <= MAX_SIDE_RATIO
            ):
                continue


            # ------------------------------------------------
            # SQUARE-LIKE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Center filter - intentionally loose
            # ------------------------------------------------

            dx = abs(
                cx - image_center[0]
            ) / W


            dy = abs(
                cy - image_center[1]
            ) / H


            if (
                dx > MAX_CENTER_OFFSET_X
                or
                dy > MAX_CENTER_OFFSET_Y
            ):
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
                rw * rh
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
                1.0 / aspect
            )


            expected_ratio = 0.28


            size_score = float(
                np.clip(

                    1.0
                    -
                    abs(
                        ratio
                        - expected_ratio
                    )
                    / expected_ratio,

                    0.0,
                    1.0
                )
            )


            # Center gets LITTLE weight now.
            # We don't want to force a genuinely shifted die.
            score = (

                3.5 * contrast

                +

                2.5 * square_score

                +

                2.0 * fill

                +

                1.0 * size_score
            )


            candidates.append(
                {
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
                        box,

                    "score":
                        float(score),
                }
            )


    if not candidates:
        return None


    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return candidates[0]


# ============================================================
# ROTATE IMAGE AROUND APPROXIMATE DIE CENTER
# ============================================================

def rotate_image(
    original,
    center,
    angle
):

    H, W = original.shape[:2]


    matrix = cv2.getRotationMatrix2D(
        (
            float(center[0]),
            float(center[1])
        ),
        angle,
        1.0
    )


    rotated = cv2.warpAffine(
        original,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


    return (
        rotated,
        matrix
    )


# ============================================================
# FIND ONE OUTER EDGE
# ============================================================

def find_profile_edge(
    profile,
    expected_position,
    search_radius,
    positive
):
    """
    positive=True:
        dark -> bright edge

    positive=False:
        bright -> dark edge
    """

    profile = profile.astype(
        np.float32
    )


    # ========================================================
    # Smooth profile
    # ========================================================

    size = max(
        9,
        int(
            len(profile) * 0.01
        )
    )


    if size % 2 == 0:
        size += 1


    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (size, 1),
        0
    ).reshape(-1)


    gradient = np.gradient(
        smooth
    )


    low = max(
        1,
        int(
            expected_position
            - search_radius
        )
    )


    high = min(
        len(profile) - 2,
        int(
            expected_position
            + search_radius
        )
    )


    if high <= low:
        return None


    local = gradient[
        low:high + 1
    ]


    if positive:

        index = (
            low
            +
            int(
                np.argmax(local)
            )
        )

    else:

        index = (
            low
            +
            int(
                np.argmin(local)
            )
        )


    return index


# ============================================================
# RE-CENTER FROM ACTUAL 4 OUTER EDGES
# ============================================================

def refine_die_edges(
    rotated,
    approx_center,
    approx_side
):
    """
    THIS IS THE IMPORTANT NEW PART.

    After straightening, detect:

        LEFT edge
        RIGHT edge
        TOP edge
        BOTTOM edge

    Then:

        center_x = (left + right)/2
        center_y = (top + bottom)/2

    So illumination on one side can no longer
    simply drag the crop center to the right.
    """

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
    # Strong blur removes inner spots
    # ========================================================

    sigma = max(
        8.0,
        approx_side * 0.035
    )


    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )


    # ========================================================
    # CENTRAL STRIPS
    #
    # We DON'T use whole image because the side halo
    # was what caused the original bias.
    # ========================================================

    band_half = int(
        approx_side * 0.30
    )


    y1 = max(
        0,
        int(cy - band_half)
    )

    y2 = min(
        H,
        int(cy + band_half)
    )


    x1 = max(
        0,
        int(cx - band_half)
    )

    x2 = min(
        W,
        int(cx + band_half)
    )


    if (
        y2 <= y1
        or
        x2 <= x1
    ):

        return None


    # ========================================================
    # Horizontal brightness profile
    # ========================================================

    x_profile = np.mean(
        smooth[
            y1:y2,
            :
        ],
        axis=0
    )


    # ========================================================
    # Vertical brightness profile
    # ========================================================

    y_profile = np.mean(
        smooth[
            :,
            x1:x2
        ],
        axis=1
    )


    # ========================================================
    # Expected positions
    # ========================================================

    expected_left = (
        cx - approx_side / 2
    )

    expected_right = (
        cx + approx_side / 2
    )

    expected_top = (
        cy - approx_side / 2
    )

    expected_bottom = (
        cy + approx_side / 2
    )


    search_radius = (
        approx_side
        * EDGE_SEARCH_RATIO
    )


    # ========================================================
    # FIND FOUR EDGES
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
        right - left
    )


    span_y = (
        bottom - top
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
        left + right
    ) / 2.0


    refined_cy = (
        top + bottom
    ) / 2.0


    # Prefer more rather than less
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
            float(die_side),

        "edges":
            (
                int(left),
                int(top),
                int(right),
                int(bottom)
            ),
    }


# ============================================================
# TIGHT CENTERED CROP
# ============================================================

def crop_refined_die(
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

            * (
                1.0
                +
                2.0 * OFFSET_RATIO
            )
        )
    )


    x1 = int(
        round(
            cx - side / 2
        )
    )


    y1 = int(
        round(
            cy - side / 2
        )
    )


    x2 = (
        x1 + side
    )

    y2 = (
        y1 + side
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

        return None, None


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
# EXACT 600x600
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
# MILD LIGHTING
# ============================================================

def lighten_if_dark(
    image
):

    if not np.issubdtype(
        image.dtype,
        np.integer
    ):

        return image.copy()


    if image.ndim == 3:

        values = image.astype(
            np.float32
        ).mean(axis=2)

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


    threshold = 0.22


    if brightness >= threshold:

        return image.copy()


    darkness = float(
        np.clip(

            (
                threshold
                -
                brightness
            )
            /
            threshold,

            0.0,
            1.0
        )
    )


    # Maximum only ~10%
    factor = (
        1.0
        +
        0.10 * darkness
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
# DEBUG
# ============================================================

def make_debug(
    rotated,
    refined,
    crop_bbox
):

    gray = gray8(
        rotated
    )


    debug = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


    left, top, right, bottom = (
        refined["edges"]
    )


    # --------------------------------------------------------
    # Yellow = actual detected die
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Green = EXACT FINAL CROP
    # --------------------------------------------------------

    (
        x1,
        y1,
        x2,
        y2
    ) = crop_bbox


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


    # --------------------------------------------------------
    # Red = refined TRUE CENTER
    # --------------------------------------------------------

    center = tuple(
        np.round(
            refined["center"]
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
# PROCESS IMAGE
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
    # 1. ROUGH DETECTION
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
            f"| no die"
        )

        return False


    if (
        detection["score"]
        <
        MIN_DETECTION_SCORE
    ):

        save_missing(
            image_path,
            missing_folder
        )

        return False


    # ========================================================
    # 2. SMALL ROTATION
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
            f"| angle="
            f"{angle:.2f}"
        )

        return False


    rotated, matrix = rotate_image(

        original,

        detection["center"],

        angle
    )


    # Because we rotate around die center,
    # its approximate coordinates remain the same.
    approx_center = (
        detection["center"]
    )


    approx_side = max(

        detection["width"],

        detection["height"]
    )


    # ========================================================
    # 3. RE-CENTER FROM LEFT/RIGHT/TOP/BOTTOM
    # ========================================================

    refined = refine_die_edges(

        rotated,

        approx_center,

        approx_side
    )


    if refined is None:

        save_missing(
            image_path,
            missing_folder
        )


        print(
            f"[MISSING] "
            f"{image_path.name} "
            f"| edge refinement failed"
        )

        return False


    # ========================================================
    # 4. FINAL CROP
    # ========================================================

    crop, crop_bbox = (
        crop_refined_die(
            rotated,
            refined
        )
    )


    if crop is None:

        save_missing(
            image_path,
            missing_folder
        )

        return False


    # ========================================================
    # 5. EXACT 600x600
    # ========================================================

    final = resize_to_600(
        crop
    )


    final = lighten_if_dark(
        final
    )


    # ========================================================
    # SAVE FINAL
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


    # ========================================================
    # DEBUG
    # ========================================================

    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    debug = make_debug(

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
                + "_debug.png"
            )
        ),
        debug
    )


    print(

        f"[OK] "
        f"{image_path.name} "

        f"| angle="
        f"{angle:.2f}° "

        f"| die="
        f"{refined['side']:.0f}px "

        f"| final=600x600"
    )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "ViBottom RECENTER TEST"
    )

    print(
        "Final images = 600x600"
    )

    print(
        "Original files remain untouched"
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

    print(
        "DONE"
    )


if __name__ == "__main__":
    main()