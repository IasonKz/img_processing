import cv2
import numpy as np
from pathlib import Path
import math


# ============================================================
# SETTINGS
# ============================================================

CROP_WIDTH = 600
CROP_HEIGHT = 600

# Το τετραγωνάκι ενδιαφέροντος είναι περίπου 1/3
# της μικρότερης διάστασης της raw εικόνας.
EXPECTED_SQUARE_FRACTION = 1.0 / 3.0

# Επιτρεπτό εύρος μεγέθους για detection
MIN_SQUARE_FRACTION = 0.18
MAX_SQUARE_FRACTION = 0.55


# ------------------------------------------------------------
# OFFSETS
#
# Άλλαξέ τα εδώ αν χρειάζεται.
# +X = δεξιά
# -X = αριστερά
# +Y = κάτω
# -Y = πάνω
# ------------------------------------------------------------

TOP_OFFSET_X = 0
TOP_OFFSET_Y = -15

BOTTOM_OFFSET_X = 0
BOTTOM_OFFSET_Y = 0


# ============================================================
# CONVERT ONLY FOR DETECTION
#
# Η πραγματική εικόνα ΔΕΝ μετατρέπεται σε 8-bit.
# Αυτό χρησιμοποιείται μόνο για computer vision.
# ============================================================

def make_detection_image(image):

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_float = gray.astype(np.float32)

    # Robust normalization
    low = np.percentile(gray_float, 1)
    high = np.percentile(gray_float, 99)

    if high <= low:
        return np.zeros_like(gray, dtype=np.uint8)

    gray_float = (gray_float - low) / (high - low)
    gray_float = np.clip(gray_float, 0, 1)

    gray8 = (gray_float * 255).astype(np.uint8)

    # Slight local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray8 = clahe.apply(gray8)

    return gray8


# ============================================================
# ORDER BOX POINTS
# ============================================================

def order_points(points):

    points = np.asarray(points, dtype=np.float32)

    result = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()

    result[0] = points[np.argmin(sums)]   # top-left
    result[2] = points[np.argmax(sums)]   # bottom-right

    result[1] = points[np.argmin(diffs)]  # top-right
    result[3] = points[np.argmax(diffs)]  # bottom-left

    return result


# ============================================================
# FIND CENTRAL SQUARE
# ============================================================

def find_central_square(image):

    gray = make_detection_image(image)

    h, w = gray.shape

    min_dim = min(h, w)

    # --------------------------------------------------------
    # Create several simple representations.
    #
    # We don't depend on only one thresholding technique.
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edges
    edges = cv2.Canny(
        blurred,
        40,
        120
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Bright-object threshold
    _, threshold1 = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Dark-object threshold
    _, threshold2 = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    masks = [
        edges,
        threshold1,
        threshold2
    ]

    candidates = []

    image_center = np.array(
        [w / 2, h / 2],
        dtype=np.float32
    )

    # ========================================================
    # SEARCH CANDIDATES
    # ========================================================

    for mask in masks:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area <= 0:
                continue

            rect = cv2.minAreaRect(contour)

            (cx, cy), (rw, rh), rect_angle = rect

            if rw <= 0 or rh <= 0:
                continue

            short_side = min(rw, rh)
            long_side = max(rw, rh)

            average_side = (
                rw + rh
            ) / 2

            side_fraction = (
                average_side / min_dim
            )

            # ------------------------------------------------
            # Expected size
            # ------------------------------------------------

            if not (
                MIN_SQUARE_FRACTION
                <= side_fraction
                <= MAX_SQUARE_FRACTION
            ):
                continue

            # ------------------------------------------------
            # Must be approximately square
            # ------------------------------------------------

            aspect_ratio = (
                long_side / short_side
            )

            if aspect_ratio > 1.4:
                continue

            # ------------------------------------------------
            # Must be reasonably close to center
            # ------------------------------------------------

            candidate_center = np.array(
                [cx, cy],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                candidate_center -
                image_center
            )

            normalized_distance = (
                distance / min_dim
            )

            if normalized_distance > 0.35:
                continue

            # ------------------------------------------------
            # SCORING
            # ------------------------------------------------

            # 1 = perfectly centered
            center_score = max(
                0,
                1 -
                normalized_distance / 0.35
            )

            # 1 = perfect square
            square_score = (
                short_side /
                long_side
            )

            # 1 = close to expected 1/3 size
            size_difference = abs(
                side_fraction -
                EXPECTED_SQUARE_FRACTION
            )

            size_score = max(
                0,
                1 -
                size_difference / 0.25
            )

            rect_area = rw * rh

            fill_ratio = min(
                area / rect_area,
                1.0
            )

            # Center is the most important thing
            score = (
                0.45 * center_score +
                0.25 * square_score +
                0.20 * size_score +
                0.10 * fill_ratio
            )

            box = cv2.boxPoints(rect)

            candidates.append(
                {
                    "score": score,
                    "center": (cx, cy),
                    "box": box,
                    "size": (rw, rh)
                }
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    if len(candidates) == 0:

        print(
            "WARNING: central square not detected."
            " Using image center."
        )

        return {
            "found": False,
            "center": (
                w / 2,
                h / 2
            ),
            "angle": 0.0,
            "box": None,
            "score": 0
        }

    # Best candidate
    best = max(
        candidates,
        key=lambda x: x["score"]
    )

    ordered = order_points(
        best["box"]
    )

    top_left = ordered[0]
    top_right = ordered[1]

    dx = (
        top_right[0] -
        top_left[0]
    )

    dy = (
        top_right[1] -
        top_left[1]
    )

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    # Square orientation only matters modulo 90°
    while angle > 45:
        angle -= 90

    while angle < -45:
        angle += 90

    return {
        "found": True,
        "center": best["center"],
        "angle": angle,
        "box": best["box"],
        "score": best["score"]
    }


# ============================================================
# ROTATE AROUND ROI CENTER
# ============================================================

def straighten_image(
    image,
    center,
    angle
):

    h, w = image.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        -angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return rotated


# ============================================================
# EXACT 600 x 600 CROP
# ============================================================

def crop_around_center(
    image,
    center,
    offset_x=0,
    offset_y=0
):

    cx, cy = center

    cx += offset_x
    cy += offset_y

    x1 = int(
        round(
            cx -
            CROP_WIDTH / 2
        )
    )

    y1 = int(
        round(
            cy -
            CROP_HEIGHT / 2
        )
    )

    x2 = x1 + CROP_WIDTH
    y2 = y1 + CROP_HEIGHT

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Padding if ever necessary.
    # This guarantees EXACTLY 600x600.
    # --------------------------------------------------------

    pad_left = max(
        0,
        -x1
    )

    pad_top = max(
        0,
        -y1
    )

    pad_right = max(
        0,
        x2 - w
    )

    pad_bottom = max(
        0,
        y2 - h
    )

    if (
        pad_left > 0 or
        pad_right > 0 or
        pad_top > 0 or
        pad_bottom > 0
    ):

        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101
        )

        x1 += pad_left
        x2 += pad_left

        y1 += pad_top
        y2 += pad_top

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    return crop


# ============================================================
# ISOLATE CENTRAL ROI
# ============================================================

def isolate_roi(
    image,
    offset_x=0,
    offset_y=0
):

    detection = find_central_square(
        image
    )

    center = detection["center"]
    angle = detection["angle"]

    print(
        f"ROI center = "
        f"({center[0]:.1f}, {center[1]:.1f})"
    )

    print(
        f"rough angle = {angle:.2f} degrees"
    )

    print(
        f"detection score = "
        f"{detection['score']:.3f}"
    )

    # --------------------------------------------------------
    # Rough straighten
    # --------------------------------------------------------

    straight = straighten_image(
        image,
        center,
        angle
    )

    # --------------------------------------------------------
    # Fixed crop around detected center
    # --------------------------------------------------------

    crop = crop_around_center(
        straight,
        center,
        offset_x,
        offset_y
    )

    return crop, detection


# ============================================================
# ECC FINE ALIGNMENT WITH REFERENCE
# ============================================================

def prepare_for_ecc(image):

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    image = image.astype(
        np.float32
    )

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    return image


def align_to_reference(
    image,
    reference
):

    target = prepare_for_ecc(
        reference
    )

    source = prepare_for_ecc(
        image
    )

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        150,
        1e-6
    )

    try:

        correlation, warp_matrix = (
            cv2.findTransformECC(
                target,
                source,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria
            )
        )

        h, w = reference.shape[:2]

        aligned = cv2.warpAffine(
            image,
            warp_matrix,
            (w, h),
            flags=(
                cv2.INTER_LINEAR |
                cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REFLECT_101
        )

        print(
            f"ECC correlation = "
            f"{correlation:.5f}"
        )

        return aligned

    except cv2.error as error:

        # IMPORTANT:
        # Do NOT reject the image.
        print(
            "WARNING: ECC failed."
        )

        print(
            "Keeping rough-aligned crop."
        )

        return image.copy()


# ============================================================
# DEBUG IMAGE
# ============================================================

def create_debug_image(
    image,
    detection
):

    preview_gray = make_detection_image(
        image
    )

    preview = cv2.cvtColor(
        preview_gray,
        cv2.COLOR_GRAY2BGR
    )

    center = detection["center"]

    cv2.circle(
        preview,
        (
            int(center[0]),
            int(center[1])
        ),
        12,
        (0, 0, 255),
        3
    )

    if detection["box"] is not None:

        box = detection["box"].astype(
            np.int32
        )

        cv2.polylines(
            preview,
            [box],
            True,
            (0, 255, 0),
            4
        )

    return preview


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    input_path,
    output_path,
    image_type,
    reference_path=None,
    debug_path=None
):

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        print(
            f"Cannot read: {input_path}"
        )

        return

    # --------------------------------------------------------
    # Choose offset
    # --------------------------------------------------------

    if image_type.lower() == "top":

        offset_x = TOP_OFFSET_X
        offset_y = TOP_OFFSET_Y

    elif image_type.lower() == "bottom":

        offset_x = BOTTOM_OFFSET_X
        offset_y = BOTTOM_OFFSET_Y

    else:

        raise ValueError(
            "image_type must be top or bottom"
        )

    # --------------------------------------------------------
    # Detect + rough straighten + crop
    # --------------------------------------------------------

    crop, detection = isolate_roi(
        image,
        offset_x,
        offset_y
    )

    # --------------------------------------------------------
    # Optional fine alignment
    # --------------------------------------------------------

    final_image = crop

    if reference_path is not None:

        reference = cv2.imread(
            str(reference_path),
            cv2.IMREAD_UNCHANGED
        )

        if reference is None:

            print(
                "Reference not found."
            )

            print(
                "Saving isolated crop only."
            )

        else:

            final_image = align_to_reference(
                crop,
                reference
            )

    # --------------------------------------------------------
    # Save processed COPY
    # --------------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        final_image
    )

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    if debug_path is not None:

        debug = create_debug_image(
            image,
            detection
        )

        debug_path = Path(
            debug_path
        )

        debug_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(debug_path),
            debug
        )

    print(
        f"Saved -> {output_path}"
    )

    print(
        f"Final shape = "
        f"{final_image.shape}"
    )


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    process_image(

        input_path=
        r"C:\data\raw_top.bmp",

        output_path=
        r"C:\data\processed\top_crop.bmp",

        image_type="top",

        # FIRST RUN:
        # No reference yet.
        reference_path=None,

        debug_path=
        r"C:\data\debug\top_detection.bmp"
    )