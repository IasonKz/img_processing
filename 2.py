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
    r"C:\Users\laka\software\classification_laka_processed_final"
)

REFERENCE_TOP = Path(
    r"C:\Users\laka\software\references\reference_top.bmp"
)

REFERENCE_BOTTOM = Path(
    r"C:\Users\laka\software\references\reference_bottom.bmp"
)


DATASETS = [
    ("ViTop", "Good", REFERENCE_TOP),
    ("ViTop", "Bad", REFERENCE_TOP),
    ("ViBottom", "Good", REFERENCE_BOTTOM),
    ("ViBottom", "Bad", REFERENCE_BOTTOM),
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

# Temporary resolution used ONLY for finding alignment.
ALIGN_SIZE = 512


# Large images like your 2856x2848 ViTop:
# crop 1200x1200 -> downsample exactly x2 -> 600x600
LARGE_IMAGE_THRESHOLD = 2200
LARGE_SOURCE_CROP_SIZE = 1200


# ============================================================
# IMAGE HELPERS
# ============================================================

def gray8(img):
    """
    Make an 8-bit grayscale COPY.

    Used only for:
        detection
        debug
        alignment

    Original image is not modified.
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
        return g


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


def debug_bgr(img):

    g = gray8(img)

    return cv2.cvtColor(
        g,
        cv2.COLOR_GRAY2BGR
    )


# ============================================================
# CENTRAL TARGET DETECTION
# ============================================================

def detect_central_target(original):
    """
    Find CENTRAL approximately-square target.

    Important for ViTop:

    Side squares are HARD rejected.
    They are not simply given a lower score.
    """

    g = gray8(original)

    H, W = g.shape[:2]

    image_cx = W / 2.0
    image_cy = H / 2.0


    # ========================================================
    # SEARCH ONLY CENTRAL PART
    # ========================================================

    SEARCH_RATIO = 0.62

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
    # PREPROCESSING
    # ========================================================

    roi = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )


    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )


    enhanced = clahe.apply(
        roi
    )


    # ========================================================
    # METHOD 1:
    # BRIGHT AREAS
    # ========================================================

    _, bright = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    # ========================================================
    # METHOD 2:
    # EDGES
    #
    # useful for darker Top images
    # ========================================================

    median = float(
        np.median(enhanced)
    )


    low = int(
        max(
            10,
            0.55 * median
        )
    )


    high = int(
        min(
            255,
            max(
                low + 30,
                1.45 * median
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
    # FIND CANDIDATES
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
        + contours2
    )


    best = None
    best_score = -1e30


    # ========================================================
    # FILTER SETTINGS
    # ========================================================

    MAX_CENTER_OFFSET = 0.20

    MIN_SIZE_RATIO = 0.10
    MAX_SIZE_RATIO = 0.48

    EXPECTED_RATIO = 0.25


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


        (
            (cx_roi, cy_roi),
            (rw, rh),
            _
        ) = rect


        if rw < 5 or rh < 5:
            continue


        # Full-image coordinates
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


        # ====================================================
        # MUST LOOK LIKE A SQUARE
        # ====================================================

        aspect = (
            long_side
            / max(
                short_side,
                1e-6
            )
        )


        if aspect > 1.75:
            continue


        # ====================================================
        # SIZE
        # ====================================================

        size_ratio = 0.5 * (

            long_side
            / float(W)

            +

            short_side
            / float(H)
        )


        if not (
            MIN_SIZE_RATIO
            <= size_ratio
            <= MAX_SIZE_RATIO
        ):
            continue


        # ====================================================
        # DISTANCE FROM IMAGE CENTER
        # ====================================================

        dx = (
            abs(
                cx - image_cx
            )
            / float(W)
        )


        dy = (
            abs(
                cy - image_cy
            )
            / float(H)
        )


        # ====================================================
        # CRITICAL:
        # COMPLETELY REJECT SIDE TARGETS
        # ====================================================

        if dx > MAX_CENTER_OFFSET:
            continue


        if dy > MAX_CENTER_OFFSET:
            continue


        # ====================================================
        # CENTER SCORE
        # ====================================================

        center_dist = np.hypot(
            dx,
            dy
        )


        center_score = (

            1.0

            - center_dist
            / (
                np.sqrt(2.0)
                * MAX_CENTER_OFFSET
            )
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
        # EXPECTED SIZE SCORE
        # ====================================================

        size_score = (

            1.0

            - abs(
                size_ratio
                - EXPECTED_RATIO
            )
            / EXPECTED_RATIO

        )


        size_score = float(
            np.clip(
                size_score,
                0.0,
                1.0
            )
        )


        # ====================================================
        # FINAL SCORE
        #
        # CENTER DOMINATES
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

            best = (
                cx,
                cy
            )


    return best


# ============================================================
# SOURCE CROP SIZE
# ============================================================

def choose_source_crop_size(
    original
):

    H, W = original.shape[:2]


    # Large Top-like image
    if max(H, W) >= LARGE_IMAGE_THRESHOLD:

        return (
            LARGE_SOURCE_CROP_SIZE
        )


    # Smaller Bottom-like image
    return FINAL_SIZE


# ============================================================
# EXACT SOURCE CROP
# ============================================================

def crop_square(
    original,
    cx,
    cy,
    crop_size
):

    H, W = original.shape[:2]


    if (
        H < crop_size
        or W < crop_size
    ):

        raise ValueError(

            f"Image {W}x{H} "
            f"is smaller than crop "
            f"{crop_size}x{crop_size}"
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


    # Keep exact crop size
    x1 = max(
        0,
        min(
            x1,
            W - crop_size
        )
    )


    y1 = max(
        0,
        min(
            y1,
            H - crop_size
        )
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
# DEBUG
# ============================================================

def make_debug(
    original,
    bbox
):

    debug = debug_bgr(
        original
    )


    (
        x1,
        y1,
        x2,
        y2
    ) = bbox


    # ========================================================
    # THIS GREEN RECTANGLE
    # IS EXACTLY THE SOURCE AREA USED
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

        5
    )


    return debug


# ============================================================
# MAKE 600 x 600
# ============================================================

def to_final_size(
    source_crop
):

    H, W = source_crop.shape[:2]


    # Smaller images:
    # already 600x600
    if (
        H == FINAL_SIZE
        and W == FINAL_SIZE
    ):

        return (
            source_crop.copy()
        )


    # Large images:
    #
    # 1200x1200
    # ->
    # 600x600
    #
    # Downsample by exactly x2
    return cv2.resize(

        source_crop,

        (
            FINAL_SIZE,
            FINAL_SIZE
        ),

        interpolation=cv2.INTER_AREA
    )


# ============================================================
# REFERENCE PREPARATION
# ============================================================

def centered_square(
    img
):

    H, W = img.shape[:2]

    side = min(
        H,
        W
    )


    x1 = (
        W - side
    ) // 2


    y1 = (
        H - side
    ) // 2


    return img[
        y1:y1 + side,
        x1:x1 + side
    ].copy()


# ============================================================
# ALIGNMENT REPRESENTATION
# ============================================================

def ecc_representation(
    img
):

    g = gray8(
        img
    )


    # Temporary resize ONLY FOR
    # alignment calculation.
    g = cv2.resize(

        g,

        (
            ALIGN_SIZE,
            ALIGN_SIZE
        ),

        interpolation=cv2.INTER_AREA
    )


    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(
            8,
            8
        )
    )


    g = clahe.apply(
        g
    )


    g = cv2.GaussianBlur(
        g,
        (5, 5),
        0
    )


    # ========================================================
    # ADD EDGE INFORMATION
    # ========================================================

    gx = cv2.Sobel(

        g,

        cv2.CV_32F,

        1,
        0,

        ksize=3
    )


    gy = cv2.Sobel(

        g,

        cv2.CV_32F,

        0,
        1,

        ksize=3
    )


    magnitude = cv2.magnitude(
        gx,
        gy
    )


    max_mag = float(
        magnitude.max()
    )


    if max_mag > 0:

        magnitude /= (
            max_mag
        )


    intensity = (

        g.astype(
            np.float32
        )

        / 255.0
    )


    result = (

        0.70
        * intensity

        +

        0.30
        * magnitude

    )


    return result.astype(
        np.float32
    )


# ============================================================
# ALIGN TO REFERENCE
# ============================================================

def align_to_reference(
    crop,
    reference
):

    """
    Try:

        0 degrees
        90 degrees
        180 degrees
        270 degrees

    For every orientation:

        ECC fine alignment
        rotation + translation

    No scale change is allowed by ECC.
    """

    template = ecc_representation(
        reference
    )


    criteria = (

        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,

        150,

        1e-6
    )


    best = None


    # ========================================================
    # TRY FOUR ORIENTATIONS
    # ========================================================

    rotations = (
        0,
        90,
        180,
        270
    )


    for k, rotation_deg in enumerate(
        rotations
    ):

        # Rotate REAL crop
        moving_native = np.rot90(
            crop,
            k=k
        ).copy()


        # Temporary representation
        moving = ecc_representation(
            moving_native
        )


        warp = np.eye(
            2,
            3,
            dtype=np.float32
        )


        try:

            ecc, warp = cv2.findTransformECC(

                template,

                moving,

                warp,

                cv2.MOTION_EUCLIDEAN,

                criteria,

                None,

                5
            )


        except cv2.error:

            continue


        if (
            best is None
            or ecc > best["ecc"]
        ):

            best = {

                "ecc":
                    float(ecc),

                "warp":
                    warp.copy(),

                "image":
                    moving_native,

                "rotation":
                    rotation_deg
            }


    # ========================================================
    # ECC FAILED
    # ========================================================

    if best is None:

        return (
            crop.copy(),
            None,
            0
        )


    # ========================================================
    # APPLY WARP TO REAL 600x600 DATA
    # ========================================================

    warp = best[
        "warp"
    ].copy()


    # Translation was calculated
    # at 512x512.
    #
    # Convert it to 600x600.
    scale = (

        FINAL_SIZE
        / float(
            ALIGN_SIZE
        )
    )


    warp[
        0,
        2
    ] *= scale


    warp[
        1,
        2
    ] *= scale


    aligned = cv2.warpAffine(

        best["image"],

        warp,

        (
            FINAL_SIZE,
            FINAL_SIZE
        ),

        flags=(
            cv2.INTER_LINEAR
            |
            cv2.WARP_INVERSE_MAP
        ),

        borderMode=(
            cv2.BORDER_REFLECT_101
        )
    )


    return (

        aligned,

        best["ecc"],

        best["rotation"]
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    output_folder,
    debug_folder,
    reference
):

    # ========================================================
    # READ ORIGINAL
    #
    # READ ONLY
    # ========================================================

    original = cv2.imread(

        str(
            image_path
        ),

        cv2.IMREAD_UNCHANGED
    )


    if original is None:

        print(
            f"[ERROR READ] "
            f"{image_path.name}"
        )

        return False


    # ========================================================
    # FIND CENTRAL TARGET
    # ========================================================

    center = detect_central_target(
        original
    )


    if center is None:

        print(
            f"[REJECT] "
            f"central target not found: "
            f"{image_path.name}"
        )

        return False


    cx, cy = center


    # ========================================================
    # DECIDE SOURCE CROP
    # ========================================================

    source_crop_size = (
        choose_source_crop_size(
            original
        )
    )


    # ========================================================
    # EXACT CROP FROM ORIGINAL
    # ========================================================

    try:

        source_crop, bbox = crop_square(

            original,

            cx,
            cy,

            source_crop_size
        )


    except ValueError as error:

        print(
            f"[REJECT] "
            f"{image_path.name}: "
            f"{error}"
        )

        return False


    # ========================================================
    # NEW DEBUG
    # ========================================================

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
        str(
            debug_path
        ),
        debug
    )


    # ========================================================
    # MAKE CLASSIFIER INPUT
    # ========================================================

    crop = to_final_size(
        source_crop
    )


    # ========================================================
    # ALIGN WITH REFERENCE
    # ========================================================

    aligned, ecc, rotation = (
        align_to_reference(

            crop,

            reference
        )
    )


    # ========================================================
    # SAVE ONLY INTO NEW FOLDER
    # ========================================================

    output_path = (

        output_folder

        / image_path.name
    )


    success = cv2.imwrite(

        str(
            output_path
        ),

        aligned
    )


    if not success:

        print(
            f"[ERROR SAVE] "
            f"{output_path}"
        )

        return False


    # ========================================================
    # STATUS
    # ========================================================

    if ecc is None:

        print(

            f"[OK/WARN] "
            f"{image_path.name} "

            f"| ECC failed; "
            f"central crop saved"
        )


    else:

        print(

            f"[OK] "
            f"{image_path.name} "

            f"| source="
            f"{source_crop_size}x"
            f"{source_crop_size} "

            f"-> 600x600 "

            f"| ECC="
            f"{ecc:.4f} "

            f"| rot="
            f"{rotation}°"
        )


    return True


# ============================================================
# PROCESS ONE FOLDER
# ============================================================

def process_folder(
    input_folder,
    output_folder,
    reference
):

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # Every dataset gets its own debug folder
    debug_folder = (

        output_folder

        / "_debug"
    )


    debug_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    total = 0
    done = 0


    print()

    print(
        "=" * 75
    )

    print(
        f"INPUT : "
        f"{input_folder}"
    )

    print(
        f"OUTPUT: "
        f"{output_folder}"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # LOOP FILES
    # ========================================================

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


        if process_image(

            image_path,

            output_folder,

            debug_folder,

            reference
        ):

            done += 1


    print(
        f"Finished: "
        f"{done}/{total}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Starting preprocessing..."
    )


    print(
        "Original folders are READ ONLY."
    )


    print(
        f"Output folder:\n"
        f"{OUTPUT_ROOT}"
    )


    # ========================================================
    # CHECK REFERENCES
    # ========================================================

    if not REFERENCE_TOP.exists():

        raise FileNotFoundError(

            f"Missing TOP reference:\n"
            f"{REFERENCE_TOP}"
        )


    if not REFERENCE_BOTTOM.exists():

        raise FileNotFoundError(

            f"Missing BOTTOM reference:\n"
            f"{REFERENCE_BOTTOM}"
        )


    # ========================================================
    # LOAD REFERENCES
    # ========================================================

    ref_top = cv2.imread(

        str(
            REFERENCE_TOP
        ),

        cv2.IMREAD_UNCHANGED
    )


    ref_bottom = cv2.imread(

        str(
            REFERENCE_BOTTOM
        ),

        cv2.IMREAD_UNCHANGED
    )


    if ref_top is None:

        raise RuntimeError(

            f"Could not read:\n"
            f"{REFERENCE_TOP}"
        )


    if ref_bottom is None:

        raise RuntimeError(

            f"Could not read:\n"
            f"{REFERENCE_BOTTOM}"
        )


    # ========================================================
    # REFERENCES ARE ALREADY MANUALLY SELECTED
    #
    # Do NOT run target detection on them again.
    # ========================================================

    ref_top = centered_square(
        ref_top
    )


    ref_bottom = centered_square(
        ref_bottom
    )


    references = {

        "ViTop":
            ref_top,

        "ViBottom":
            ref_bottom
    }


    # ========================================================
    # PROCESS ALL FOUR DATASETS
    # ========================================================

    for (
        position,
        label,
        _
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


        if not input_folder.exists():

            print(

                f"[MISSING FOLDER] "
                f"{input_folder}"
            )

            continue


        if position == "ViTop":

            ref_name = (
                "reference_top.bmp"
            )


        else:

            ref_name = (
                "reference_bottom.bmp"
            )


        print()

        print(
            f"{position}/{label} "
            f"using {ref_name}"
        )


        process_folder(

            input_folder,

            output_folder,

            references[
                position
            ]
        )


    print()

    print(
        "=" * 75
    )

    print(
        "DONE"
    )

    print(
        "Original images were NEVER "
        "overwritten, moved or deleted."
    )

    print(
        f"Processed data:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()