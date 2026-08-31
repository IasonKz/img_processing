# img_processing



import cv2
import numpy as np
import shutil
from pathlib import Path
import math


# ============================================================
# SETTINGS
# ============================================================

CROP_SIZE = 600

# Alignment thresholds.
# Αυτά πιθανότατα θα χρειαστούν tuning στα πραγματικά δεδομένα.
MIN_MATCHES = 12
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.35

MAX_ROTATION = 25.0       # degrees
MIN_SCALE = 0.90
MAX_SCALE = 1.10

MIN_ECC_SCORE = 0.45

# Αν η εικόνα είναι κάτω από αυτό το brightness ratio
# σε σχέση με το reference, φωτίζουμε ΜΟΝΟ το preview.
DARK_RATIO = 0.75
MAX_PREVIEW_GAIN = 1.6


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def read_image(path):
    """
    Read image WITHOUT modifying the original file.
    IMREAD_UNCHANGED preserves 16-bit images.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    return img


def to_gray(img):
    if img.ndim == 2:
        return img

    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ValueError("Unsupported image format")


def to_float01(img):
    """
    Convert uint8 / uint16 image to float in range approximately 0...1.
    """
    img = img.astype(np.float32)

    if img.max() == 0:
        return img

    if img.max() > 255:
        return img / 65535.0

    return img / 255.0


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(img, reference):
    """
    Reject obviously useless images:
    - almost black
    - almost completely flat
    """

    gray = to_float01(to_gray(img))
    ref_gray = to_float01(to_gray(reference))

    p99 = np.percentile(gray, 99)
    ref_p99 = np.percentile(ref_gray, 99)

    dynamic = np.percentile(gray, 99) - np.percentile(gray, 1)
    ref_dynamic = (
        np.percentile(ref_gray, 99)
        - np.percentile(ref_gray, 1)
    )

    if p99 < 0.02 * max(ref_p99, 1e-6):
        return False, "almost_black"

    if dynamic < 0.02 * max(ref_dynamic, 1e-6):
        return False, "almost_flat"

    return True, "OK"


# ============================================================
# IMAGE FOR ALIGNMENT ONLY
# ============================================================

def registration_image(img):
    """
    Creates an 8-bit image used ONLY for ORB/ECC.

    The actual original/crop remains untouched in bit depth.
    """

    gray = to_float01(to_gray(img))

    low = np.percentile(gray, 1)
    high = np.percentile(gray, 99)

    if high - low < 1e-8:
        return np.zeros(gray.shape, dtype=np.uint8)

    normalized = (gray - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)

    result = (normalized * 255).astype(np.uint8)

    # Mild normalization so alignment is less sensitive to illumination.
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    result = clahe.apply(result)

    return cv2.GaussianBlur(result, (3, 3), 0)


# ============================================================
# STEP 1 — COARSE ALIGNMENT
# ORB + RANSAC
# ============================================================

def coarse_alignment(img, reference):
    """
    Rough alignment:
    ORB -> feature matching -> RANSAC

    Returns transformation INPUT -> REFERENCE.
    """

    img_reg = registration_image(img)
    ref_reg = registration_image(reference)

    orb = cv2.ORB_create(
        nfeatures=5000,
        fastThreshold=7
    )

    kp_img, des_img = orb.detectAndCompute(img_reg, None)
    kp_ref, des_ref = orb.detectAndCompute(ref_reg, None)

    if des_img is None or des_ref is None:
        raise RuntimeError("ORB could not find enough features")

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    matches = matcher.knnMatch(
        des_img,
        des_ref,
        k=2
    )

    good = []

    # Lowe ratio test
    for pair in matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCHES:
        raise RuntimeError(
            f"Not enough ORB matches: {len(good)}"
        )

    points_img = np.float32(
        [kp_img[m.queryIdx].pt for m in good]
    )

    points_ref = np.float32(
        [kp_ref[m.trainIdx].pt for m in good]
    )

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points_img,
        points_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=4000,
        confidence=0.995
    )

    if matrix is None:
        raise RuntimeError("RANSAC alignment failed")

    inliers = int(np.sum(inlier_mask))
    inlier_ratio = inliers / len(good)

    if inliers < MIN_INLIERS:
        raise RuntimeError(
            f"Too few RANSAC inliers: {inliers}"
        )

    if inlier_ratio < MIN_INLIER_RATIO:
        raise RuntimeError(
            f"Low RANSAC inlier ratio: {inlier_ratio:.2f}"
        )

    # ---------------------------------------------
    # Inspect scale + rotation
    # ---------------------------------------------

    a = matrix[0, 0]
    c = matrix[1, 0]

    scale = math.sqrt(a * a + c * c)

    rotation = math.degrees(
        math.atan2(c, a)
    )

    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise RuntimeError(
            f"Unreasonable scale: {scale:.3f}"
        )

    if abs(rotation) > MAX_ROTATION:
        raise RuntimeError(
            f"Unreasonable rotation: {rotation:.1f} deg"
        )

    # -------------------------------------------------
    # IMPORTANT:
    # remove scaling.
    #
    # We only want rotation + translation.
    # -------------------------------------------------

    theta = math.radians(rotation)

    rigid = np.array(
        [
            [
                math.cos(theta),
                -math.sin(theta),
                matrix[0, 2]
            ],
            [
                math.sin(theta),
                math.cos(theta),
                matrix[1, 2]
            ]
        ],
        dtype=np.float32
    )

    info = {
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "rotation": rotation,
        "estimated_scale": scale
    }

    return rigid, info


# ============================================================
# APPLY COARSE TRANSFORMATION
# ============================================================

def apply_affine(img, matrix, reference):
    """
    Transform image into reference coordinate system.
    """

    height, width = reference.shape[:2]

    aligned = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned


# ============================================================
# STEP 2 — FINE ALIGNMENT
# ECC
# ============================================================

def fine_alignment_ecc(coarse_img, reference):
    """
    Fine adjustment using ECC.

    Allows:
        rotation
        x translation
        y translation

    Does NOT allow arbitrary perspective distortion.
    """

    img_reg = registration_image(coarse_img)
    ref_reg = registration_image(reference)

    img_float = img_reg.astype(np.float32) / 255.0
    ref_float = ref_reg.astype(np.float32) / 255.0

    warp = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        150,
        1e-6
    )

    try:

        score, warp = cv2.findTransformECC(
            ref_float,
            img_float,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

    except cv2.error:

        raise RuntimeError(
            "ECC failed to converge"
        )

    if score < MIN_ECC_SCORE:

        raise RuntimeError(
            f"ECC score too low: {score:.3f}"
        )

    height, width = reference.shape[:2]

    aligned = cv2.warpAffine(
        coarse_img,
        warp,
        (width, height),

        # Important for findTransformECC
        flags=cv2.INTER_LINEAR
        | cv2.WARP_INVERSE_MAP,

        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned, score


# ============================================================
# FIXED 600x600 CROP
# ============================================================

def fixed_crop(img, center_x, center_y, size=600):

    half = size // 2

    x1 = int(center_x - half)
    y1 = int(center_y - half)

    x2 = x1 + size
    y2 = y1 + size

    height, width = img.shape[:2]

    if (
        x1 < 0
        or y1 < 0
        or x2 > width
        or y2 > height
    ):
        raise RuntimeError(
            "600x600 crop is outside image"
        )

    crop = img[y1:y2, x1:x2].copy()

    if crop.shape[:2] != (size, size):
        raise RuntimeError(
            "Crop has incorrect size"
        )

    return crop


# ============================================================
# HUMAN PREVIEW
# Gentle brightness enhancement
# ============================================================

def create_preview(crop, reference_crop):
    """
    The crop itself is NOT changed.

    This creates a separate 8-bit image only for
    easier human inspection.
    """

    img = to_float01(crop)
    ref = to_float01(reference_crop)

    img_gray = to_float01(to_gray(crop))
    ref_gray = to_float01(to_gray(reference_crop))

    img_brightness = np.percentile(
        img_gray,
        90
    )

    ref_brightness = np.percentile(
        ref_gray,
        90
    )

    ratio = (
        img_brightness
        / max(ref_brightness, 1e-6)
    )

    gain = 1.0

    if ratio < DARK_RATIO:

        gain = min(
            MAX_PREVIEW_GAIN,
            0.9 / max(ratio, 1e-6)
        )

    preview = img * gain

    # Very mild gamma correction only for dark images
    if gain > 1.0:
        preview = np.power(
            np.clip(preview, 0, 1),
            0.90
        )

    preview = np.clip(
        preview,
        0,
        1
    )

    preview = (
        preview * 255
    ).astype(np.uint8)

    return preview, gain


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    reference_path,
    output_folder,
    view="bottom"
):

    image_path = Path(image_path)
    reference_path = Path(reference_path)
    output_folder = Path(output_folder)

    # -------------------------------------------------
    # Separate folders.
    # ORIGINAL IS NEVER USED AS OUTPUT.
    # -------------------------------------------------

    crop_folder = (
        output_folder
        / view
        / "raw_crop"
    )

    preview_folder = (
        output_folder
        / view
        / "preview"
    )

    rejected_folder = (
        output_folder
        / view
        / "rejected"
    )

    crop_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    preview_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    rejected_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read original + reference.
    # No writing happens here.
    img = read_image(image_path)
    reference = read_image(reference_path)

    # ---------------------------------------------
    # Quality check
    # ---------------------------------------------

    valid, reason = check_image_quality(
        img,
        reference
    )

    if not valid:

        destination = (
            rejected_folder
            / image_path.name
        )

        # COPY, never move!
        shutil.copy2(
            image_path,
            destination
        )

        print(
            f"REJECTED {image_path.name}: {reason}"
        )

        return False

    try:

        # =============================================
        # 1. COARSE ALIGNMENT
        # =============================================

        coarse_matrix, coarse_info = (
            coarse_alignment(
                img,
                reference
            )
        )

        coarse = apply_affine(
            img,
            coarse_matrix,
            reference
        )

        # =============================================
        # 2. FINE ECC ALIGNMENT
        # =============================================

        aligned, ecc_score = (
            fine_alignment_ecc(
                coarse,
                reference
            )
        )

        # =============================================
        # 3. FIXED CROP
        # =============================================

        height, width = reference.shape[:2]

        # For now use centre of reference.
        #
        # Later you can replace these with manually
        # selected TOP/BOTTOM crop centres.
        center_x = width / 2
        center_y = height / 2

        crop = fixed_crop(
            aligned,
            center_x,
            center_y,
            CROP_SIZE
        )

        reference_crop = fixed_crop(
            reference,
            center_x,
            center_y,
            CROP_SIZE
        )

        # =============================================
        # 4. HUMAN PREVIEW
        # =============================================

        preview, gain = create_preview(
            crop,
            reference_crop
        )

        # =============================================
        # 5. SAVE NEW FILES
        # =============================================

        # PNG preserves uint16.
        #
        # IMPORTANT:
        # this is a NEW file.
        # Original BMP remains untouched.
        crop_path = (
            crop_folder
            / f"{image_path.stem}.png"
        )

        preview_path = (
            preview_folder
            / f"{image_path.stem}.png"
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )

        print(
            f"OK: {image_path.name}"
        )

        print(
            f"    rotation = "
            f"{coarse_info['rotation']:.2f} deg"
        )

        print(
            f"    matches = "
            f"{coarse_info['matches']}"
        )

        print(
            f"    inliers = "
            f"{coarse_info['inliers']}"
        )

        print(
            f"    ECC = "
            f"{ecc_score:.3f}"
        )

        print(
            f"    preview brightness gain = "
            f"{gain:.2f}"
        )

        return True

    except Exception as error:

        # =============================================
        # REJECT FAILED IMAGE
        # =============================================

        destination = (
            rejected_folder
            / image_path.name
        )

        # Again: COPY ONLY.
        shutil.copy2(
            image_path,
            destination
        )

        print(
            f"REJECTED {image_path.name}: "
            f"{error}"
        )

        return False


# ============================================================
# PROCESS ENTIRE FOLDER
# ============================================================

def process_folder(
    input_folder,
    reference_path,
    output_folder,
    view="bottom"
):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # =============================================
    # SAFETY CHECK
    # =============================================

    if (
        input_folder.resolve()
        == output_folder.resolve()
    ):
        raise RuntimeError(
            "Input and output folders "
            "must be different!"
        )

    extensions = {
        ".bmp",
        ".png",
        ".tif",
        ".tiff"
    }

    images = [
        path
        for path in input_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        )
    ]

    print(
        f"\nProcessing {len(images)} "
        f"{view.upper()} images\n"
    )

    successful = 0
    rejected = 0

    for image_path in images:

        result = process_image(
            image_path=image_path,
            reference_path=reference_path,
            output_folder=output_folder,
            view=view
        )

        if result:
            successful += 1
        else:
            rejected += 1

    print("\n==========================")
    print("FINISHED")
    print("==========================")

    print(
        f"Successful: {successful}"
    )

    print(
        f"Rejected:   {rejected}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CHANGE THESE PATHS
    # ========================================================

    VIEW = "bottom"
    # VIEW = "top"

    INPUT_FOLDER = r"C:\path\to\bottom_images"

    REFERENCE_IMAGE = (
        r"C:\path\to\references"
        r"\bottom_reference.bmp"
    )

    OUTPUT_FOLDER = (
        r"C:\path\to\processed"
    )

    process_folder(
        input_folder=INPUT_FOLDER,
        reference_path=REFERENCE_IMAGE,
        output_folder=OUTPUT_FOLDER,
        view=VIEW
    )