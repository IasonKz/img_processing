import cv2
import numpy as np


MAX_TRANSLATION = 8


# ============================================================
# ROBUST GLOBAL LIGHTING MATCH
# ============================================================

def match_global_lighting(reference, moving):
    """
    Correct small global brightness/contrast differences.

    Does NOT perform local histogram equalization,
    so local defects remain visible.
    """

    ref = reference.astype(np.float32)
    mov = moving.astype(np.float32)

    # Avoid extreme pixels
    ref_low, ref_high = np.percentile(ref, [10, 90])
    mov_low, mov_high = np.percentile(mov, [10, 90])

    if mov_high - mov_low < 1e-6:
        return moving.copy()

    # Match robust contrast
    gain = (
        (ref_high - ref_low)
        /
        (mov_high - mov_low)
    )

    # Don't allow huge correction
    gain = np.clip(
        gain,
        0.8,
        1.25
    )

    # Match robust brightness
    ref_med = np.median(ref)
    mov_med = np.median(mov)

    offset = (
        ref_med
        - gain * mov_med
    )

    corrected = (
        gain * mov
        + offset
    )

    corrected = np.clip(
        corrected,
        0.0,
        1.0
    )

    return corrected.astype(np.float32)


# ============================================================
# GRADIENT IMAGE
# ============================================================

def gradient_image(img):
    """
    Use structure instead of raw brightness
    for estimating tiny translation.
    """

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    grad = cv2.magnitude(
        gx,
        gy
    )

    return grad


# ============================================================
# SMALL TRANSLATION ONLY
# ============================================================

def align_translation(reference, moving):

    ref_grad = gradient_image(reference)
    mov_grad = gradient_image(moving)

    try:
        shift, response = cv2.phaseCorrelate(
            ref_grad,
            mov_grad
        )

    except cv2.error:
        return moving.copy(), 0.0, 0.0, 0.0


    dx = float(shift[0])
    dy = float(shift[1])


    # Never allow a large correction
    if (
        abs(dx) > MAX_TRANSLATION
        or
        abs(dy) > MAX_TRANSLATION
    ):
        return moving.copy(), 0.0, 0.0, float(response)


    H, W = moving.shape


    M = np.array(
        [
            [1.0, 0.0, -dx],
            [0.0, 1.0, -dy],
        ],
        dtype=np.float32
    )


    aligned = cv2.warpAffine(
        moving,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


    return (
        aligned,
        dx,
        dy,
        float(response)
    )


# ============================================================
# ROBUST MATCH SCORE
# ============================================================

def robust_match_score(reference, moving):
    """
    Lower = better.

    IMPORTANT:
    Ignore the highest residual pixels when choosing rotation.

    Why?
    Because real damage may exist there and we don't want
    damage itself to make us choose the wrong orientation.
    """

    diff = np.abs(
        reference - moving
    )


    # Ignore borders
    BORDER = 20

    diff = diff[
        BORDER:-BORDER,
        BORDER:-BORDER
    ]


    values = diff.reshape(-1)


    # Keep only lowest 85% differences.
    #
    # Large local changes / damage do not dominate
    # orientation selection.
    cutoff = np.percentile(
        values,
        85
    )


    normal_pixels = values[
        values <= cutoff
    ]


    if len(normal_pixels) == 0:
        return np.inf


    return float(
        np.mean(normal_pixels)
    )


# ============================================================
# TEST 0 / 90 / 180 / 270
# ============================================================

def find_best_post_orientation(
    start,
    post
):
    """
    START and POST must be float grayscale images in [0,1].

    Returns:
        best_post
        angle
        score
        dx
        dy
    """

    best = None


    rotations = [
        (0,   0),
        (90,  1),
        (180, 2),
        (270, 3),
    ]


    for angle, k in rotations:

        # ====================================================
        # ROTATE EXACTLY 90° MULTIPLES
        # No interpolation here.
        # ====================================================

        candidate = np.rot90(
            post,
            k=k
        ).copy()


        # ====================================================
        # MATCH GLOBAL LIGHTING
        # ====================================================

        candidate = match_global_lighting(
            start,
            candidate
        )


        # ====================================================
        # SMALL X/Y CORRECTION ONLY
        # ====================================================

        candidate, dx, dy, response = (
            align_translation(
                start,
                candidate
            )
        )


        # ====================================================
        # HOW WELL DOES IT MATCH?
        # ====================================================

        score = robust_match_score(
            start,
            candidate
        )


        print(
            f"rotation={angle:3d}° "
            f"score={score:.6f} "
            f"shift=({dx:.2f},{dy:.2f})"
        )


        if (
            best is None
            or score < best["score"]
        ):

            best = {
                "image": candidate,
                "angle": angle,
                "score": score,
                "dx": dx,
                "dy": dy,
                "response": response,
            }


    return best


# ============================================================
# FINAL RESIDUAL
# ============================================================

def compare_start_post(
    start,
    post
):

    best = find_best_post_orientation(
        start,
        post
    )


    aligned_post = best[
        "image"
    ]


    residual = np.abs(
        aligned_post
        -
        start
    )


    # Ignore interpolation/crop borders
    BORDER = 20

    valid_residual = residual.copy()

    valid_residual[:BORDER, :] = 0
    valid_residual[-BORDER:, :] = 0
    valid_residual[:, :BORDER] = 0
    valid_residual[:, -BORDER:] = 0


    return (
        aligned_post,
        valid_residual,
        best
    )
