import cv2
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CROP_WIDTH = 600
CROP_HEIGHT = 600

# Fixed offsets
TOP_OFFSET_X = 0
TOP_OFFSET_Y = -15

BOTTOM_OFFSET_X = 0
BOTTOM_OFFSET_Y = 0


# ============================================================
# SIMPLE CENTER CROP
# ============================================================

def center_crop(image, crop_width, crop_height,
                offset_x=0, offset_y=0):

    h, w = image.shape[:2]

    # Image center + small fixed offset
    center_x = w // 2 + offset_x
    center_y = h // 2 + offset_y

    # Crop coordinates
    x1 = center_x - crop_width // 2
    y1 = center_y - crop_height // 2

    x2 = x1 + crop_width
    y2 = y1 + crop_height

    # Make sure crop stays inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    # If we touched an edge, move crop back
    # so that it remains exactly 600x600
    if x2 - x1 < crop_width:
        if x1 == 0:
            x2 = min(w, crop_width)
        else:
            x1 = max(0, w - crop_width)

    if y2 - y1 < crop_height:
        if y1 == 0:
            y2 = min(h, crop_height)
        else:
            y1 = max(0, h - crop_height)

    cropped = image[y1:y2, x1:x2].copy()

    return cropped


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(input_path, output_path, image_type):

    # Read without changing original bit depth
    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        print(f"Could not read: {input_path}")
        return

    if image_type.lower() == "top":

        crop = center_crop(
            image,
            CROP_WIDTH,
            CROP_HEIGHT,
            TOP_OFFSET_X,
            TOP_OFFSET_Y
        )

    elif image_type.lower() == "bottom":

        crop = center_crop(
            image,
            CROP_WIDTH,
            CROP_HEIGHT,
            BOTTOM_OFFSET_X,
            BOTTOM_OFFSET_Y
        )

    else:
        raise ValueError(
            "image_type must be 'top' or 'bottom'"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        crop
    )

    print(
        f"Saved: {output_path} "
        f"shape={crop.shape}"
    )


# ============================================================
# EXAMPLE
# ============================================================

process_image(
    r"C:\data\top_image.bmp",
    r"C:\processed\top_image.bmp",
    "top"
)

process_image(
    r"C:\data\bottom_image.bmp",
    r"C:\processed\bottom_image.bmp",
    "bottom"
)