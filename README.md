import cv2
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

TOP_IMAGE = r"C:\path\to\your\top_image.bmp"
BOTTOM_IMAGE = r"C:\path\to\your\bottom_image.bmp"

OUTPUT_FOLDER = r"C:\path\to\references"

OUTPUT_SIZE = 600


# ============================================================
# SELECT 4 CORNERS
# ============================================================

def select_four_corners(image, window_name):
    """
    Click corners in this order:

        1 -------- 2
        |          |
        |          |
        4 -------- 3

    1 = top-left
    2 = top-right
    3 = bottom-right
    4 = bottom-left
    """

    points = []

    # --------------------------------------------------------
    # Make display image
    # We use a separate image ONLY for visualization.
    # The original image data are not modified.
    # --------------------------------------------------------

    if image.dtype == np.uint16:
        display = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    else:
        display = image.copy()

    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    original_display = display.copy()

    def mouse_callback(event, x, y, flags, param):

        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 4:

                points.append((x, y))

                # Draw selected point
                cv2.circle(
                    display,
                    (x, y),
                    7,
                    (0, 255, 0),
                    -1
                )

                # Point number
                cv2.putText(
                    display,
                    str(len(points)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Draw lines between points
                if len(points) > 1:
                    cv2.line(
                        display,
                        points[-2],
                        points[-1],
                        (0, 255, 0),
                        2
                    )

                # Close polygon after 4th point
                if len(points) == 4:
                    cv2.line(
                        display,
                        points[3],
                        points[0],
                        (0, 255, 0),
                        2
                    )

                cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print(f"--- {window_name} ---")
    print("Click corners:")
    print("1. TOP LEFT")
    print("2. TOP RIGHT")
    print("3. BOTTOM RIGHT")
    print("4. BOTTOM LEFT")
    print()
    print("Press ENTER when finished.")
    print("Press R to reset.")
    print("Press ESC to cancel.")

    while True:

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ENTER
        if key == 13:

            if len(points) == 4:
                break
            else:
                print("You must select exactly 4 points.")

        # R = reset
        elif key == ord("r"):

            points.clear()
            display = original_display.copy()

            print("Points reset.")

        # ESC = cancel
        elif key == 27:

            cv2.destroyWindow(window_name)
            raise RuntimeError("Selection cancelled.")

    cv2.destroyWindow(window_name)

    return np.array(points, dtype=np.float32)


# ============================================================
# CREATE REFERENCE IMAGE
# ============================================================

def create_reference(input_path, output_path, name):

    print()
    print("=" * 60)
    print(f"Creating {name} reference")
    print("=" * 60)

    # Read original without changing bit depth
    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{input_path}"
        )

    print(f"Input image: {input_path}")
    print(f"Shape: {image.shape}")
    print(f"dtype: {image.dtype}")

    # User selects ROI corners
    source_points = select_four_corners(
        image,
        f"Select corners - {name}"
    )

    # --------------------------------------------------------
    # Destination points
    #
    # The selected quadrilateral will become a perfect
    # 600 x 600 square.
    # --------------------------------------------------------

    destination_points = np.array(
        [
            [0, 0],
            [OUTPUT_SIZE - 1, 0],
            [OUTPUT_SIZE - 1, OUTPUT_SIZE - 1],
            [0, OUTPUT_SIZE - 1]
        ],
        dtype=np.float32
    )

    # Perspective transformation
    transform_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    # Warp ORIGINAL image
    reference = cv2.warpPerspective(
        image,
        transform_matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR
    )

    # Save
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        reference
    )

    if not success:
        raise RuntimeError(
            f"Could not save:\n{output_path}"
        )

    print()
    print(f"Saved reference:")
    print(output_path)

    print(f"Reference size: {reference.shape}")
    print(f"Reference dtype: {reference.dtype}")

    return reference


# ============================================================
# MAIN
# ============================================================

def main():

    output_folder = Path(OUTPUT_FOLDER)

    top_output = output_folder / "reference_top.bmp"
    bottom_output = output_folder / "reference_bottom.bmp"

    # TOP
    create_reference(
        TOP_IMAGE,
        top_output,
        "TOP"
    )

    # BOTTOM
    create_reference(
        BOTTOM_IMAGE,
        bottom_output,
        "BOTTOM"
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"TOP reference:")
    print(top_output)

    print()
    print(f"BOTTOM reference:")
    print(bottom_output)

    print()
    print("Original images were NOT modified.")


if __name__ == "__main__":
    main()