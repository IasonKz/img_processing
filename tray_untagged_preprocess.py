#!/usr/bin/env python3
"""Create an UNTAGGED vig/top/bottom dataset directly from a tester tray.

Requires Python 3.10+ and: python -m pip install numpy opencv-python
Run with no arguments for folder pickers, or:
    python tray_untagged_preprocess.py --tray "D:/data/C00000094" --output "D:/data/c94_untagged"

NO CSV reading, EvaluationResults dependency, labels, model, uploads or downloads.
All recognized VI captures are retained by default (including before/after).
Use --stage final or --stage start to select only those TOP/BOTTOM captures.
VIG is selected by FlatFieldBright or ViVig filename tokens, not Align*/SideImage.
Adjust CAPTURE_PATTERNS below or supply --top-pattern/--bottom-pattern/--vig-pattern
if another tester uses different names. Patterns are case-insensitive regexes.

TOP/BOTTOM preprocessing is copied unchanged from good_bad_folders-3.py:
native-scale square crop and geometric alignment, 4%/3% margins; no fixed resize.
It DOES NOT infer/correct the absolute 0/90/180/270 orientation of the die.
TOP/BOTTOM pixel brightness is unchanged except explicit uint16 -> uint8 conversion.
VIG keeps original spatial dimensions and uses the SAME gain/gamma brightness
(target=0.80, nonzero percentile=99.5). --no-vig-brightness disables that step.
uint16 defaults to white level 65535; use --white-level 4095 ONLY for a known
12-bit sensor stored in uint16. Conversion to uint8 loses intensity precision.
Geometric interpolation is linear; originals remain untouched.

The dataset has only three subdirectories: vig/, top/, bottom/.
JSON reports are at its root. Failed ORIGINALS go to the sibling
<output-name>__review/, never into the dataset. Optional --debug overlays go
to sibling <output-name>__debug/. Output and these folders must be new/empty.
Exit 0 = completed; 2 = preprocessing failures; 1 = invalid setup.
--dry-run scans names and prints counts without writing or testing geometry.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies: python -m pip install numpy opencv-python") from exc

# Optional paths for PyCharm Run. Leave None for folder selection dialogs.
TRAY_ROOT = None   # Example: Path(r"D:\data\C00000094")
OUTPUT_ROOT = None # Example: Path(r"D:\data\c94_untagged")

VIG_TARGET_LEVEL = 0.80
VIG_REFERENCE_PERCENTILE = 99.5
TASKS = ("vig", "top", "bottom")
IMAGE_EXTENSIONS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
CAPTURE_PATTERNS = {
    "top": r"(?:^|_)ViTop[A-Za-z0-9]*(?=_|\.|$)",
    "bottom": r"(?:^|_)ViBottom[A-Za-z0-9]*(?=_|\.|$)",
    "vig": r"(?:^|_)(?:FlatFieldBright|ViVig[A-Za-z0-9]*)(?=_|\.|$)",
}

# Integrated TOP/BOTTOM geometry (no external processing script).
TOP_OFFSET_RATIO = 0.04
BOTTOM_OFFSET_RATIO = 0.03
TOP_SEARCH_RATIO = 0.62
TOP_MAX_CENTER_OFFSET_X = 0.2
TOP_MAX_CENTER_OFFSET_Y = 0.2
TOP_MIN_SIZE_RATIO = 0.08
TOP_MAX_SIZE_RATIO = 0.42
TOP_EXPECTED_SIZE_RATIO = 0.18
TOP_MIN_DETECTION_SCORE = 7.0
TOP_MAX_ROTATION_DEG = 15.0
BOTTOM_MAX_CENTER_OFFSET_X = 0.34
BOTTOM_MAX_CENTER_OFFSET_Y = 0.34
BOTTOM_MIN_SIDE_RATIO = 0.14
BOTTOM_MAX_SIDE_RATIO = 0.52
BOTTOM_MAX_ROTATION_DEG = 18.0
BOTTOM_MIN_DETECTION_SCORE = 3.0
BOTTOM_EDGE_SEARCH_RATIO = 0.28

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
            g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    if g.dtype == np.uint8:
        return g.copy()
    x = g.astype(np.float32)
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * x).astype(np.uint8)

def order_points(points):
    pts = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered

def rectangle_angle(box):
    box = order_points(box)
    tl, tr, br, bl = box
    vector = tr - tl
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    while angle >= 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return float(angle)

def detect_top_die(original):
    g = gray8(original)
    H, W = g.shape[:2]
    image_cx = W / 2.0
    image_cy = H / 2.0
    search_w = int(W * TOP_SEARCH_RATIO)
    search_h = int(H * TOP_SEARCH_RATIO)
    sx1 = max(0, int(image_cx - search_w / 2))
    sy1 = max(0, int(image_cy - search_h / 2))
    sx2 = min(W, sx1 + search_w)
    sy2 = min(H, sy1 + search_h)
    roi = g[sy1:sy2, sx1:sx2].copy()
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    _, bright = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = float(np.median(enhanced))
    low = int(max(10, 0.6 * median))
    high = int(min(255, max(low + 30, 1.4 * median)))
    edges = cv2.Canny(enhanced, low, high)
    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel9, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel5, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel9, iterations=3)
    edges = cv2.dilate(edges, kernel5, iterations=1)
    contours1, _ = cv2.findContours(bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1 + contours2
    best = None
    best_score = -1e+30
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box[:, 0] += sx1
        box[:, 1] += sy1
        box = order_points(box)
        tl, tr, br, bl = box
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        width = (top + bottom) / 2.0
        height = (left + right) / 2.0
        if width < 5 or height < 5:
            continue
        long_side = max(width, height)
        short_side = min(width, height)
        aspect = long_side / short_side
        if aspect > 1.7:
            continue
        center = np.mean(box, axis=0)
        cx = float(center[0])
        cy = float(center[1])
        dx = abs(cx - image_cx) / W
        dy = abs(cy - image_cy) / H
        if dx > TOP_MAX_CENTER_OFFSET_X:
            continue
        if dy > TOP_MAX_CENTER_OFFSET_Y:
            continue
        size_ratio = 0.5 * (width / W + height / H)
        if size_ratio < TOP_MIN_SIZE_RATIO or size_ratio > TOP_MAX_SIZE_RATIO:
            continue
        center_distance = np.sqrt(dx * dx + dy * dy)
        max_distance = np.sqrt(TOP_MAX_CENTER_OFFSET_X ** 2 + TOP_MAX_CENTER_OFFSET_Y ** 2)
        center_score = 1.0 - center_distance / max_distance
        center_score = float(np.clip(center_score, 0.0, 1.0))
        square_score = 1.0 / aspect
        rect_area = width * height
        fill_score = min(area / max(rect_area, 1.0), 1.0)
        size_score = 1.0 - abs(size_ratio - TOP_EXPECTED_SIZE_RATIO) / TOP_EXPECTED_SIZE_RATIO
        size_score = float(np.clip(size_score, 0.0, 1.0))
        score = 10.0 * center_score + 2.0 * square_score + 1.5 * fill_score + 1.5 * size_score
        if score > best_score:
            best_score = score
            best = {'center': np.array([cx, cy], dtype=np.float32), 'box': box.copy(), 'width': float(width), 'height': float(height), 'angle': rectangle_angle(box), 'score': float(score)}
    return best

def build_top_square(detection):
    box = order_points(detection['box'])
    tl, tr, br, bl = box
    center = detection['center']
    u = (tr - tl + (br - bl)) / 2.0
    norm = np.linalg.norm(u)
    if norm <= 1e-06:
        return None
    u = u / norm
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]], dtype=np.float32)
    if v[1] < 0:
        v = -v
    die_side = max(detection['width'], detection['height'])
    square_side = die_side * (1.0 + 2.0 * TOP_OFFSET_RATIO)
    half = square_side / 2.0
    tl2 = center - u * half - v * half
    tr2 = center + u * half - v * half
    br2 = center + u * half + v * half
    bl2 = center - u * half + v * half
    square = np.array([tl2, tr2, br2, bl2], dtype=np.float32)
    return (square, square_side)

def square_inside_image(square, image):
    H, W = image.shape[:2]
    xs = square[:, 0]
    ys = square[:, 1]
    return bool(np.all(xs >= 0) and np.all(xs < W) and np.all(ys >= 0) and np.all(ys < H))

def rectify_top(original, square, square_side):
    source = order_points(square)
    native_size = max(2, int(math.ceil(square_side)) + 1)
    destination = np.array([[0, 0], [native_size - 1, 0], [native_size - 1, native_size - 1], [0, native_size - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(original, transform, (native_size, native_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop

def make_top_debug(original, square, detection):
    g = gray8(original)
    debug = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    points = np.round(square).astype(np.int32)
    cv2.polylines(debug, [points.reshape((-1, 1, 2))], True, (0, 255, 0), 4)
    center = tuple(np.round(detection['center']).astype(int))
    cv2.circle(debug, center, 6, (0, 0, 255), -1)
    cv2.putText(debug, f"TOP score={detection['score']:.2f} angle={detection['angle']:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return debug

def rectangle_contrast(gray, box):
    H, W = gray.shape[:2]
    box = np.asarray(box, dtype=np.float32)
    center = np.mean(box, axis=0)
    inner_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(inner_mask, np.round(box).astype(np.int32), 255)
    outer_box = center + 1.28 * (box - center)
    outer_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(outer_mask, np.round(outer_box).astype(np.int32), 255)
    ring_mask = cv2.subtract(outer_mask, inner_mask)
    inside = gray[inner_mask > 0]
    outside = gray[ring_mask > 0]
    if len(inside) == 0 or len(outside) == 0:
        return 0.0
    return (float(np.mean(inside)) - float(np.mean(outside))) / 255.0

def detect_bottom_die(original):
    gray = gray8(original)
    H, W = gray.shape[:2]
    D = min(H, W)
    image_center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    sigma = max(12.0, D * 0.018)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    otsu_value, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = [float(otsu_value), float(np.percentile(smooth, 60)), float(np.percentile(smooth, 66)), float(np.percentile(smooth, 72))]
    close_size = max(15, int(D * 0.025))
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    candidates = []
    for threshold in thresholds:
        _, mask = cv2.threshold(smooth, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            hull = cv2.convexHull(contour)
            rect = cv2.minAreaRect(hull)
            (cx, cy), (rw, rh), _ = rect
            if rw < 10 or rh < 10:
                continue
            long_side = max(rw, rh)
            short_side = min(rw, rh)
            side_ratio = long_side / D
            if not BOTTOM_MIN_SIDE_RATIO <= side_ratio <= BOTTOM_MAX_SIDE_RATIO:
                continue
            aspect = long_side / max(short_side, 1.0)
            if aspect > 1.55:
                continue
            dx = abs(cx - image_center[0]) / W
            dy = abs(cy - image_center[1]) / H
            if dx > BOTTOM_MAX_CENTER_OFFSET_X:
                continue
            if dy > BOTTOM_MAX_CENTER_OFFSET_Y:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            contrast = rectangle_contrast(gray, box)
            rect_area = rw * rh
            fill = float(np.clip(area / max(rect_area, 1.0), 0.0, 1.0))
            square_score = 1.0 / aspect
            expected_ratio = 0.28
            size_score = 1.0 - abs(side_ratio - expected_ratio) / expected_ratio
            size_score = float(np.clip(size_score, 0.0, 1.0))
            score = 3.5 * contrast + 2.5 * square_score + 2.0 * fill + 1.0 * size_score
            candidates.append({'center': np.array([cx, cy], dtype=np.float32), 'width': float(rw), 'height': float(rh), 'box': box, 'score': float(score)})
    if not candidates:
        return None
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[0]

def rotate_bottom_image(original, center, angle):
    H, W = original.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotated = cv2.warpAffine(original, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated

def find_profile_edge(profile, expected_position, search_radius, positive):
    profile = profile.astype(np.float32)
    size = max(9, int(len(profile) * 0.01))
    if size % 2 == 0:
        size += 1
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (size, 1), 0).reshape(-1)
    gradient = np.gradient(smooth)
    low = max(1, int(expected_position - search_radius))
    high = min(len(profile) - 2, int(expected_position + search_radius))
    if high <= low:
        return None
    local = gradient[low:high + 1]
    if positive:
        index = low + int(np.argmax(local))
    else:
        index = low + int(np.argmin(local))
    return index

def refine_bottom_edges(rotated, approx_center, approx_side):
    gray = gray8(rotated)
    H, W = gray.shape[:2]
    cx = float(approx_center[0])
    cy = float(approx_center[1])
    sigma = max(8.0, approx_side * 0.035)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    band_half = int(approx_side * 0.3)
    y1 = max(0, int(cy - band_half))
    y2 = min(H, int(cy + band_half))
    x1 = max(0, int(cx - band_half))
    x2 = min(W, int(cx + band_half))
    if y2 <= y1 or x2 <= x1:
        return None
    x_profile = np.mean(smooth[y1:y2, :], axis=0)
    y_profile = np.mean(smooth[:, x1:x2], axis=1)
    expected_left = cx - approx_side / 2
    expected_right = cx + approx_side / 2
    expected_top = cy - approx_side / 2
    expected_bottom = cy + approx_side / 2
    search_radius = approx_side * BOTTOM_EDGE_SEARCH_RATIO
    left = find_profile_edge(x_profile, expected_left, search_radius, positive=True)
    right = find_profile_edge(x_profile, expected_right, search_radius, positive=False)
    top = find_profile_edge(y_profile, expected_top, search_radius, positive=True)
    bottom = find_profile_edge(y_profile, expected_bottom, search_radius, positive=False)
    if left is None or right is None or top is None or (bottom is None):
        return None
    span_x = right - left
    span_y = bottom - top
    if span_x < approx_side * 0.65 or span_x > approx_side * 1.35:
        return None
    if span_y < approx_side * 0.65 or span_y > approx_side * 1.35:
        return None
    refined_cx = (left + right) / 2.0
    refined_cy = (top + bottom) / 2.0
    die_side = max(span_x, span_y)
    return {'center': np.array([refined_cx, refined_cy], dtype=np.float32), 'side': float(die_side), 'edges': (int(left), int(top), int(right), int(bottom))}

def crop_bottom(rotated, refined):
    H, W = rotated.shape[:2]
    cx = float(refined['center'][0])
    cy = float(refined['center'][1])
    side = int(round(refined['side'] * (1.0 + 2.0 * BOTTOM_OFFSET_RATIO)))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side
    if x1 < 0 or y1 < 0 or x2 > W or (y2 > H):
        return (None, None)
    crop = rotated[y1:y2, x1:x2].copy()
    return (crop, (x1, y1, x2, y2))

def make_bottom_debug(rotated, refined, crop_bbox):
    g = gray8(rotated)
    debug = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    left, top, right, bottom = refined['edges']
    cv2.rectangle(debug, (left, top), (right, bottom), (0, 255, 255), 3)
    x1, y1, x2, y2 = crop_bbox
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 4)
    center = tuple(np.round(refined['center']).astype(int))
    cv2.circle(debug, center, 6, (0, 0, 255), -1)
    return debug


# Folder handling and CLI. The geometry functions above come from the old script.
class DetectionFailure(ValueError):
    pass


def process_array(original, task, debug=False):
    """Use the original top/bottom geometry; do not change brightness."""
    if original.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise DetectionFailure(f'Unsupported bit depth: {original.dtype}')
    if original.ndim != 2 and not (original.ndim == 3 and original.shape[2] in (3, 4)):
        raise DetectionFailure(f'Unsupported image shape: {original.shape}')
    if min(original.shape[:2]) < 32:
        raise DetectionFailure('Image is too small for the original detector')
    overlay = None
    if task == 'top':
        detection = detect_top_die(original)
        if detection is None:
            raise DetectionFailure('TOP: no die detected')
        if detection['score'] < TOP_MIN_DETECTION_SCORE:
            raise DetectionFailure('TOP: low detection score')
        if abs(detection['angle']) > TOP_MAX_ROTATION_DEG:
            raise DetectionFailure('TOP: rotation exceeds 15 degrees')
        result = build_top_square(detection)
        if result is None:
            raise DetectionFailure('TOP: invalid square geometry')
        square, square_side = result
        if not square_inside_image(square, original):
            raise DetectionFailure('TOP: crop lies outside image')
        crop = rectify_top(original, square, square_side)
        angle = detection['angle']
        if debug:
            overlay = make_top_debug(original, square, detection)
    elif task == 'bottom':
        detection = detect_bottom_die(original)
        if detection is None:
            raise DetectionFailure('BOTTOM: no die detected')
        if detection['score'] < BOTTOM_MIN_DETECTION_SCORE:
            raise DetectionFailure('BOTTOM: low detection score')
        angle = rectangle_angle(detection['box'])
        if abs(angle) > BOTTOM_MAX_ROTATION_DEG:
            raise DetectionFailure('BOTTOM: rotation exceeds 18 degrees')
        rotated = rotate_bottom_image(original, detection['center'], angle)
        refined = refine_bottom_edges(rotated, detection['center'],
                                      max(detection['width'], detection['height']))
        if refined is None:
            raise DetectionFailure('BOTTOM: edge refinement failed')
        crop, bbox = crop_bottom(rotated, refined)
        if crop is None:
            raise DetectionFailure('BOTTOM: crop lies outside image')
        if debug:
            overlay = make_bottom_debug(rotated, refined, bbox)
    else:
        raise ValueError(f'Only top and bottom are supported, got {task!r}')

    native_side = crop.shape[0]
    final = crop
    return final, {'angle_deg': float(angle), 'score': float(detection['score']),
                   'native_side_px': native_side, 'output_side_px': final.shape[0],
                   'dtype': str(final.dtype)}, overlay




def decode_image(path):
    if cv2.imcount(str(path)) > 1:
        raise ValueError('Multi-frame image: refusing to discard frames')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Cannot decode image')
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError(f'Unsupported dtype: {image.dtype}')
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError(f'Unsupported image shape: {image.shape}')
    if image.ndim == 3 and image.shape[2] == 4:
        if not np.all(image[..., 3] == np.iinfo(image.dtype).max):
            raise ValueError('Non-opaque alpha requires an explicit compositing policy')
        image = image[..., :3].copy()
    return image


def intensity_to_v5(image, task, brightness=True, white_level=None):
    """One monotonic LUT; no spatial operations or implicit precision loss."""
    maximum_level = np.iinfo(image.dtype).max
    white = float(white_level if image.dtype == np.uint16 and white_level is not None else maximum_level)
    if not 0 < white <= maximum_level:
        raise ValueError('White level must be within the source integer range')
    peak = int(image.max())
    if peak > white:
        raise ValueError(f'Input maximum {peak} exceeds white level {white}; refusing clipping')
    gain, gamma, reference = 1.0, 1.0, None
    if task == 'vig' and brightness:
        samples = image[image > 0]
        if samples.size:
            reference = float(np.percentile(samples, VIG_REFERENCE_PERCENTILE))
            gain = max(1.0, min(VIG_TARGET_LEVEL * white / reference, white / peak))
            scaled = reference * gain / white
            if 0 < scaled < VIG_TARGET_LEVEL:
                gamma = math.log(VIG_TARGET_LEVEL) / math.log(scaled)
    levels = np.arange(maximum_level + 1, dtype=np.float64)
    lut = np.rint(np.power(np.clip(levels * gain / white, 0, 1), gamma) * 255).astype(np.uint8)
    result = lut[image]
    return result, dict(source_dtype=str(image.dtype), output_dtype='uint8',
                        input_min=int(image.min()), input_max=peak,
                        white_level=white, gain=gain, gamma=gamma,
                        reference=reference, all_black=peak == 0,
                        output_zero_fraction=float(np.mean(result == 0)),
                        output_white_fraction=float(np.mean(result == 255)))


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError('PNG encoding failed')
    # Validate encoded bytes before committing the destination.
    check = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if check is None or not np.array_equal(check, image):
        raise ValueError('PNG round-trip validation failed')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())


def prepare_one(source, task, brightness=True, white_level=None, debug=False):
    original = decode_image(source)
    metadata = {'input_width': original.shape[1], 'input_height': original.shape[0]}
    overlay = None
    if task in ('top', 'bottom'):
        prepared, geometry, overlay = process_array(original, task, debug=debug)
        metadata['geometry'] = geometry
    else:
        prepared = original
    result, intensity = intensity_to_v5(prepared, task, brightness, white_level)
    metadata.update(intensity)
    metadata.update(output_width=result.shape[1], output_height=result.shape[0])
    if task == 'vig' and result.shape != original.shape:
        raise ValueError('VIG dimensions changed unexpectedly')
    return result, metadata, overlay



def pick_folders(tray, output):
    """No GUI toolkit dependency unless the folder picker is requested."""
    if tray is not None and output is not None:
        return Path(tray), Path(output)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
    except Exception as exc:
        raise ValueError("Folder dialogs unavailable. Use --tray PATH --output PATH.") from exc
    try:
        if tray is None:
            selected = filedialog.askdirectory(
                parent=root, title="1. Select the ORIGINAL tester tray folder (C000000...)")
            if not selected:
                raise SystemExit(0)
            tray = Path(selected)
        if output is None:
            selected = filedialog.askdirectory(
                parent=root, title="2. Select a NEW or EMPTY output folder (outside the tray)",
                initialdir=str(Path(tray).parent), mustexist=False)
            if not selected:
                raise SystemExit(0)
            output = Path(selected)
    finally:
        root.destroy()
    return Path(tray), Path(output)


def find_scan_root(tray):
    """Prefer raw measurement data; never scan EvaluationResults."""
    measurement = [p for p in tray.iterdir()
                   if p.is_dir() and not p.is_symlink() and p.name.casefold() == "measurementdata"]
    if len(measurement) > 1:
        raise ValueError("Ambiguous MeasurementData directories")
    if measurement:
        results = [p for p in measurement[0].iterdir()
                   if p.is_dir() and not p.is_symlink() and p.name.casefold() == "results"]
        if len(results) > 1:
            raise ValueError("Ambiguous Results directories")
        return results[0] if results else measurement[0]
    return tray  # Also accept MeasurementData/Results itself as the selected input.


def compile_patterns(overrides):
    return {task: re.compile(overrides.get(task) or CAPTURE_PATTERNS[task], re.IGNORECASE)
            for task in TASKS}


def capture_stage(name, task):
    if task == "vig":
        return "not_applicable"
    match = re.search(r"(?:^|_)Vi(?:Top|Bottom)([A-Za-z0-9]*)(?=_|\.|$)", name, re.I)
    if not match:
        return "unknown"
    token = match.group(1).casefold()
    if "final" in token or "post" in token:
        return "final"
    if "start" in token or "pre" in token:
        return "start"
    return "unknown"


def detect_task(name, patterns):
    matches = [task for task, pattern in patterns.items() if pattern.search(name)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous capture name matches several tasks: {name}")
    return matches[0] if matches else None


def scan_images(tray, patterns, stage="all"):
    root = find_scan_root(tray)
    plan, skipped = [], []
    def scan_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs
                         if d.casefold() != "evaluationresults" and not (base / d).is_symlink())
        for name in sorted(files):
            source = base / name
            if source.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            relative = str(source.relative_to(tray))
            if source.is_symlink():
                skipped.append(dict(source=relative, status="skipped", reason="symlink"))
                continue
            task = detect_task(name, patterns)
            if task is None:
                skipped.append(dict(source=relative, status="skipped", reason="unrecognized_capture"))
                continue
            actual_stage = capture_stage(name, task)
            if task != "vig" and stage != "all" and actual_stage != stage:
                skipped.append(dict(source=relative, task=task, status="skipped",
                                    reason="stage_filter", capture_stage=actual_stage))
                continue
            plan.append((task, source, actual_stage))
    plan.sort(key=lambda item: (TASKS.index(item[0]), str(item[1]).casefold()))
    return root, plan, skipped


def validate_output(tray, output, debug):
    if output == tray or output.is_relative_to(tray) or tray.is_relative_to(output):
        raise ValueError("Output must be OUTSIDE the input tray, and must not contain it.")
    review = output.with_name(output.name + "__review")
    debug_root = output.with_name(output.name + "__debug")
    for folder in [output, review] + ([debug_root] if debug else []):
        if folder == tray or folder.is_relative_to(tray) or tray.is_relative_to(folder):
            raise ValueError(f"Auxiliary folder overlaps the input: {folder}")
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise ValueError(f"Choose a NEW or EMPTY folder. Already contains files: {folder}")
    return review, debug_root


def allocate_names(plan, tray):
    names, reserved = [], set()
    for task, source, stage in plan:
        name = source.stem + ".png"
        key = (task, name.casefold())
        if key in reserved:
            relative = source.relative_to(tray).as_posix()
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
            name = f"{source.stem}__{digest}.png"
            number = 2
            while (task, name.casefold()) in reserved:
                name = f"{source.stem}__{digest}_{number}.png"
                number += 1
        reserved.add((task, name.casefold()))
        names.append((task, source, stage, name))
    return names


def copy_original_exclusive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def export_images(plan, skipped, tray, scan_root, output, args, patterns):
    review, debug_root = validate_output(tray, output, args.debug)
    destinations = allocate_names(plan, tray)
    output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        (output / task).mkdir()
    settings = dict(
        format_version=1, dataset_kind="untagged", labels_present=False,
        original_tray=str(tray), scan_root=str(scan_root), uses_csv=False,
        capture_patterns={t: p.pattern for t, p in patterns.items()},
        stage=args.stage, fixed_resize=False, output_bit_depth=8,
        vig_brightness=not args.no_vig_brightness,
        vig_target=VIG_TARGET_LEVEL, vig_percentile=VIG_REFERENCE_PERCENTILE,
        uint16_white_level=args.white_level or 65535,
        top_margin=TOP_OFFSET_RATIO, bottom_margin=BOTTOM_OFFSET_RATIO,
        geometry_source="good_bad_folders-3.py (unchanged geometry)",
        alignment_interpolation="linear", absolute_quarter_turn_orientation_normalization=False,
        review_folder=str(review), debug_folder=str(debug_root) if args.debug else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output / "preparation.json", settings)
    counts, failures, black_images, review_copy_failures = Counter(), 0, 0, 0
    with (output / "processing.jsonl").open("x", encoding="utf-8") as log:
        for record in skipped:
            log.write(json.dumps(record, ensure_ascii=True) + "\n")
        for number, (task, source, stage, name) in enumerate(destinations, 1):
            destination = output / task / name
            record = dict(task=task, source=str(source.relative_to(tray)), capture_stage=stage,
                          destination=str(destination.relative_to(output)))
            try:
                image, metadata, overlay = prepare_one(
                    source, task, brightness=not args.no_vig_brightness,
                    white_level=args.white_level, debug=args.debug)
                write_png(destination, image)
                record.update(status="ok", **metadata)
                counts[task] += 1
                if metadata["all_black"]:
                    black_images += 1
                    print(f"WARNING all-black {task} image retained: {source.name}")
                if overlay is not None:
                    try:
                        write_png(debug_root / task / name, overlay)
                    except (ValueError, OSError, cv2.error) as exc:
                        record["debug_error"] = str(exc)
                        print(f"WARNING debug overlay could not be saved: {exc}")
                if not args.quiet or number % 100 == 0 or number == len(destinations):
                    print(f"[{number}/{len(destinations)}] {task}: {name} "
                          f"{image.shape[1]}x{image.shape[0]} uint8")
            except (ValueError, OSError, cv2.error) as exc:
                failures += 1
                record.update(status="failed", error=str(exc))
                print(f"FAILED {task}: {source.name}: {exc}")
                review_file = review / task / (Path(name).stem + source.suffix)
                try:
                    copy_original_exclusive(source, review_file)
                    record["review_original"] = str(review_file)
                except OSError as copy_error:
                    review_copy_failures += 1
                    record["review_copy_error"] = str(copy_error)
                    print(f"WARNING original could not be copied for review: {copy_error}")
            log.write(json.dumps(record, ensure_ascii=True) + "\n")
            log.flush()
    summary = dict(
        processed={task: counts[task] for task in TASKS}, failed_images=failures,
        review_copy_failures=review_copy_failures, all_black_vig_images=black_images,
        skipped_images=len(skipped),
        skipped_reasons=dict(Counter(r["reason"] for r in skipped)),
        selected_images=len(plan), complete=failures == 0)
    write_json(output / "summary.json", summary)
    return counts, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tray", type=Path, default=TRAY_ROOT,
                        help="Original tray folder, or raw Results folder")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT,
                        help="New/empty output folder OUTSIDE the input")
    parser.add_argument("--stage", choices=("all", "final", "start"), default="all",
                        help="TOP/BOTTOM capture selection; VIG unaffected (default all)")
    parser.add_argument("--top-pattern", help="Override case-insensitive TOP filename regex")
    parser.add_argument("--bottom-pattern", help="Override BOTTOM filename regex")
    parser.add_argument("--vig-pattern", help="Override VIG filename regex")
    parser.add_argument("--no-vig-brightness", action="store_true")
    parser.add_argument("--white-level", type=int, help="Known uint16 white level, e.g. 4095")
    parser.add_argument("--dry-run", action="store_true", help="Scan names only, write nothing")
    parser.add_argument("--debug", action="store_true", help="Sibling full-size crop overlays")
    parser.add_argument("--quiet", action="store_true", help="Print progress only every 100 images")
    args = parser.parse_args(argv)
    try:
        if args.white_level is not None and not 1 <= args.white_level <= 65535:
            raise ValueError("--white-level must be 1..65535")
        if not 0 < VIG_TARGET_LEVEL < 1 or not 0 < VIG_REFERENCE_PERCENTILE <= 100:
            raise ValueError("Invalid VIG brightness settings")
        # Dry-run with a --tray needs no output dialog.
        if args.dry_run and args.tray is not None:
            tray, output = args.tray, args.output
        else:
            tray, output = pick_folders(args.tray, args.output)
        tray = tray.expanduser().resolve()
        if not tray.is_dir():
            raise ValueError(f"Tray does not exist: {tray}")
        patterns = compile_patterns({task: getattr(args, task + "_pattern") for task in TASKS})
        if output is not None:
            output = output.expanduser().resolve()
            validate_output(tray, output, args.debug)
        scan_root, plan, skipped = scan_images(tray, patterns, args.stage)
        print(f"Scanning: {scan_root}")
        counts = Counter(task for task, _, _ in plan)
        for task in TASKS:
            print(f"Selected {task}: {counts[task]}")
            for _, source, stage in [item for item in plan if item[0] == task][:3]:
                print(f"  {source.name} [{stage}]")
            if counts[task] == 0:
                print(f"WARNING no {task} captures matched. Check the filename pattern.")
        skipped_counts = Counter(item["reason"] for item in skipped)
        print(f"Skipped images: {len(skipped)} ({dict(skipped_counts)})")
        unknown = [r["source"] for r in skipped if r["reason"] == "unrecognized_capture"]
        if unknown:
            print("Examples of ignored image names:")
            for name in unknown[:5]:
                print(f"  {name}")
        if not plan:
            raise ValueError("No supported image captures found. Check --tray and patterns.")
        if args.dry_run:
            print("DRY RUN: no files written; crop detection/pixel conversion NOT tested.")
            return 0
        counts, failures = export_images(plan, skipped, tray, scan_root, output, args, patterns)
        print("\nDataset: " + str(output))
        for task in TASKS:
            print(f"  {task}: {counts[task]} images")
        print("Originals were not modified. No good/bad labels were assigned.")
        if failures:
            print(f"INCOMPLETE: {failures} failures. See processing.jsonl and "
                  f"{output.name}__review next to the dataset.")
            return 2
        print("Completed.")
        return 0
    except (ValueError, OSError, UnicodeError, re.error, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
