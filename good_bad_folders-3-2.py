"""Prepare manually labelled datasets for Inspection Studio v5.

Install (Windows PowerShell): py -3.12 -m pip install numpy opencv-python
Run: py -3.12 good_bad_folders-3.py
Or: py -3.12 good_bad_folders-3.py --csv evaluation_results.csv --output D:/prepared

Edit CSV_FILES and OUTPUT_ROOT below for the IDE Run button. Output must be new
or empty. CSV labels determine good/bad; no model predictions are used.

Output: vig/good, vig/bad, top/good, top/bad, bottom/good, bottom/bad.
VIG: original dimensions, global gain/gamma brightness, explicit uint8 PNG.
TOP/BOTTOM: embedded task-specific detection, alignment and native-scale crop.
There is NO fixed-size resize, thumbnail generation or downsampling. Alignment
uses linear interpolation: aligned pixels are not identical to source pixels.
TOP keeps 4% margin, BOTTOM 3%. Cropping removes material outside this ROI;
check overlays, especially for outer defects, before accepting the dataset.

v5 accepts 8-bit images. uint16 -> uint8 quantization is explicit and loses
intensity precision, NOT spatial resolution. Originals remain unchanged.
Top/bottom default conversion uses the full dtype range, without brightness
equalization. For a known 12-bit sensor stored in uint16, set --white-level 4095.
Never infer sensor range independently from each top/bottom image.
VIG brightness is per-image and may suppress absolute exposure differences;
use --no-vig-brightness if absolute brightness is part of the defect definition.
Apply this SAME preparation to future inference images before using v5.

Failed processing goes to _review, outside training folders, with original
labels and errors in processing.jsonl. Exit 2 means incomplete output.
--dry-run validates CSV references only; it does not validate crop detection.
--debug writes full-size crop overlays to _debug, outside training folders.
"""

import argparse
import ast
import csv
import hashlib
import math
import os
import shutil
import json
import sys
try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit('Install dependencies: py -3.12 -m pip install numpy opencv-python') from exc
from collections import Counter
from pathlib import Path, PureWindowsPath


# CONFIGURATION: change these paths before running with the IDE's Run button.
CSV_FILES = [
    Path(r"C:\Users\laka\software\data\C00000094\EvaluationResults\c94_vi_top_bot\evaluation_results.csv"),
    Path(r"C:\Users\laka\software\data\C00000094\EvaluationResults\c94_vi_vig\evaluation_results.csv"),
]
OUTPUT_ROOT = Path(r"C:\Users\laka\software\good_bad_split_94_v5")

# Optional: current tray folder, if the CSV still contains paths from another
# computer. The full suffix after the matching tray name must still exist.
# Example: Path(r"D:\data\C00000094") or Path("/home/iason/data/C00000094")
TRAY_ROOT = None

GOOD_TAGS = {"p"}
BAD_TAGS = {"f", "x"}  # x counts as bad, matching the photographed script.
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"

# VIG has no crop or spatial resampling. All model inputs are explicit 8-bit PNG.
VIG_BRIGHTNESS_ENABLED = True
VIG_TARGET_LEVEL = 0.80  # 0 < target < 1; e.g. 0.90 makes dark vig brighter.
VIG_REFERENCE_PERCENTILE = 99.5  # Calculated over nonzero intensity pixels.

TASK_COLUMNS = {
    "top": {
        "labels": ("Quality_VI_Top_Inner_Label", "Quality_VI_Top_Outer_Label"),
        "sources": ("Quality_VI_Top_Inner_Sources", "Quality_VI_Top_Outer_Sources"),
    },
    "bottom": {
        "labels": ("Quality_VI_Bottom_Label",),
        "sources": ("Quality_VI_Bottom_Sources",),
    },
    "vig": {
        "labels": (
            "Quality_VI_Vig_Particles_Label",
            "Quality_VI_Vig_PnP_Label",
            "Quality_VI_Vig_Wave_Label",
            "Quality_VI_Vig_Others_Label",
        ),
        "sources": (
            "Quality_VI_Vig_Particles_Sources",
            "Quality_VI_Vig_PnP_Sources",
            "Quality_VI_Vig_Wave_Sources",
            "Quality_VI_Vig_Others_Sources",
        ),
    },
}


def classify(row, label_columns):
    """Decide using this task's labels only; unknown tags are never good."""
    labels = [str(row.get(column) or "").strip().lower() for column in label_columns]
    if any(label in BAD_TAGS for label in labels):
        return "bad"
    if labels and all(label in GOOD_TAGS for label in labels):
        return "good"
    return None


def extract_paths(cell):
    """Read a Python list of path strings as in the original CSV, or one path."""
    value = str(cell or "").strip()
    if value.lower() in ("", "[]", "nan", "none", "null"):
        return []
    if value.startswith(("[", "(", "'", '"')):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse Sources value: {value!r}") from exc
    else:
        parsed = value
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in parsed
    ):
        raise ValueError(f"Sources must contain path strings: {value!r}")
    return list(dict.fromkeys(item.strip() for item in parsed))


def resolve_source(raw, csv_file, tray_root):
    """Resolve exact paths, never search by unit ID or basename alone."""
    normalized = raw.replace("\\", "/")
    windows_absolute = PureWindowsPath(raw).is_absolute()
    local = Path(normalized).expanduser()

    # With an explicit new tray root, prefer it over an old absolute path.
    if tray_root is not None:
        parts = normalized.split("/")
        matches = [i for i, part in enumerate(parts) if part.casefold() == tray_root.name.casefold()]
        if matches:
            if len(matches) != 1:
                raise ValueError(f"Tray name occurs more than once in path: {raw}")
            tail = parts[matches[0] + 1:]
            if not tail or ".." in tail:
                raise ValueError(f"Invalid path below tray: {raw}")
            mapped = tray_root.joinpath(*tail).resolve()
            if not mapped.is_relative_to(tray_root):
                raise ValueError(f"Path leaves the selected tray: {raw}")
            return mapped if mapped.is_file() else None

    if local.is_absolute() or windows_absolute:
        if windows_absolute and os.name != "nt":
            return None
        return local.resolve() if local.is_file() else None

    candidates = [csv_file.parent / local]
    if tray_root is not None:
        candidates.append(tray_root / local)
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(existing) > 1:
        raise ValueError(f"Ambiguous relative path: {raw}; use an absolute Sources path.")
    return next(iter(existing), None)


def build_plan(csv_files, tray_root=None):
    """Validate all rows first; refuse conflicting labels for the same image."""
    plan = {}
    stats = Counter()
    available_tasks = set()
    for csv_file in csv_files:
        print(f"Reading: {csv_file}")
        with csv_file.open("r", encoding=CSV_ENCODING, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=CSV_DELIMITER)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_file}")
            headers = [name.strip() for name in reader.fieldnames]
            if len(headers) != len(set(headers)):
                raise ValueError(f"Duplicate column names in {csv_file}")
            reader.fieldnames = headers
            active = {}
            for task, columns in TASK_COLUMNS.items():
                required = set(columns["labels"] + columns["sources"])
                present = required.intersection(headers)
                if present and present != required:
                    raise ValueError(f"{csv_file}: incomplete {task} columns. Missing: {sorted(required - present)}")
                if present:
                    active[task] = columns
            if not active:
                raise ValueError(
                    f"No supported task columns in {csv_file}.\n"
                    f"Actual headers: {headers}\nCheck TASK_COLUMNS, delimiter and encoding."
                )
            print("  Available tasks: " + ", ".join(active))
            available_tasks.update(active)
            for row in reader:
                context = f"{csv_file.name}, line {reader.line_num}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{context}: row length does not match the header.")
                for task, columns in active.items():
                    label = classify(row, columns["labels"])
                    if label is None:
                        stats[(task, "unknown_rows")] += 1
                        print(f"  SKIP unknown {task} tag: {context}, unit {row.get('Unit Identifier', '?')}")
                        continue
                    sources = []
                    for column in columns["sources"]:
                        try:
                            sources.extend(extract_paths(row[column]))
                        except ValueError as exc:
                            raise ValueError(f"{context}, {column}: {exc}") from exc
                    if not sources:
                        stats[(task, "empty_sources_rows")] += 1
                    for raw in dict.fromkeys(sources):
                        source = resolve_source(raw, csv_file, tray_root)
                        if source is None:
                            stats[(task, "missing_references")] += 1
                            print(f"  MISSING {task}: {raw}")
                            continue
                        key = (task, os.path.normcase(str(source)))
                        if key in plan:
                            if plan[key][1] != label:
                                raise ValueError(
                                    f"Conflicting {task} labels for {source}: "
                                    f"{plan[key][3]} versus {context}. No files copied."
                                )
                            stats[(task, "duplicate_references")] += 1
                            continue
                        plan[key] = (task, label, source, context)
    for task in TASK_COLUMNS:
        if task not in available_tasks:
            print(f"WARNING: no {task} columns found in the selected CSVs; {task} will be empty.")
    return list(plan.values()), stats



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


def export_plan(plan, output, brightness=True, white_level=None, debug=False):
    if not plan:
        raise ValueError('No labelled existing images to process')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Choose a NEW or EMPTY output folder: {output}')
    if any(source.is_relative_to(output) for _, _, source, _ in plan):
        raise ValueError('Input image is inside the output folder')
    destinations, used = [], set()
    for task, label, source, context in plan:
        name = source.with_suffix('.png').name
        # Reserve filenames across both labels of each task.
        key = (task, name.casefold())
        if key in used:
            digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
            name = f'{source.stem}__{digest}.png'
            number = 2
            while (task, name.casefold()) in used:
                name = f'{source.stem}__{digest}_{number}.png'
                number += 1
        used.add((task, name.casefold()))
        destinations.append((task, label, source, output / task / label / name, context))
    for task in TASK_COLUMNS:
        for label in ('good', 'bad'):
            (output / task / label).mkdir(parents=True, exist_ok=True)
    settings = dict(format_version=1, compatible_with='Inspection Studio v5',
                    spatial_resize=False, output_bit_depth=8,
                    vig_brightness=brightness, vig_target=VIG_TARGET_LEVEL,
                    vig_percentile=VIG_REFERENCE_PERCENTILE,
                    uint16_white_level=white_level or 65535,
                    top_margin=TOP_OFFSET_RATIO, bottom_margin=BOTTOM_OFFSET_RATIO,
                    alignment_interpolation='linear',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'preparation.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    counts, failures = Counter(), 0
    with (output / 'processing.jsonl').open('x', encoding='utf-8') as log:
        for number, (task, label, source, destination, context) in enumerate(destinations, 1):
            record = dict(task=task, label=label, source=str(source), csv_context=context,
                          destination=str(destination.relative_to(output)))
            try:
                result, metadata, overlay = prepare_one(source, task, brightness, white_level, debug)
                if overlay is not None:
                    write_png(output / '_debug' / task / label / destination.name, overlay)
                write_png(destination, result)
                record.update(status='ok', **metadata)
                counts[(task, label)] += 1
                if metadata['all_black']:
                    print(f'WARNING all-black image: {source}')
                print(f'[{number}/{len(destinations)}] {task}/{label}: {destination.name} '
                      f'{result.shape[1]}x{result.shape[0]} uint8')
            except (ValueError, OSError, cv2.error) as exc:
                failures += 1
                record.update(status='failed', error=str(exc))
                print(f'FAILED {source}: {exc}')
                review = output / '_review' / task / label / (destination.stem + source.suffix)
                try:
                    review.parent.mkdir(parents=True, exist_ok=True)
                    with source.open('rb') as src, review.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                    record['review_original'] = str(review.relative_to(output))
                except OSError as copy_error:
                    record['review_copy_error'] = str(copy_error)
            log.write(json.dumps(record, ensure_ascii=True) + '\n')
            log.flush()
    return counts, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', nargs='+', type=Path, default=CSV_FILES)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--tray', type=Path, default=TRAY_ROOT, help='Optional relocated tray root')
    parser.add_argument('--dry-run', action='store_true', help='Check CSV paths and labels only')
    parser.add_argument('--debug', action='store_true', help='Write native-size crop overlays')
    parser.add_argument('--no-vig-brightness', action='store_true', help='Disable gain/gamma; still export explicit uint8 PNG')
    parser.add_argument('--white-level', type=int, default=None, help='Known uint16 sensor white level (default 65535); values above it fail')
    args = parser.parse_args()
    try:
        if args.white_level is not None and not 1 <= args.white_level <= 65535:
            raise ValueError('--white-level must be 1..65535')
        if not 0 < VIG_TARGET_LEVEL < 1 or not 0 < VIG_REFERENCE_PERCENTILE <= 100:
            raise ValueError('Invalid VIG brightness configuration')
        if GOOD_TAGS & BAD_TAGS:
            raise ValueError('GOOD_TAGS and BAD_TAGS overlap')
        tray = args.tray.expanduser().resolve() if args.tray else None
        if tray is not None and not tray.is_dir():
            raise ValueError(f'Tray folder does not exist: {tray}')
        files = [p.expanduser().resolve() for p in args.csv]
        output = args.output.expanduser().resolve()
        plan, stats = build_plan(files, tray)
        failures = 0
        if args.dry_run:
            if not plan:
                raise ValueError('No labelled existing source images')
            counts = Counter((task, label) for task, label, _, _ in plan)
            print('DRY RUN: geometry and pixel conversion not evaluated. No files written.')
        else:
            counts, failures = export_plan(plan, output,
                VIG_BRIGHTNESS_ENABLED and not args.no_vig_brightness, args.white_level, args.debug)
            summary = dict(counts={f'{t}/{l}': n for (t, l), n in counts.items()},
                           failed_images=failures,
                           csv_issues={f'{t}/{k}': n for (t, k), n in stats.items()})
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        for task in TASK_COLUMNS:
            print(f"{task}: good={counts[task, 'good']} bad={counts[task, 'bad']}")
        omitted = sum(n for (task, kind), n in stats.items() if kind != 'duplicate_references')
        if failures or omitted:
            print(f'INCOMPLETE: {failures} processing failures; {omitted} CSV omissions. Review console and logs.')
            return 2
        print('Completed. Originals were not modified.')
        return 0
    except (OSError, UnicodeError, ValueError, csv.Error, cv2.error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
