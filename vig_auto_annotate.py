"""Experimental, offline annotation of standardized VIG camera images.

Edit INPUT_DIR below, then run:
    python vig_auto_annotate.py

The script never edits the input images. It writes copies with overlays to a
sibling folder named <input-folder>_annotated, plus annotation_results.csv.
Dependencies: Python 3.10+, opencv-python, numpy.

This is an experimental classical-computer-vision ensemble. It does not use
trained weights, and its confidence values are review cues, not calibrated
probabilities. Validate it on representative, manually checked tray images.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# USER SETTINGS: edit this one path, then run the script.
# ---------------------------------------------------------------------------
INPUT_DIR = r"C:\path\to\vig_images"  # e.g. r"D:\tray_01\vig"
OUTPUT_DIR = ""  # blank = create <input-folder>_annotated beside the input

# These defaults are deliberately conservative; adjust after reviewing a tray.
RECTIFIED_SIZE = 1000
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WRITE_CSV = True
OVERLAY_ALPHA = 0.95

# BGR annotation colors
C_DIE = (40, 220, 40)
C_CORNER = (255, 210, 0)
C_OCT = (220, 40, 230)
C_LABEL = (0, 220, 255)
C_CENTER = (0, 0, 255)
C_TEXT = (255, 255, 255)


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four points as TL, TR, BR, BL using their sum/difference."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]
    out = np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]], np.float32)
    # Reject degenerate or duplicate assignments.
    if len(np.unique(np.round(out, 1), axis=0)) != 4:
        return cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    return out


def coarse_die(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the main sharp rectangular component at reduced resolution."""
    h, w = gray.shape[:2]
    ds = max(2, min(5, int(round(min(h, w) / 500))))
    small = cv2.resize(gray, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA).astype(np.float32)
    high = np.abs(small - cv2.GaussianBlur(small, (0, 0), 2.0))
    energy = cv2.GaussianBlur(high, (0, 0), 5.0)
    threshold = max(float(np.percentile(energy, 91)), float(np.median(energy) + 2.0 * np.std(energy)))
    mask = (energy > threshold).astype(np.uint8)
    k = max(5, int(round(min(mask.shape) * 0.025)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    candidates = []
    sh, sw = small.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.18 * sw or bh < 0.18 * sh or area < 0.025 * sw * sh:
            continue
        ar = bw / max(bh, 1)
        if not 0.55 <= ar <= 1.8:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        center_penalty = math.hypot((cx - sw / 2) / sw, (cy - sh / 2) / sh)
        candidates.append((area * (1 - min(center_penalty, 0.8)), x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates)
    return int(x * ds), int(y * ds), int(bw * ds), int(bh * ds)


def _fit_line(points: Iterable[tuple[float, float]], tol: float = 4.0, seed: int = 0):
    pts = np.asarray(list(points), np.float64)
    if len(pts) < 12:
        return None
    rng = np.random.default_rng(seed)
    best = None
    best_n = 0
    for _ in range(min(240, max(80, len(pts)))):
        a, b = rng.choice(len(pts), 2, replace=False)
        v = pts[b] - pts[a]
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        normal = np.array([-v[1], v[0]]) / norm
        inliers = np.abs((pts - pts[a]) @ normal) <= tol
        ni = int(inliers.sum())
        if ni > best_n:
            best, best_n = inliers, ni
    if best is None or best_n < max(10, 0.25 * len(pts)):
        return None
    selected = pts[best]
    mean = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - mean, full_matrices=False)
    direction = vt[0]
    return mean, direction, best_n / len(pts)


def _intersect(l1, l2):
    (p1, v1, _), (p2, v2, _) = l1, l2
    mat = np.column_stack((v1, -v2))
    if abs(np.linalg.det(mat)) < 1e-8:
        return None
    t = np.linalg.solve(mat, p2 - p1)
    return p1 + t[0] * v1


def _refine_quad(gray: np.ndarray, box: tuple[int, int, int, int], gradient_mode: bool):
    """Fit four supported edges using either gradient or bright-edge peaks."""
    x, y, w, h = box
    H, W = gray.shape
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.3)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    band = max(18, int(0.075 * min(w, h)))
    step = max(3, int(min(w, h) / 300))
    sides = {}
    starts = {"L": x, "R": x + w, "T": y, "B": y + h}
    for side in ("L", "R", "T", "B"):
        vertical = side in ("L", "R")
        lo_along = int((y if vertical else x) + 0.10 * (h if vertical else w))
        hi_along = int((y if vertical else x) + 0.90 * (h if vertical else w))
        alongs = range(max(0, lo_along), min(H if vertical else W, hi_along), step)
        points = []
        # Search near the initial boundary; afterwards center each local search
        # on the fitted line to accommodate modest rotation/perspective.
        for a in alongs:
            if side not in sides:
                pred = starts[side]
            else:
                p, v, _ = sides[side]
                denom = v[1] if vertical else v[0]
                if abs(denom) < 1e-7:
                    pred = starts[side]
                else:
                    t = (a - (p[1] if vertical else p[0])) / denom
                    pred = p[0] + t * v[0] if vertical else p[1] + t * v[1]
            low, high = int(max(1, pred - band)), int(min((W if vertical else H) - 1, pred + band))
            if high <= low + 3:
                continue
            idx = int(a)
            if gradient_mode:
                profile = np.abs(gx[idx, low:high]) if vertical else np.abs(gy[low:high, idx])
            else:
                profile = smooth[idx, low:high] if vertical else smooth[low:high, idx]
            if profile.size < 4:
                continue
            k = int(np.argmax(profile))
            strength = float(profile[k])
            if gradient_mode:
                # Ignore isolated texture peaks; RANSAC will retain a straight edge.
                local = float(np.percentile(profile, 60))
                if strength < max(4.0, local * 1.25):
                    continue
            else:
                if strength < max(70.0, float(np.percentile(profile, 65))):
                    continue
            coord = low + k
            if 0 < k < len(profile) - 1:
                curvature = float(profile[k - 1] - 2 * profile[k] + profile[k + 1])
                if abs(curvature) > 1e-6:
                    coord += float(0.5 * (profile[k - 1] - profile[k + 1]) / curvature)
            points.append((coord, a) if vertical else (a, coord))
        line = _fit_line(points, tol=max(3.0, 0.004 * min(w, h)), seed=ord(side))
        if line is None:
            return None
        sides[side] = line
    corners = [_intersect(sides["T"], sides["L"]), _intersect(sides["T"], sides["R"]),
               _intersect(sides["B"], sides["R"]), _intersect(sides["B"], sides["L"])]
    if any(p is None for p in corners):
        return None
    q = np.asarray(corners, np.float32)
    q = _order_quad(q)
    lengths = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    if min(lengths) < 0.18 * min(H, W) or max(lengths) > 1.25 * max(H, W):
        return None
    if not cv2.isContourConvex(q.astype(np.int32)):
        return None
    support = float(np.mean([v[2] for v in sides.values()]))
    return q, support


def _contour_quad(gray: np.ndarray):
    """Independent fallback: seek a large, near-square four-sided contour."""
    H, W = gray.shape
    work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    best = None
    for lo, hi in ((25, 80), (45, 130), (70, 180)):
        edges = cv2.Canny(work, lo, hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = abs(cv2.contourArea(c))
            if area < 0.12 * H * W:
                continue
            p = cv2.arcLength(c, True)
            for eps in (0.01, 0.02, 0.035, 0.05):
                approx = cv2.approxPolyDP(c, eps * p, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                q = _order_quad(approx[:, 0, :])
                lens = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
                ar1 = (lens[0] + lens[2]) / max(lens[1] + lens[3], 1)
                if not 0.58 <= ar1 <= 1.72:
                    continue
                cx, cy = q.mean(axis=0)
                center_penalty = math.hypot((cx - W / 2) / W, (cy - H / 2) / H)
                score = area / (H * W) - 0.35 * center_penalty - 0.12 * abs(math.log(ar1))
                if best is None or score > best[0]:
                    best = score, q
    return (best[1], 0.42) if best else None


def detect_die(bgr: np.ndarray):
    gray = _gray(bgr)
    box = coarse_die(gray)
    attempts = []
    if box:
        for gradient in (True, False):
            got = _refine_quad(gray, box, gradient_mode=gradient)
            if got:
                quad, support = got
                side = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
                ratio = (side[0] + side[2]) / max(side[1] + side[3], 1)
                center = quad.mean(axis=0)
                center_score = max(0.0, 1.0 - math.hypot((center[0] - gray.shape[1] / 2) / gray.shape[1],
                                                            (center[1] - gray.shape[0] / 2) / gray.shape[0]))
                score = 0.55 * support + 0.25 * math.exp(-abs(math.log(ratio))) + 0.20 * center_score
                attempts.append((score, quad, "gradient_edges" if gradient else "bright_edges", support))
    fallback = _contour_quad(gray)
    if fallback:
        attempts.append((fallback[1], fallback[0], "contour_fallback", fallback[1]))
    if not attempts:
        return None
    score, quad, method, support = max(attempts, key=lambda z: z[0])
    if score < 0.30:
        return None
    return {"quad": quad, "confidence": float(min(0.99, score)), "method": method, "support": support}


def _warp_die(bgr: np.ndarray, quad: np.ndarray, size: int = RECTIFIED_SIZE):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, M, (size, size), flags=cv2.INTER_CUBIC), M


def _red_mask(bgr: np.ndarray):
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    redness = cv2.GaussianBlur(r - 0.5 * (b + g), (0, 0), 2.5)
    h, w = redness.shape
    central = redness[int(.20*h):int(.80*h), int(.20*w):int(.80*w)]
    threshold = max(20.0, float(np.percentile(central, 84)))
    mask = (redness > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return redness, mask


def _symmetry_cost(gray: np.ndarray, center: np.ndarray, radius: float):
    h, w = gray.shape
    rin, rout = max(5, 0.10 * radius), min(0.50 * radius, 0.25 * min(h, w))
    if rout <= rin + 3:
        return float("inf")
    radii = np.arange(rin, rout, max(2.5, radius / 24), dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False, dtype=np.float32)
    xs = (center[0] + np.outer(radii, np.cos(angles))).astype(np.float32)
    ys = (center[1] + np.outer(radii, np.sin(angles))).astype(np.float32)
    values = cv2.remap(gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # A centered ring has lower angular variation at each radius.
    return float(np.median(values.var(axis=1)))


def detect_center(bgr: np.ndarray):
    redness, mask = _red_mask(bgr)
    h, w = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        if area < 0.0007 * w * h or not (0.22*w < cx < 0.78*w and 0.22*h < cy < 0.78*h):
            continue
        rad = math.sqrt(area / math.pi)
        if not 0.025 * w < rad < 0.30 * w:
            continue
        component = labels == i
        red_score = float(np.mean(redness[component]))
        central_score = 1.0 - math.hypot((cx-w/2)/w, (cy-h/2)/h)
        candidates.append((area * (1 + max(0, red_score)/80) * central_score, np.array([cx, cy]), rad))
    if not candidates:
        return {"center": np.array([w/2, h/2], np.float32), "radius": .18*w, "confidence": .16,
                "method": "die_center_fallback"}
    _, seed, radius = max(candidates, key=lambda z: z[0])
    gray = cv2.GaussianBlur(_gray(bgr), (0, 0), 1.2)
    # Small robust symmetry search refines a seed even if part of the red ring is damaged.
    candidates2 = []
    span = max(4, int(.018*w))
    for step, off in ((2.0, span), (0.8, 2.5)):
        vals = np.arange(-off, off + 0.01, step)
        candidates2 = []
        for dy in vals:
            for dx in vals:
                c = seed + np.array([dx, dy], np.float32)
                if 0 < c[0] < w and 0 < c[1] < h:
                    candidates2.append((_symmetry_cost(gray, c, radius), c))
        if candidates2:
            seed = min(candidates2, key=lambda z: z[0])[1]
    quality = _symmetry_cost(gray, seed, radius)
    # Normalize a within-image comparison proxy; never interpreted as probability.
    confidence = float(np.clip(0.52 + 0.16 * min(1.0, radius/(.12*w)), .45, .78))
    return {"center": seed.astype(np.float32), "radius": float(radius), "confidence": confidence,
            "method": "red_pattern+radial_symmetry", "symmetry_cost": quality}


def _poly_quality(poly: np.ndarray, center: np.ndarray, W: int, H: int):
    p = poly.reshape(-1, 2).astype(np.float32)
    area = abs(cv2.contourArea(p))
    if len(p) < 6 or len(p) > 10 or area < .02*W*H or area > .62*W*H:
        return None
    x, y, w, h = cv2.boundingRect(p)
    if min(w, h) < .16 * min(W, H) or max(w, h) > .88 * min(W, H):
        return None
    if not cv2.isContourConvex(p.astype(np.int32)):
        return None
    c = p.mean(axis=0)
    dist = np.linalg.norm(c - center)
    if dist > .18 * min(W, H):
        return None
    r = np.linalg.norm(p-c, axis=1)
    radial_cv = float(np.std(r)/(np.mean(r)+1e-5))
    radius = max(w,h)/2.0
    # The camera insert is a die-scale octagon, while concentric lens rings
    # produce many smaller polygon approximations. Prefer the larger standard
    # insert outline and leave its exact size to batch consensus.
    target_radius = .32 * min(W,H)
    size_score = math.exp(-abs(radius-target_radius)/(.17*min(W,H)))
    n_score = 1.0 - abs(len(p)-8)/4.0
    center_score = max(0.0, 1.0 - dist/(.18*min(W,H)))
    square_score = math.exp(-abs(math.log(max(w,1)/max(h,1))))
    radial_score = max(0.0, 1.0-radial_cv/.35)
    return float(.26*n_score + .19*center_score + .15*square_score + .10*radial_score + .30*size_score), c, radius


def detect_octagon(bgr: np.ndarray, center: np.ndarray):
    """Contour ensemble over contrast and threshold variants."""
    gray = _gray(bgr)
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants = []
    for src in (gray, clahe):
        blur = cv2.GaussianBlur(src, (5, 5), 0.8)
        for lo, hi in ((20, 65), (35, 105), (55, 150)):
            e = cv2.Canny(blur, lo, hi)
            variants.append(cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)))
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        variants.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)))
        variants.append(cv2.morphologyEx(255-otsu, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)))
    _, red = _red_mask(bgr)
    variants.append(cv2.morphologyEx(red*255, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8)))
    best = None
    for mask in variants:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < .25 * min(w,h):
                continue
            for eps in (.008, .012, .018, .025, .035, .05):
                approx = cv2.approxPolyDP(contour, eps*perimeter, True)
                result = _poly_quality(approx, center, w, h)
                if result is None:
                    continue
                score, c, radius = result
                if best is None or score > best[0]:
                    best = (score, approx[:,0,:].astype(np.float32), c.astype(np.float32), float(radius))
    if best is None:
        return None
    return {"polygon": best[1], "center": best[2], "radius": best[3], "confidence": best[0],
            "method": "multithreshold_contours"}


def _diag_share(mask_patch: np.ndarray):
    contours, _ = cv2.findContours(mask_patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)[:,0,:].astype(np.float64)
    if len(c) < 12:
        return 0.0
    stride = max(4, len(c)//40)
    d = np.roll(c, -stride, axis=0) - c
    ang = np.degrees(np.arctan2(d[:,1],d[:,0])) % 90.0
    return float(np.mean((ang > 24) & (ang < 66)))


def detect_label_and_corners(bgr: np.ndarray, exclude_polygon: np.ndarray | None = None):
    """Find three standardized bright marks; the less diagonal pad is the label."""
    h, w = bgr.shape[:2]
    gray = _gray(bgr)
    redness, red = _red_mask(bgr)
    # Use locally enhanced intensity and a high floor. A low global threshold
    # can merge the whole die background into one giant component and swallow
    # all three marks.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    thresh = int(np.clip(max(145, np.percentile(enhanced, 82)), 145, 220))
    bright = ((enhanced >= thresh) & (red == 0)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    if exclude_polygon is not None:
        # The bright inner-camera outline can touch a corner label/fiducial.
        # Remove a narrow band of that outline before connected-component
        # analysis so the regions remain separable.
        exclusion = np.zeros_like(bright)
        cv2.fillPoly(exclusion, [np.asarray(exclude_polygon, np.int32).reshape(-1,1,2)], 1)
        exclusion = cv2.dilate(exclusion, np.ones((max(9,int(.025*min(h,w))) | 1,)*2,np.uint8))
        bright[exclusion > 0] = 0
    border = max(3, int(.008*min(h,w)))
    bright[:border] = 0; bright[-border:] = 0; bright[:,:border] = 0; bright[:,-border:] = 0
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    comps = []
    for i in range(1,n):
        x,y,bw,bh,area = map(int, stats[i])
        cx,cy = centroids[i]
        if area < .00065*w*h or bw < .035*w or bh < .035*h or bw > .30*w or bh > .30*h:
            continue
        if not (min(cx,w-cx)<.39*w and min(cy,h-cy)<.39*h):
            continue
        corner = ("T" if cy<h/2 else "B") + ("L" if cx<w/2 else "R")
        patch = (labels[y:y+bh,x:x+bw] == i).astype(np.uint8)
        diag = _diag_share(patch)
        # Prefer substantial bright objects in outer corner zones.
        score = area * (1 - min(abs(cx-w/2)/(w/2),1)*.18) * (1 - min(abs(cy-h/2)/(h/2),1)*.18)
        comps.append({"corner":corner,"bbox":(x,y,bw,bh),"area":area,"diag":diag,"score":score})
    # Retain at most one candidate per corner; keep the three strongest corners.
    by_corner = {}
    for c in comps:
        if c["corner"] not in by_corner or c["score"] > by_corner[c["corner"]]["score"]:
            by_corner[c["corner"]] = c
    chosen = sorted(by_corner.values(), key=lambda c:c["score"], reverse=True)[:3]
    if len(chosen) < 3:
        return {"marks":chosen,"label":None,"confidence":0.0,"method":"bright_components"}
    pad = min(chosen, key=lambda c:c["diag"])
    fids = [c for c in chosen if c is not pad]
    confidence = .48
    if pad["diag"] < .20 and min(x["diag"] for x in fids) > .08:
        confidence += .22
    if len({x["corner"] for x in chosen}) == 3:
        confidence += .18
    if pad["area"] > .0012*w*h:
        confidence += .08
    return {"marks":chosen,"label":pad,"confidence":float(min(confidence,.96)),"method":"three_mark_geometry"}


def _make_octagon(center: np.ndarray, radius: float, W: int, H: int):
    # An 8-vertex chamfered square; standard pose is established by die rectification.
    r = max(5.0, float(radius))
    a = .70*r
    pts = np.array([[-a,-r],[a,-r],[r,-a],[r,a],[a,r],[-a,r],[-r,a],[-r,-a]],np.float32)
    pts += np.asarray(center,np.float32)
    pts[:,0] = np.clip(pts[:,0],0,W-1); pts[:,1] = np.clip(pts[:,1],0,H-1)
    return pts


def _median_octagon_prior(records: list[dict[str,Any]]):
    good = [r["octagon"] for r in records if r.get("octagon") and r["octagon"]["confidence"] >= .46]
    if len(good) < 2:
        return None
    centers = np.array([x["center"] for x in good],np.float32)
    radii = np.array([x["radius"] for x in good],np.float32)
    center = np.median(centers,axis=0)
    radius = float(np.median(radii))
    return {"center":center,"radius":radius,"n":len(good)}


def _median_label_prior(records: list[dict[str,Any]]):
    good = [r["label_result"]["label"]["bbox"] for r in records
            if r.get("label_result") and r["label_result"].get("label") and r["label_result"]["confidence"] >= .65]
    if len(good) < 2:
        return None
    arr = np.array(good,np.float32)
    centers = arr[:, :2] + arr[:, 2:] / 2
    med_center = np.median(centers,axis=0)
    # Do not blend different camera orientations into a fictitious middle
    # location. In that case only the per-image label detections are used.
    deviations = np.linalg.norm(centers-med_center,axis=1)
    if float(np.median(deviations)) > .06*RECTIFIED_SIZE:
        return None
    return tuple(np.round(np.median(arr,axis=0)).astype(int).tolist()), len(good)


def _transform(points: np.ndarray, M: np.ndarray):
    p = np.asarray(points,np.float32).reshape(-1,1,2)
    return cv2.perspectiveTransform(p,M).reshape(-1,2)


def _draw_text(img, text, org, color, scale):
    x,y = int(org[0]),int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1,int(scale*2.2))
    (tw,th),base = cv2.getTextSize(text,font,scale,thickness)
    x=max(0,min(x,img.shape[1]-tw-4)); y=max(th+base,min(y,img.shape[0]-2))
    cv2.rectangle(img,(x-2,y-th-base-3),(x+tw+3,y+base),(15,15,15),-1)
    cv2.putText(img,text,(x,y),font,scale,color,thickness,cv2.LINE_AA)


def _draw_mapped_poly(img, points, M_inv, color, thickness=3):
    q = np.round(_transform(np.asarray(points,np.float32),M_inv)).astype(np.int32)
    cv2.polylines(img,[q.reshape(-1,1,2)],True,color,thickness,cv2.LINE_AA)
    return q


def annotate_record(path: Path, record: dict[str,Any], oct_prior, label_prior, output_path: Path):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rec=record
    quad=rec.get("quad")
    if quad is None:
        overlay=bgr.copy()
        scale=max(.55,min(1.2,min(bgr.shape[:2])/900))
        _draw_text(overlay,"NO DIE GEOMETRY - REVIEW",(12,35),(0,0,255),scale)
        return cv2.imwrite(str(output_path),overlay,[cv2.IMWRITE_JPEG_QUALITY,94])
    warped,M=_warp_die(bgr,quad)
    inv=np.linalg.inv(M)
    overlay=bgr.copy()
    scale=max(.55,min(1.25,min(bgr.shape[:2])/900))
    dieq=np.round(quad).astype(np.int32)
    cv2.polylines(overlay,[dieq.reshape(-1,1,2)],True,C_DIE,max(2,int(scale*3)),cv2.LINE_AA)
    _draw_text(overlay,"DIE  %.2f"%rec["die_confidence"],dieq[0],C_DIE,scale)

    octa=rec.get("octagon")
    if octa:
        polygon=octa.get("polygon")
        if polygon is None:
            polygon=_make_octagon(octa["center"],octa["radius"],RECTIFIED_SIZE,RECTIFIED_SIZE)
        pts=_draw_mapped_poly(overlay,polygon,inv,C_OCT,max(2,int(scale*3)))
        origin=pts.mean(axis=0)
        tag="OCTAGON" if not octa.get("inferred") else "OCTAGON* prior"
        _draw_text(overlay,tag,origin+np.array([8,-8]),C_OCT,scale)
    center=rec.get("center")
    if center is not None:
        c=_transform(np.array([center]),inv)[0]
        p=tuple(np.round(c).astype(int))
        r=max(8,int(scale*15))
        cv2.drawMarker(overlay,p,C_CENTER,cv2.MARKER_CROSS,r,max(2,int(scale*3)),cv2.LINE_AA)
        cv2.circle(overlay,p,max(4,int(scale*5)),C_CENTER,-1,cv2.LINE_AA)
        _draw_text(overlay,"CENTER* prior" if rec.get("center_inferred") else "CENTER",
                   (p[0]+r,p[1]-r),C_CENTER,scale)

    lr=rec.get("label_result") or {}
    marks=lr.get("marks",[])
    label=lr.get("label")
    for mark in marks:
        x,y,w,h=mark["bbox"]
        rect=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
        pts=_draw_mapped_poly(overlay,rect,inv,C_LABEL if mark is label else C_CORNER,max(2,int(scale*3)))
        label_txt="LABEL" if mark is label else f"CORNER {mark['corner']}"
        _draw_text(overlay,label_txt,pts[0]+np.array([0,-5]),C_LABEL if mark is label else C_CORNER,scale)
    if label is None and label_prior is not None:
        (x,y,w,h),nprior=label_prior
        rect=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
        pts=_draw_mapped_poly(overlay,rect,inv,C_LABEL,max(2,int(scale*3)))
        # Dashed overlay signals that this box came from other images in the batch.
        # Add a clear asterisk in the tag; CSV also records the source.
        _draw_text(overlay,"LABEL* batch prior",pts[0]+np.array([0,-5]),C_LABEL,scale)
        rec["label_inferred"]=True

    quality=rec.get("status","REVIEW")
    _draw_text(overlay,quality,(12, max(28,int(.05*overlay.shape[0]))), C_DIE if quality=="OK" else (0,165,255),scale)
    return cv2.imwrite(str(output_path),overlay,[cv2.IMWRITE_JPEG_QUALITY,94])


def process_one(path: Path):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None:
        return {"path":path,"status":"UNREADABLE","quad":None}
    die=detect_die(bgr)
    if not die:
        return {"path":path,"status":"NO_DIE","quad":None}
    warped,_=_warp_die(bgr,die["quad"])
    center_res=detect_center(warped)
    octa=detect_octagon(warped,center_res["center"])
    labels=detect_label_and_corners(warped, octa.get("polygon") if octa else None)
    # Protect against implausible contour: batch consensus may later provide a safer estimate.
    if octa and octa["confidence"] < .43:
        octa=None
    status="OK"
    if die["confidence"]<.50 or center_res["confidence"]<.45 or octa is None or labels.get("label") is None:
        status="REVIEW"
    return {"path":path,"status":status,"quad":die["quad"],"die_confidence":die["confidence"],
            "die_method":die["method"],"edge_support":die["support"],"center":center_res["center"],
            "center_radius":center_res["radius"],"center_confidence":center_res["confidence"],
            "center_method":center_res["method"],"octagon":octa,"label_result":labels}


def _resolve_batch_priors(records):
    center_records = [r for r in records if r.get("center") is not None and r.get("center_confidence",0) >= .45]
    center_prior = None
    if len(center_records) >= 2:
        center_prior = np.median(np.array([r["center"] for r in center_records],np.float32),axis=0)
        for r in records:
            if r.get("quad") is None or r.get("center") is None:
                continue
            if r.get("center_confidence",0) < .45 or np.linalg.norm(r["center"]-center_prior) > .08*RECTIFIED_SIZE:
                r["center"] = center_prior.copy()
                r["center_inferred"] = True
                r["center_method"] = "batch_median_prior"
                r["center_confidence"] = min(float(r.get("center_confidence",0)),.40)
    oct_prior=_median_octagon_prior(records)
    label_prior=_median_label_prior(records)
    if oct_prior:
        for r in records:
            octa=r.get("octagon")
            use_prior=octa is None
            if octa is not None:
                dist=np.linalg.norm(octa["center"]-oct_prior["center"])
                size_dev=abs(octa["radius"]-oct_prior["radius"])/max(oct_prior["radius"],1)
                use_prior=(dist>.065*RECTIFIED_SIZE or size_dev>.22)
            if use_prior:
                c=oct_prior["center"].copy(); rad=oct_prior["radius"]
                # Retain each image's independently estimated center when available.
                if r.get("center") is not None and r.get("center_confidence",0)>.45:
                    c=.65*r["center"]+.35*c
                r["octagon"]={"polygon":None,"center":c,"radius":rad,"confidence":.42,
                              "method":"batch_median_prior","inferred":True}
    if label_prior:
        prior_box,n=label_prior
        for r in records:
            lr=r.get("label_result") or {}
            label=lr.get("label")
            use_prior=label is None or lr.get("confidence",0)<.60
            if label is not None and not use_prior:
                x,y,w,h=label["bbox"]
                px,py,pw,ph=prior_box
                dist=math.hypot((x+w/2)-(px+pw/2),(y+h/2)-(py+ph/2))
                use_prior=dist>.07*RECTIFIED_SIZE or max(abs(w-pw)/max(pw,1),abs(h-ph)/max(ph,1))>.30
            if use_prior:
                r["label_inferred"]=True
                if r.get("label_result") is None:
                    r["label_result"]={"marks":[],"label":None,"confidence":0.0,"method":"batch_median_prior"}
                else:
                    r["label_result"]["label"]=None
                    r["label_result"]["method"]="batch_median_prior"
    for r in records:
        if r.get("quad") is None:
            continue
        if r.get("octagon") is None or (r.get("label_result") or {}).get("label") is None:
            r["status"]="REVIEW"
        elif (r["die_confidence"]>=.50 and
              (r.get("center_confidence",0)>=.45 or r.get("center_inferred"))):
            inferred = (r.get("label_inferred") or r.get("center_inferred") or
                        r["octagon"].get("inferred"))
            r["status"]="REVIEW*" if inferred else "OK"
        else:
            r["status"]="REVIEW"
    return oct_prior,label_prior


def _serial_poly(value):
    if value is None: return ""
    return ";".join("%.1f,%.1f"%(p[0],p[1]) for p in np.asarray(value).reshape(-1,2))


def run(input_dir: str, output_dir: str = ""):
    src=Path(input_dir).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Input folder does not exist: {src}\nEdit INPUT_DIR at the top of the script.")
    out=Path(output_dir).expanduser().resolve() if output_dir.strip() else src.with_name(src.name+"_annotated")
    files=sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                 and out not in p.parents)
    if not files:
        raise SystemExit(f"No supported images found in: {src}")
    out.mkdir(parents=True,exist_ok=True)
    print(f"Input:  {src}\nOutput: {out}\nImages: {len(files)}\n")
    records=[]
    for i,path in enumerate(files,1):
        try:
            rec=process_one(path)
        except Exception as exc:
            rec={"path":path,"status":"ERROR","quad":None,"error":str(exc)}
        records.append(rec)
        print(f"[{i:>4}/{len(files)}] {path.name}: {rec['status']}" +
              (f" | die {rec.get('die_confidence',0):.2f}, center {rec.get('center_confidence',0):.2f}, "
               f"oct {'yes' if rec.get('octagon') else 'no'}, label {'yes' if (rec.get('label_result') or {}).get('label') else 'no'}" if rec.get('quad') is not None else ""))
    oct_prior,label_prior=_resolve_batch_priors(records)
    print("\nBatch priors: octagon="+(f"{oct_prior['n']} images" if oct_prior else "not available")+
          ", label="+(f"{label_prior[1]} images" if label_prior else "not available"))
    rows=[]
    for rec in records:
        path=rec["path"]
        rel=path.relative_to(src)
        target=out/rel.parent/(path.stem+"_annotated.jpg")
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            saved=annotate_record(path,rec,oct_prior,label_prior,target)
            if not saved:
                rec["status"]="WRITE_ERROR"
        except Exception as exc:
            rec["status"]="ERROR"
            rec["error"]=str(exc)
        lr=rec.get("label_result") or {}
        octa=rec.get("octagon") or {}
        center=rec.get("center")
        quad=rec.get("quad")
        rows.append({
            "file":str(rel),"annotated_file":str(target.relative_to(out)),"status":rec.get("status"),
            "die_method":rec.get("die_method",""),"die_confidence":round(float(rec.get("die_confidence",0)),3),
            "edge_support":round(float(rec.get("edge_support",0)),3),"center_method":rec.get("center_method",""),
            "center_inferred_from_batch":bool(rec.get("center_inferred",False)),
            "center_confidence":round(float(rec.get("center_confidence",0)),3),
            "center_x_rectified":round(float(center[0]),1) if center is not None else "",
            "center_y_rectified":round(float(center[1]),1) if center is not None else "",
            "octagon_method":octa.get("method",""),"octagon_confidence":round(float(octa.get("confidence",0)),3),
            "octagon_inferred_from_batch":bool(octa.get("inferred",False)),
            "octagon_polygon_rectified":_serial_poly(octa.get("polygon")),
            "label_method":lr.get("method",""),"label_confidence":round(float(lr.get("confidence",0)),3),
            "label_inferred_from_batch":bool(rec.get("label_inferred",False)),
            "label_corner":(lr.get("label") or {}).get("corner",""),
            "label_bbox_rectified":(lr.get("label") or {}).get("bbox",label_prior[0] if label_prior and rec.get("label_inferred") else ""),
            "corner_marks_found":sum(1 for x in lr.get("marks",[]) if x is not lr.get("label")),
            "die_quad_xy":_serial_poly(quad),"error":rec.get("error","")
        })
    if WRITE_CSV:
        csv_path=out/"annotation_results.csv"
        fields=list(rows[0].keys())
        with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        print(f"\nCSV: {csv_path}")
    counts={s:sum(r.get("status")==s for r in records) for s in sorted(set(r.get("status") for r in records))}
    print(f"Done. Annotated images: {len(records)} | statuses: {counts}")
    print("Images tagged REVIEW/REVIEW* need manual inspection. A '*' means at least one region used the batch median prior.")
    return out


if __name__ == "__main__":
    run(INPUT_DIR, OUTPUT_DIR)
