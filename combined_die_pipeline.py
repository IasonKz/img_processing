"""Combined version of the three provided scripts.

Runs the same three stages in sequence:
  1. Detect die geometry and lens centre -> geometry.csv
  2. Perspective-warp each OK die -> unit_roi/
  3. Detect the label pad on each warped die -> roi_boxes.csv + label crops

No additional detection logic is introduced; the functions, thresholds and
measurements are the same as in the three source scripts.
"""

import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np


# ============================================================================
# SCRIPT 3: detect_lens_center.py
# ============================================================================

# --------------------------------------------------------------- die outline
def coarse_die(gray, ds=4):
    """Largest in-focus blob = the die. Returns x, y, w, h in full resolution."""
    s = cv2.resize(
        gray,
        None,
        fx=1 / ds,
        fy=1 / ds,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    hp = np.abs(s - cv2.GaussianBlur(s, (0, 0), 2.0))
    e = cv2.GaussianBlur(hp, (0, 0), 6.0)
    m = (e > max(np.percentile(e, 92), 1.5)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    if w * ds < 0.2 * gray.shape[1] or h * ds < 0.2 * gray.shape[0]:
        return None
    return int(x * ds), int(y * ds), int(w * ds), int(h * ds)


def _fit_line(pts, iters=200, tol=5.0, seed=0):
    """RANSAC + total-least-squares line fit. Returns (point, unit direction)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float64)
    best_inl, best_n = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        nrm = np.array([-d[1] / L, d[0] / L])
        inl = np.abs((pts - pts[i]) @ nrm) < tol
        if inl.sum() > best_n:
            best_inl, best_n = inl, int(inl.sum())
    P = pts[best_inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def refine_die_quad(gray, box, band=70, step=4, min_edge_bright=90):
    """Fit the four bright die edges. Returns corners TL, TR, BR, BL (float)."""
    x, y, w, h = box
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    H, W = gray.shape
    lines = {}

    for side in ("L", "R", "T", "B"):
        horiz = side in ("L", "R")
        start = {"L": x, "R": x + w, "T": y, "B": y + h}[side]
        B = band

        for it in range(3):
            pts = []
            along = (
                range(int(y + 0.10 * h), int(y + 0.90 * h), step)
                if horiz
                else range(int(x + 0.10 * w), int(x + 0.90 * w), step)
            )

            for a in along:
                if it == 0 or side not in lines:
                    c0 = start
                else:
                    p0, v = lines[side]
                    denom = v[1] if horiz else v[0]
                    if abs(denom) < 1e-9:
                        c0 = start
                    else:
                        t = (a - (p0[1] if horiz else p0[0])) / denom
                        c0 = p0[0] + t * v[0] if horiz else p0[1] + t * v[1]

                lo, hi = int(c0 - B), int(c0 + B)
                if horiz:
                    if lo < 0 or hi > W or not (0 <= a < H):
                        continue
                    prof = f[a, lo:hi]
                else:
                    if lo < 0 or hi > H or not (0 <= a < W):
                        continue
                    prof = f[lo:hi, a]

                if prof.size < 5:
                    continue
                k = int(np.argmax(prof))
                if prof[k] < min_edge_bright:
                    continue

                if 0 < k < prof.size - 1:  # parabolic sub-pixel peak
                    d2 = prof[k - 1] - 2 * prof[k] + prof[k + 1]
                    if abs(d2) > 1e-6:
                        k = k + 0.5 * (prof[k - 1] - prof[k + 1]) / d2

                c = lo + k
                pts.append((c, a) if horiz else (a, c))

            if len(pts) < 20:
                break
            lines[side] = _fit_line(pts)
            B = 25

    if len(lines) < 4:
        return None

    def inter(s1, s2):
        (p1, v1), (p2, v2) = lines[s1], lines[s2]
        A = np.array([v1, -v2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        t = np.linalg.solve(A, p2 - p1)
        return p1 + t[0] * v1

    corners = [
        inter("T", "L"),
        inter("T", "R"),
        inter("B", "R"),
        inter("B", "L"),
    ]
    if any(c is None for c in corners):
        return None
    return np.array(corners)


# --------------------------------------------------------------- lens centre
def lens_seed(bgr, quad):
    """Redness segmentation of the ring pattern -> (centroid, radius)."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    redness = cv2.GaussianBlur(r - 0.5 * (b + g), (0, 0), 3)
    m = (redness > 35).astype(np.uint8)
    die = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(die, quad.astype(np.int32), 1)
    m &= die
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (
        np.array(cent[i], np.float64),
        float(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)),
    )


def _center_ls(gray, c0, r_in, r_out, iters=4):
    """Least-squares intersection of the radial ring-gradient lines."""
    c = np.array(c0, np.float64)
    for _ in range(iters):
        R = int(r_out * 1.15) + 8
        x0, y0 = max(int(c[0]) - R, 0), max(int(c[1]) - R, 0)
        x1 = min(int(c[0]) + R, gray.shape[1])
        y1 = min(int(c[1]) + R, gray.shape[0])
        sub = gray[y0:y1, x0:x1].astype(np.float32)
        hp = sub - cv2.GaussianBlur(sub, (0, 0), 4.0)
        gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rr = np.hypot(xx - c[0], yy - c[1])
        sel = (rr > r_in) & (rr < r_out) & (mag > np.percentile(mag, 80))
        if sel.sum() < 500:
            return c

        px, py = xx[sel], yy[sel]
        ux, uy = gx[sel] / mag[sel], gy[sel] / mag[sel]
        w = mag[sel] ** 2
        a11 = np.sum(w * (1 - ux * ux))
        a12 = np.sum(w * (-ux * uy))
        a22 = np.sum(w * (1 - uy * uy))
        b1 = np.sum(w * ((1 - ux * ux) * px + (-ux * uy) * py))
        b2 = np.sum(w * ((-ux * uy) * px + (1 - uy * uy) * py))

        try:
            c_new = np.linalg.solve(
                np.array([[a11, a12], [a12, a22]]),
                np.array([b1, b2]),
            )
        except np.linalg.LinAlgError:
            return c

        step = float(np.hypot(*(c_new - c)))
        c = c_new
        if step < 0.05:
            break
    return c


def symmetry_cost(gray, c, r_in, r_out, n_ang=180):
    """Mean angular variance of the ring pattern about c (lower = better)."""
    radii = np.arange(r_in, r_out, 2.0)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    xs = (c[0] + np.outer(radii, np.cos(ang))).astype(np.float32)
    ys = (c[1] + np.outer(radii, np.sin(ang))).astype(np.float32)
    vals = cv2.remap(gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    return float(vals.var(axis=1).mean())


def _scan(gray, c0, r_in, r_out, span, step):
    offs = np.arange(-span, span + 1e-9, step)
    best_c, best_v = np.array(c0, np.float64), None
    for dy in offs:
        for dx in offs:
            c = np.array([c0[0] + dx, c0[1] + dy])
            v = symmetry_cost(gray, c, r_in, r_out)
            if best_v is None or v < best_v:
                best_c, best_v = c, v
    return best_c, best_v


def lens_center(gray, seed, R, span=12):
    """Ring centre by maximum rotational symmetry. Returns (centre, cost)."""
    r_in, r_out = 0.12 * R, 0.45 * R
    c, _ = _scan(gray, np.array(seed, np.float64), r_in, r_out, span, 2.0)
    c, _ = _scan(gray, c, r_in, r_out, 2.0, 0.5)
    return c, symmetry_cost(gray, c, r_in, r_out)


# --------------------------------------------------------------------- output
def draw_geometry_overlay(bgr, quad, center, R):
    ov = bgr.copy()
    q = quad.astype(np.int32)
    cv2.polylines(ov, [q], True, (0, 255, 0), 5, cv2.LINE_AA)
    for p in q:
        cv2.circle(ov, tuple(p), 16, (0, 255, 255), -1, cv2.LINE_AA)
    dc = quad.mean(0).astype(int)
    cv2.drawMarker(
        ov,
        tuple(dc),
        (0, 255, 255),
        cv2.MARKER_TILTED_CROSS,
        60,
        4,
    )
    c = np.round(center).astype(int)
    cv2.circle(ov, tuple(c), int(R), (255, 255, 0), 3, cv2.LINE_AA)
    cv2.line(
        ov,
        (c[0] - 130, c[1]),
        (c[0] + 130, c[1]),
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.line(
        ov,
        (c[0], c[1] - 130),
        (c[0], c[1] + 130),
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.circle(ov, tuple(c), 9, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        ov,
        tuple(dc),
        tuple(c),
        (255, 0, 255),
        4,
        cv2.LINE_AA,
        tipLength=0.15,
    )
    return ov


def process_geometry(path, debug_dir=None):
    bgr = cv2.imread(path)
    if bgr is None:
        return {"status": "UNREADABLE"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = coarse_die(gray)
    if box is None:
        return {"status": "NO_DIE"}

    quad = refine_die_quad(gray, box)
    if quad is None:
        return {"status": "NO_EDGES"}

    seed, R = lens_seed(bgr, quad)
    if seed is None:
        return {"status": "NO_LENS"}

    center, cost = lens_center(gray, seed, R)

    dc = quad.mean(0)
    sides = [
        float(np.hypot(*(quad[(i + 1) % 4] - quad[i])))
        for i in range(4)
    ]
    top = quad[1] - quad[0]
    tilt = float(np.degrees(np.arctan2(top[1], top[0])))
    off = center - dc

    res = {
        "status": "OK",
        "quad": quad,
        "center": center,
        "die_center": dc,
        "R": R,
        "tilt_deg": tilt,
        "sides": sides,
        "offset": off,
        "sym_cost": cost,
    }

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(
            os.path.join(debug_dir, name + "_geom.jpg"),
            draw_geometry_overlay(bgr, quad, center, R),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    return res


def run_geometry_stage(input_dir, pattern, out_csv, debug=False, debug_dir="debug_geom"):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise SystemExit(f"No files matching {pattern} in {input_dir}")

    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"rc(\d+)cc(\d+)", name, re.IGNORECASE)
        crow, ccol = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

        r = process_geometry(f, debug_dir if debug else None)
        row = {
            "file": os.path.basename(f),
            "container_row": crow,
            "container_col": ccol,
            "status": r["status"],
        }

        if r["status"] == "OK":
            q, c, dc = r["quad"], r["center"], r["die_center"]
            row.update(
                {
                    "lens_cx": round(float(c[0]), 1),
                    "lens_cy": round(float(c[1]), 1),
                    "die_cx": round(float(dc[0]), 1),
                    "die_cy": round(float(dc[1]), 1),
                    "offset_x": round(float(r["offset"][0]), 1),
                    "offset_y": round(float(r["offset"][1]), 1),
                    "offset_r": round(float(np.hypot(*r["offset"])), 1),
                    "die_side_px": round(float(np.mean(r["sides"])), 1),
                    "die_tilt_deg": round(r["tilt_deg"], 2),
                    "lens_radius_px": round(r["R"], 1),
                    "sym_cost": round(r["sym_cost"], 1),
                }
            )
            for i, k in enumerate(("tl", "tr", "br", "bl")):
                row[k + "_x"] = round(float(q[i][0]), 1)
                row[k + "_y"] = round(float(q[i][1]), 1)

        rows.append(row)
        print(
            f"{name}: {row['status']}"
            + (
                " lens=(%.1f, %.1f) decentre=%.1f px tilt=%.2f deg"
                % (
                    row["lens_cx"],
                    row["lens_cy"],
                    row["offset_r"],
                    row["die_tilt_deg"],
                )
                if row["status"] == "OK"
                else ""
            )
        )

    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: (
            k not in ("file", "container_row", "container_col", "status"),
            k,
        ),
    )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(
        f"\nWrote {out_csv} ({len(rows)} units)"
        + (f"; overlays in {debug_dir}/" if debug else "")
    )
    return rows


# ============================================================================
# SCRIPT 2: perspective rectification using geometry.csv
# ============================================================================

def run_warp_stage(input_dir, geometry_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    geometry_candidates = [
        os.path.join(os.getcwd(), geometry_path),
        os.path.join(input_dir, geometry_path),
        os.path.join(os.path.dirname(__file__), geometry_path),
    ]
    resolved_geometry_path = next(
        (p for p in geometry_candidates if os.path.exists(p)), None
    )
    if resolved_geometry_path is None:
        raise SystemExit(
            f"geometry.csv not found; looked in {os.getcwd()}, "
            f"{input_dir}, and {os.path.dirname(__file__)}"
        )

    with open(resolved_geometry_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    count = 0
    output_files = []

    for row in rows:
        if row.get("status") != "OK":
            continue

        image_name = row.get("file")
        if not image_name:
            continue

        image_path = os.path.join(input_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_name}")
            continue

        src_pts = np.float32(
            [
                [float(row["tl_x"]), float(row["tl_y"])],
                [float(row["tr_x"]), float(row["tr_y"])],
                [float(row["br_x"]), float(row["br_y"])],
                [float(row["bl_x"]), float(row["bl_y"])],
            ]
        )

        top = np.linalg.norm(src_pts[1] - src_pts[0])
        bottom = np.linalg.norm(src_pts[2] - src_pts[3])
        left = np.linalg.norm(src_pts[3] - src_pts[0])
        right = np.linalg.norm(src_pts[2] - src_pts[1])
        width = max(200, int(round((top + bottom) / 2.0)))
        height = max(200, int(round((left + right) / 2.0)))

        dst_pts = np.float32(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )

        out_path = os.path.join(out_dir, image_name)
        ok = cv2.imwrite(out_path, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            count += 1
            output_files.append(out_path)
            print(f"Saved {image_name} -> {out_path}")
        else:
            print(f"Failed to save {out_path}")

    print(f"\nSaved {count} unit ROI images to {out_dir}")
    return output_files


# ============================================================================
# SCRIPT 1: detect_label_roi.py
# ============================================================================

# canonical pose = empty corner bottom-left -> label pad top-right
_ROT_FOR_EMPTY = {
    "BL": None,
    "TL": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "TR": cv2.ROTATE_180,
    "BR": cv2.ROTATE_90_CLOCKWISE,
}
_ROT_DEG = {"BL": 0, "TL": 90, "TR": 180, "BR": 270}
_OPPOSITE = {"TL": "BR", "BR": "TL", "TR": "BL", "BL": "TR"}


def lens_mask(bgr, grow=41):
    """Mask of the red ring pattern, dilated to swallow its bright rim."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    red = cv2.GaussianBlur(r - 0.5 * (b + g), (0, 0), 3)
    m = (red > 30).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return m
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == i).astype(np.uint8)
    return cv2.dilate(m, np.ones((grow, grow), np.uint8))


def _diag_share(mask_patch, stride=10):
    """Share of the outline running near 45 deg (chamfer detector)."""
    cnts, _ = cv2.findContours(
        mask_patch.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(c) < stride + 2:
        return 0.0
    d = np.roll(c, -stride, axis=0) - c
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    return float(np.mean((ang > 25) & (ang < 65)))


def clean_bright_mask(bgr, bright_thresh=110, border_frac=0.005, span=0.70):
    """Bright structures with the lens and the die's own outer rim removed."""
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = ((gray > bright_thresh) & (lens_mask(bgr) == 0)).astype(np.uint8)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )

    b = max(int(border_frac * min(H, W)), 3)
    bright[:b, :] = 0
    bright[-b:, :] = 0
    bright[:, :b] = 0
    bright[:, -b:] = 0

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    for i in range(1, n):
        if (
            stats[i, cv2.CC_STAT_WIDTH] > span * W
            or stats[i, cv2.CC_STAT_HEIGHT] > span * H
        ):
            bright[lab == i] = 0
    return bright


def corner_shapes(bright, shape):
    """The three bright corner marks. Returns list of dicts, one per shape."""
    H, W = shape
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
    out = []

    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 0.0008 * W * H or w < 0.06 * W or h < 0.06 * H:
            continue

        cx, cy = cent[i]
        if not (
            min(cx, W - cx) < 0.35 * W
            and min(cy, H - cy) < 0.35 * H
        ):
            continue

        corner = ("T" if cy < H / 2 else "B") + (
            "L" if cx < W / 2 else "R"
        )
        out.append(
            {
                "corner": corner,
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": int(a),
                "diag": _diag_share((lab[y : y + h, x : x + w] == i)),
            }
        )

    return out


def refine_pad_rect(bright, bbox, pad_frac=0.30, min_peak=0.45):
    """Tighten the pad rectangle from projection profiles of the clean mask."""
    x, y, w, h = bbox
    mx, my = int(pad_frac * w), int(pad_frac * h)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1 = min(x + w + mx, bright.shape[1])
    y1 = min(y + h + my, bright.shape[0])

    m = bright[y0:y1, x0:x1].astype(np.float32)
    if m.size == 0:
        return bbox

    cs, rs = m.sum(0), m.sum(1)

    def edges(prof, lo_hint, hi_hint):
        if prof.max() <= 0:
            return lo_hint, hi_hint
        idx = np.nonzero(prof >= min_peak * prof.max())[0]
        if len(idx) < 2:
            return lo_hint, hi_hint
        return int(idx[0]), int(idx[-1])

    cx0, cx1 = edges(cs, x - x0, x + w - x0)
    cy0, cy1 = edges(rs, y - y0, y + h - y0)
    rw, rh = cx1 - cx0 + 1, cy1 - cy0 + 1

    if rw < 0.5 * w or rh < 0.5 * h or rw > 2 * w or rh > 2 * h:
        return bbox

    return (x0 + cx0, y0 + cy0, rw, rh)


def detect_label(bgr, margin=0.25, bright_thresh=110):
    """Returns dict with pad rect, ROI, corner roles and orientation."""
    H, W = bgr.shape[:2]
    bright = clean_bright_mask(bgr, bright_thresh)
    shapes = corner_shapes(bright, (H, W))

    if len(shapes) < 3:
        return {"status": "MARKS_NOT_FOUND", "n_shapes": len(shapes)}

    shapes = sorted(shapes, key=lambda s: -s["area"])[:3]
    pad = min(shapes, key=lambda s: s["diag"])
    fids = [s for s in shapes if s is not pad]

    if pad["diag"] > 0.15 or min(f["diag"] for f in fids) < 0.15:
        status = "LOW_CONFIDENCE"
    else:
        status = "OK"

    if len({s["corner"] for s in shapes}) < 3:
        status = "LOW_CONFIDENCE"

    rect = refine_pad_rect(bright, pad["bbox"])
    x, y, w, h = rect
    mx, my = int(round(margin * w)), int(round(margin * h))

    roi = (
        max(x - mx, 0),
        max(y - my, 0),
        min(x + w + mx, W) - max(x - mx, 0),
        min(y + h + my, H) - max(y - my, 0),
    )

    empty = _OPPOSITE[pad["corner"]]
    all_corners = {"TL", "TR", "BL", "BR"}
    empty_by_elimination = all_corners - {s["corner"] for s in shapes}

    if empty_by_elimination and empty not in empty_by_elimination:
        status = "LOW_CONFIDENCE"

    return {
        "status": status,
        "pad_rect": rect,
        "roi": roi,
        "pad_corner": pad["corner"],
        "empty_corner": empty,
        "fiducial_corners": sorted(f["corner"] for f in fids),
        "rot_code": _ROT_FOR_EMPTY[empty],
        "rot_deg": _ROT_DEG[empty],
        "pad_diag": pad["diag"],
        "fid_diag_min": min(f["diag"] for f in fids),
        "shapes": shapes,
    }


def upright_crop(bgr, res):
    """The ROI, rotated into the canonical pose (label reading upright)."""
    x, y, w, h = res["roi"]
    crop = bgr[y : y + h, x : x + w]
    return crop if res["rot_code"] is None else cv2.rotate(crop, res["rot_code"])


def draw_label_overlay(bgr, res):
    ov = bgr.copy()
    H, W = ov.shape[:2]

    for s in res["shapes"]:
        x, y, w, h = s["bbox"]
        is_pad = s["corner"] == res["pad_corner"]
        cv2.rectangle(
            ov,
            (x, y),
            (x + w, y + h),
            (0, 255, 0) if is_pad else (255, 255, 0),
            3,
        )
        cv2.putText(
            ov,
            "%s d=%.2f" % (s["corner"], s["diag"]),
            (x, max(y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if is_pad else (255, 255, 0),
            2,
        )

    x, y, w, h = res["roi"]
    cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 255), 6)
    cv2.putText(
        ov,
        "ROI",
        (x, max(y - 50, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 255),
        3,
    )

    ec = res["empty_corner"]
    ex = 40 if "L" in ec else W - 240
    ey = 60 if "T" in ec else H - 40
    cv2.putText(
        ov,
        "EMPTY",
        (ex, ey),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )
    return ov


def deskew_die(bgr, size=1231):
    """Crop and deskew the die from a full frame using local geometry functions."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = coarse_die(gray)
    if box is None:
        return bgr
    quad = refine_die_quad(gray, box)
    if quad is None:
        return bgr
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        np.float32,
    )
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, M, (size, size))


def run_label_stage(
    files,
    out_csv,
    crops_out,
    debug=False,
    debug_dir="debug_roi",
    margin=0.25,
    bright_thresh=110,
    size_tol=0.12,
    full_images=False,
    die_size=1231,
):
    os.makedirs(crops_out, exist_ok=True)
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    rows = []

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        m = re.search(r"rc(\d+)cc(\d+)", name, re.IGNORECASE)
        crow, ccol = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

        bgr = cv2.imread(f)
        if full_images:
            bgr = deskew_die(bgr, die_size)

        res = detect_label(
            bgr,
            margin=margin,
            bright_thresh=bright_thresh,
        )

        row = {
            "file": os.path.basename(f),
            "container_row": crow,
            "container_col": ccol,
            "status": res["status"],
        }

        if "roi" in res:
            px, py, pw, ph = res["pad_rect"]
            rx, ry, rw, rh = res["roi"]
            row.update(
                {
                    "pad_x": px,
                    "pad_y": py,
                    "pad_w": pw,
                    "pad_h": ph,
                    "roi_x": rx,
                    "roi_y": ry,
                    "roi_w": rw,
                    "roi_h": rh,
                    "pad_corner": res["pad_corner"],
                    "empty_corner": res["empty_corner"],
                    "rotation_deg_ccw": res["rot_deg"],
                    "pad_diag": round(res["pad_diag"], 3),
                    "fid_diag_min": round(res["fid_diag_min"], 3),
                }
            )

            cv2.imwrite(
                os.path.join(crops_out, name + "_label.png"),
                upright_crop(bgr, res),
            )

            if debug:
                cv2.imwrite(
                    os.path.join(debug_dir, name + "_roi.jpg"),
                    draw_label_overlay(bgr, res),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )

        rows.append(row)

        print(
            f"{name}: {row['status']}"
            + (
                " pad@%s roi=(%d,%d,%d,%d) rot=%d diag pad/fid=%.3f/%.3f"
                % (
                    row["pad_corner"],
                    row["roi_x"],
                    row["roi_y"],
                    row["roi_w"],
                    row["roi_h"],
                    row["rotation_deg_ccw"],
                    row["pad_diag"],
                    row["fid_diag_min"],
                )
                if "roi" in res
                else ""
            )
        )

    # Batch consistency: identical to the original script.
    canon = []
    for r in rows:
        if "pad_w" not in r:
            continue
        swap = r["rotation_deg_ccw"] in (90, 270)
        canon.append(
            (r["pad_h"], r["pad_w"])
            if swap
            else (r["pad_w"], r["pad_h"])
        )

    if canon:
        mw = float(np.median([c[0] for c in canon]))
        mh = float(np.median([c[1] for c in canon]))

        for r in rows:
            if "pad_w" not in r:
                continue
            swap = r["rotation_deg_ccw"] in (90, 270)
            cw, ch = (
                (r["pad_h"], r["pad_w"])
                if swap
                else (r["pad_w"], r["pad_h"])
            )
            dev = max(abs(cw - mw) / mw, abs(ch - mh) / mh)
            r["pad_size_dev"] = round(dev, 3)
            if dev > size_tol:
                r["status"] = "PAD_SIZE_OUTLIER"

        print("\nmedian canonical pad: %.0f x %.0f px" % (mw, mh))

    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: (
            k not in ("file", "container_row", "container_col", "status"),
            k,
        ),
    )

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    flagged = [r["file"] for r in rows if r["status"] != "OK"]
    if flagged:
        print("flagged for review: " + ", ".join(flagged))

    print(
        f"\nWrote {out_csv} ({len(rows)} units); crops in {crops_out}/"
        + (f", overlays in {debug_dir}/" if debug else "")
    )

    return rows


# ============================================================================
# COMBINED MAIN: stage 3 -> stage 2 -> stage 1
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="rc*cc*.jpg")

    ap.add_argument("--geometry", default="geometry.csv")
    ap.add_argument("--unit-roi-dir", default="unit_roi")

    ap.add_argument("--roi-out", default="roi_boxes.csv")
    ap.add_argument("--crops-out", default="roi_crops")

    ap.add_argument("--debug-geom-dir", default="debug_geom")
    ap.add_argument("--debug-roi-dir", default="debug_roi")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="ROI padding as a fraction of the pad rectangle",
    )
    ap.add_argument("--bright-thresh", type=int, default=110)
    ap.add_argument(
        "--full-images",
        action="store_true",
        help=(
            "run label detection on the original full microscope frames and "
            "deskew each die first, as in detect_label_roi.py"
        ),
    )
    ap.add_argument("--die-size", type=int, default=1231)
    ap.add_argument(
        "--size-tol",
        type=float,
        default=0.12,
        help="max canonical pad-size deviation from the batch median",
    )

    args = ap.parse_args()

    # Stage 1 of the combined pipeline: original script 3.
    run_geometry_stage(
        input_dir=args.input_dir,
        pattern=args.pattern,
        out_csv=args.geometry,
        debug=args.debug,
        debug_dir=args.debug_geom_dir,
    )

    # Stage 2 of the combined pipeline: original script 2.
    unit_files = run_warp_stage(
        input_dir=args.input_dir,
        geometry_path=args.geometry,
        out_dir=args.unit_roi_dir,
    )

    # Stage 3 of the combined pipeline: original script 1.
    if args.full_images:
        label_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    else:
        label_files = unit_files

    run_label_stage(
        files=label_files,
        out_csv=args.roi_out,
        crops_out=args.crops_out,
        debug=args.debug,
        debug_dir=args.debug_roi_dir,
        margin=args.margin,
        bright_thresh=args.bright_thresh,
        size_tol=args.size_tol,
        full_images=args.full_images,
        die_size=args.die_size,
    )


if __name__ == "__main__":
    main()
