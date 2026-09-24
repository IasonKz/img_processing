"""Copy manually labelled tray images into top/bottom/vig -> good/bad.

Python 3.10+; fully offline processing.
Vig brightness needs numpy and OpenCV: python -m pip install numpy opencv-python
Use --no-vig-brightness for the original standard-library-only copy mode.
Edit CONFIGURATION below, or run:
    python good_bad_folders.py --csv PATH_TO_CSV --output NEW_OUTPUT_FOLDER
Several CSVs (e.g. top/bottom and vig) can follow --csv.

The two default CSV paths and all task columns match the supplied photographs.
The files are processed independently, without joining rows by unit or position.
The top/bottom f/x rule is retained from the original photographed script.
Vig uses the four checks shown in the CSV: Particles, PnP, Wave and Others.
Any f/x makes vig bad; all four must be p to make vig good.
The explicit pass tag "p" is configurable below.
"""

import argparse
import ast
import csv
import hashlib
import math
import os
import shutil
from collections import Counter
from pathlib import Path, PureWindowsPath


# CONFIGURATION: change these paths before running with the IDE's Run button.
CSV_FILES = [
    Path(r"C:\Users\laka\software\data\C00000094\EvaluationResults\c94_vi_top_bot\evaluation_results.csv"),
    Path(r"C:\Users\laka\software\data\C00000094\EvaluationResults\c94_vi_vig\evaluation_results.csv"),
]
OUTPUT_ROOT = Path(r"C:\Users\laka\software\good_bad_split_94_bright")

# Optional: current tray folder, if the CSV still contains paths from another
# computer. The full suffix after the matching tray name must still exist.
# Example: Path(r"D:\data\C00000094") or Path("/home/iason/data/C00000094")
TRAY_ROOT = None

GOOD_TAGS = {"p"}
BAD_TAGS = {"f", "x"}  # x counts as bad, matching the photographed script.
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"

# Only vig is brightened. No crop, resize, denoising or sharpening is applied.
# Vig outputs are losslessly encoded PNG, retaining uint8/uint16 bit depth.
# Top/bottom and all source files stay byte-for-byte unchanged.
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


def image_dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValueError(
            "Vig brightness needs numpy and OpenCV. Run: "
            "python -m pip install numpy opencv-python\n"
            "Or use --no-vig-brightness for unchanged copies."
        ) from exc
    return cv2, np


def brighten_vig(source):
    """Global monotonic gain/gamma; preserve spatial shape, dtype and alpha.

    Gain is limited by the brightest input sample to avoid hard clipping.
    Gamma then lifts the reference percentile towards the target, if needed.
    The transformation changes intensity values; it is not radiometric data.
    """
    cv2, np = image_dependencies()
    if not 0 < VIG_TARGET_LEVEL < 1 or not 0 < VIG_REFERENCE_PERCENTILE <= 100:
        raise ValueError("Invalid VIG_TARGET_LEVEL or VIG_REFERENCE_PERCENTILE.")
    try:
        if cv2.imcount(str(source)) > 1:
            raise ValueError(f"Multi-frame vig image is unsupported: {source}")
        image = cv2.imdecode(np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f"Cannot decode vig image {source}: {exc}") from exc
    if image is None:
        raise ValueError(f"Cannot decode vig image: {source}")
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError(
            f"Vig image {source} is {image.dtype}; only uint8/uint16 are supported. "
            "No automatic bit-depth conversion is performed."
        )
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError(f"Unsupported vig image shape {image.shape}: {source}")
    pixels = image[..., :3] if image.ndim == 3 else image
    samples = pixels[pixels > 0]
    maximum = int(pixels.max())
    level_max = np.iinfo(image.dtype).max
    gain, gamma = 1.0, 1.0
    result = image.copy()
    if samples.size:
        reference = float(np.percentile(samples, VIG_REFERENCE_PERCENTILE))
        gain = max(1.0, min(VIG_TARGET_LEVEL * level_max / reference, level_max / maximum))
        scaled_reference = reference * gain / level_max
        if 0 < scaled_reference < VIG_TARGET_LEVEL:
            gamma = math.log(VIG_TARGET_LEVEL) / math.log(scaled_reference)
        # A single monotonic lookup table is applied to the full image.
        # No local contrast processing, geometry changes or filtering.
        levels = np.arange(level_max + 1, dtype=np.float64)
        table = np.rint(np.power(np.minimum(levels * gain / level_max, 1.0), gamma) * level_max).astype(image.dtype)
        transformed = table[pixels]
        if image.ndim == 3:
            result[..., :3] = transformed  # Preserve alpha exactly, if present.
        else:
            result = transformed
    try:
        ok, encoded = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    except cv2.error as exc:
        raise ValueError(f"Cannot encode vig image {source}: {exc}") from exc
    if not ok:
        raise ValueError(f"Cannot encode vig image: {source}")
    info = {
        'dtype': str(image.dtype), 'width': image.shape[1], 'height': image.shape[0],
        'minimum': int(pixels.min()), 'maximum': maximum, 'gain': gain,
        'gamma': gamma, 'all_black': maximum == 0,
    }
    return encoded.tobytes(), info


def copy_plan(plan, output_root, vig_brightness=None):
    """Copy into a new/empty output; keep all originals and avoid overwrites."""
    if not plan:
        raise ValueError("No labelled, existing source images to copy. Check the messages above.")
    if vig_brightness is None:
        vig_brightness = VIG_BRIGHTNESS_ENABLED
    if vig_brightness and any(task == 'vig' for task, _, _, _ in plan):
        image_dependencies()  # Check dependencies before creating output folders.
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError(f"Choose a NEW or EMPTY output folder: {output_root}")
    for _, _, source, _ in plan:
        if source.is_relative_to(output_root):
            raise ValueError("An input image is inside the output folder. Choose a different output.")

    # Plan every destination before writing. Different files can share a name.
    destinations = []
    used = set()
    for task, label, source, _ in plan:
        name = source.with_suffix('.png').name if task == 'vig' and vig_brightness else source.name
        destination = output_root / task / label / name
        suffix = destination.suffix
        key = str(destination).casefold()
        if key in used:
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
            destination = destination.with_name(f"{source.stem}__{digest}{suffix}")
            number = 2
            while str(destination).casefold() in used:
                destination = destination.with_name(f"{source.stem}__{digest}_{number}{suffix}")
                number += 1
        used.add(str(destination).casefold())
        destinations.append((task, label, source, destination))

    for task in TASK_COLUMNS:
        for label in ("good", "bad"):
            (output_root / task / label).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    vig_processed = 0
    for task, label, source, destination in destinations:
        # Exclusive creation also protects against a second concurrent run.
        if task == 'vig' and vig_brightness:
            encoded, info = brighten_vig(source)
            with destination.open('xb') as output_stream:
                output_stream.write(encoded)
            vig_processed += 1
            if info['all_black']:
                print(f"WARNING: all-zero vig image; brightness cannot recover signal: {source}")
            if vig_processed <= 3:
                print(
                    f"  VIG {source.name}: {info['dtype']} {info['width']}x{info['height']}, "
                    f"input range={info['minimum']}..{info['maximum']}, "
                    f"gain={info['gain']:.3g}, gamma={info['gamma']:.3g} -> PNG"
                )
        else:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
        shutil.copystat(source, destination)
        counts[(task, label)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, default=CSV_FILES, help="One or more manually tagged CSVs")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="New or empty output folder")
    parser.add_argument("--tray", type=Path, default=TRAY_ROOT, help="Optional current tray root for path remapping")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count without copying")
    parser.add_argument("--no-vig-brightness", action="store_true", help="Copy vig unchanged as well")
    args = parser.parse_args()
    try:
        tray_root = args.tray.expanduser().resolve() if args.tray is not None else None
        if tray_root is not None and not tray_root.is_dir():
            raise ValueError(f"Tray folder does not exist: {tray_root}")
        csv_files = [path.expanduser().resolve() for path in args.csv]
        output_root = args.output.expanduser().resolve()
        if GOOD_TAGS & BAD_TAGS:
            raise ValueError("GOOD_TAGS and BAD_TAGS must not overlap.")
        plan, stats = build_plan(csv_files, tray_root)
        if args.dry_run:
            if not plan:
                raise ValueError("No labelled, existing source images found.")
            counts = Counter((task, label) for task, label, _, _ in plan)
            print("\nDRY RUN: planned image counts; no files written.")
        else:
            counts = copy_plan(plan, output_root, VIG_BRIGHTNESS_ENABLED and not args.no_vig_brightness)
            print(f"\nFinished. Output: {output_root}")
        for task in TASK_COLUMNS:
            print(
                f"{task:6} good={counts[(task, 'good')]} bad={counts[(task, 'bad')]} "
                f"unknown_rows={stats[(task, 'unknown_rows')]} "
                f"empty_sources_rows={stats[(task, 'empty_sources_rows')]} "
                f"missing_references={stats[(task, 'missing_references')]} "
                f"duplicate_references={stats[(task, 'duplicate_references')]}"
            )
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
