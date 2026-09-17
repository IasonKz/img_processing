from __future__ import annotations

import csv
import hashlib
import json
import copy
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import yaml

LABELS = {"bad": 0, "good": 1}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "threshold", "selection", "test")
MODEL_NAMES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "swin_t", "patchcore")

DEFAULTS = {
    "pretrained": True, "image_size": "auto", "image_size_limit": 4096, "resize": "pad",
    "device": "auto", "cpu_threads": 4, "num_workers": 0,
    "models": list(MODEL_NAMES), "seeds": [42, 123], "split_seed": 42,
    "split_fractions": {"train": .55, "val": .10, "threshold": .10, "selection": .10, "test": .15},
    "groups_csv": None, "group_regex": None, "grouping": "pixel_hash", "previous_run": None,
    "near_duplicate_distance": 2, "near_duplicate_report_limit": 10000,
    "fine_tune_previous": True, "ensembles": True, "export_images": True,
    "continue_on_model_error": False, "independent_units_confirmed": False,
    "policy": {"max_bad_escape_rate": .01, "max_good_reject_rate": .20, "confidence": .95, "threshold_margin": True},
    "release": {"require_statistical_evidence": True, "require_complete_comparison": True},
    "preserve_native_resolution": True,
    "augmentation": {"brightness": 0.0, "contrast": 0.0, "rotation_degrees": 0, "rotation90": False},
    "training": {"epochs": 30, "warmup_epochs": 2, "patience": 7, "batch_size": 2,
                 "eval_batch_size": 2, "accumulation_steps": 8, "learning_rate": .0001,
                 "head_learning_rate": .001, "weight_decay": .0001,
                 "fine_tune_lr_factor": .2, "amp": True, "freeze_batch_norm": True,
                 "imbalance": "auto", "imbalance_trigger": 1.5, "max_class_weight": 20.0,
                 "mode": "full", "optimizer": "adamw", "scheduler": "cosine",
                 "momentum": .9, "telemetry_every": 25},
    "patchcore": {"backbone": "resnet18", "reservoir_size": 10000, "coreset_size": 512,
                  "projection_dim": 32, "query_chunk_size": 1024, "memory_chunk_size": 1024},
    "deployment": {"opset": 17, "score_atol": 1e-5, "score_rtol": 1e-4,
                   "max_decision_disagreements": 0},
    "search": {"enabled": False, "total_budget_seconds": 7200,
               "max_trials_per_model": 8, "startup_trials": 3, "min_finetune_epochs": 2,
               "shortlist_per_model": 1, "minimum_trial_seconds": 60,
               "selection_reserve_fraction": .15, "seed": 42, "space": {}},
}
PROFILES = {
    "laptop": {"models": ["resnet18"], "seeds": [42], "ensembles": False,
               "device": "cpu", "pilot": True,
               "budget_seconds": 7200,
               "search": {"total_budget_seconds": 7200, "max_trials_per_model": 8, "startup_trials": 3,
                          "space": {"mode": ["head_only", "full"], "learning_rate": [1e-6, 3e-5],
                                    "head_learning_rate": [3e-4, 1e-3], "scheduler": ["cosine"],
                                    "imbalance": ["none", "weighted_loss", "sampler"]}},
               "training": {"epochs": 15, "warmup_epochs": 2, "patience": 5,
                            "batch_size": 1, "eval_batch_size": 1, "accumulation_steps": 4,
                            "learning_rate": .00001, "amp": False},
               "patchcore": {"reservoir_size": 4000, "coreset_size": 256}},
    "workstation": {"models": list(MODEL_NAMES), "seeds": [42, 123],
                    "device": "cuda", "pilot": False, "budget_seconds": 21600,
                    "search": {"total_budget_seconds": 21600, "max_trials_per_model": 16, "startup_trials": 4,
                               "space": {"mode": ["head_only", "full", "last_block"],
                                         "learning_rate": [1e-6, 1e-4], "head_learning_rate": [1e-4, 3e-3],
                                         "scheduler": ["cosine", "plateau"]}},
                    "training": {"learning_rate": .00001}},
    "workstation_large": {"models": list(MODEL_NAMES), "seeds": [42, 123],
                          "device": "cuda", "pilot": False, "budget_seconds": 43200,
                          "search": {"total_budget_seconds": 43200, "max_trials_per_model": 24, "startup_trials": 5,
                                     "space": {"mode": ["head_only", "full", "last_block"],
                                               "learning_rate": [1e-6, 1e-4], "head_learning_rate": [1e-4, 3e-3],
                                               "scheduler": ["cosine", "plateau"], "optimizer": ["adamw", "sgd"]}},
                          "training": {"learning_rate": .00001}},
}


def merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge_dict(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(root, relative):
    """Resolve a stored relative path without permitting absolute paths or traversal."""
    from pathlib import PureWindowsPath
    relative = str(relative).replace("\\", "/")
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path resolves outside its root: {relative}")
    return target


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dataset_fingerprint(rows):
    return object_hash(sorted((r["pixel_hash"], int(r["label"]), r["group"], r["split"]) for r in rows))


@contextmanager
def run_lock(run):
    """Exclusive local mutation lock. Interrupted locks require an explicit recovery command."""
    path = Path(run) / "operation.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Run is locked: {path}. Check the owning process; use resume --recover-lock only after it has stopped.") from exc
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "created_at": utc_now()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)


# v3 file-I/O patch: keep atomic snapshots while tolerating brief Windows locks.
# No changes to training, thresholds, configuration or serialized file formats.
def _replace_file_with_retry(source, destination):
    """Retry only Windows access/sharing/lock failures, never delete the target.

    An open reader may temporarily prevent replacement on Windows. WinError 5
    may also mean a permanent permission problem, so retries are bounded and
    the final OSError is re-raised unchanged. No ACL/read-only bypass is used.
    The 16 attempts add at most about 2.31 seconds of backoff, excluding I/O.
    """
    for attempt in range(16):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 15:
                raise
            time.sleep(min(0.01 * (2 ** attempt), 0.2))


@contextmanager
def _atomic_text_writer(path, *, encoding="utf-8", newline=None):
    """Publish a closed, complete file in one replace, retaining the old one on error.

    Each call owns a different temporary file in the destination directory.
    This avoids collisions between writers, but is not a replacement for the
    existing run_lock: concurrent successful writes are still last-writer-wins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(temporary)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding=encoding, newline=newline)
        except BaseException:
            os.close(fd)
            raise
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        # The write handle must be closed before os.replace on Windows.
        _replace_file_with_retry(tmp, path)
    finally:
        # Cleanup only OUR temporary file. Never mask a write/replace exception
        # or delete the last valid destination as a workaround for a lock.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_json(path, value):
    # Serialize before touching the filesystem; invalid/nonfinite data still fails.
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with _atomic_text_writer(path) as handle:
        handle.write(text)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    with _atomic_text_writer(path, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value, base):
    from pathlib import PureWindowsPath
    if os.name != "nt" and PureWindowsPath(str(value)).drive:
        raise ValueError(f"Windows path cannot resolve on this OS: {value}. Configure a local path or restore with --data-root.")
    return str((Path(base) / Path(value).expanduser()).resolve())


def load_config(path, profile=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    cfg = merge_dict(merge_dict(DEFAULTS, PROFILES[profile]) if profile else DEFAULTS, raw)
    if profile:
        cfg["profile"] = profile
    if not isinstance(cfg, dict) or cfg.get("task") not in ("top", "bottom", "vig"):
        raise ValueError("Config task must be top, bottom or vig.")
    base = path.parent
    for key in ("data_root", "output_root", "weights_dir"):
        cfg[key] = resolve_path(cfg[key], base)
    for key in ("previous_run", "groups_csv"):
        if cfg.get(key):
            cfg[key] = resolve_path(cfg[key], base)
    if Path(cfg["output_root"]).is_relative_to(Path(cfg["data_root"])):
        raise ValueError("output_root must be outside data_root to avoid reading predictions as labels.")
    validate_config(cfg)
    cfg["config_file"] = str(path)
    return cfg


def validate_config(cfg):
    if cfg.get("task") not in ("top", "bottom", "vig"):
        raise ValueError("Task must be top, bottom or vig.")
    size = cfg.get("image_size", "auto")
    if size != "auto" and (not isinstance(size, int) or isinstance(size, bool) or size < 32):
        raise ValueError("image_size must be 'auto' or an integer >=32.")
    if not isinstance(cfg.get("image_size_limit"), int) or cfg["image_size_limit"] < 32:
        raise ValueError("image_size_limit must be an integer >=32.")
    if cfg.get("resize", "pad") not in ("pad", "fit"):
        raise ValueError("resize must be pad (no scaling) or fit (explicit down/upscaling).")
    if cfg.get("preserve_native_resolution", False) and cfg.get("resize", "pad") != "pad":
        raise ValueError("preserve_native_resolution requires resize=pad. No cropping or resampling is allowed.")
    if cfg.get("grouping", "pixel_hash") not in ("pixel_hash", "tray_unit"):
        raise ValueError("grouping must be pixel_hash or tray_unit; groups_csv/group_regex can override it.")
    aug = cfg.get("augmentation", {})
    if any(float(aug.get(key, 0)) != 0 for key in ("brightness", "contrast", "rotation_degrees")):
        raise ValueError("This version permits only exact quarter-turn augmentation. Set brightness, contrast and rotation_degrees to 0; use rotation90: true.")
    if not isinstance(aug.get("rotation90", False), bool):
        raise ValueError("augmentation.rotation90 must be YAML true/false.")
    fractions = cfg["split_fractions"]
    if not isinstance(fractions, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in fractions.values()):
        raise ValueError("split_fractions must be a mapping of finite numeric values.")
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-6:
        raise ValueError("split_fractions must contain train/val/threshold/selection/test and sum to 1.")
    if any(v <= 0 for v in fractions.values()):
        raise ValueError("All five split fractions must be positive.")
    if cfg.get("group_regex"):
        import re
        if "group" not in re.compile(cfg["group_regex"]).groupindex:
            raise ValueError("group_regex must contain a named capture (?P<group>...).")
    for key in ("max_bad_escape_rate", "max_good_reject_rate"):
        if not 0 <= cfg["policy"][key] <= 1:
            raise ValueError(f"policy.{key} must be between 0 and 1.")
    if not 0 < cfg["policy"].get("confidence", .95) < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if len(set(cfg["models"])) != len(cfg["models"]):
        raise ValueError("Duplicate model names.")
    if set(cfg["models"]) - set(MODEL_NAMES):
        raise ValueError(f"models must be drawn from {MODEL_NAMES}.")
    if not cfg["models"] or not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Provide nonempty models and unique seeds.")
    train = cfg["training"]
    if train.get("mode", "full") not in ("full", "head_only", "last_block"):
        raise ValueError("training.mode must be full, head_only or last_block.")
    if train.get("optimizer", "adamw") not in ("adamw", "sgd"):
        raise ValueError("training.optimizer must be adamw or sgd.")
    if train.get("scheduler", "cosine") not in ("cosine", "plateau", "constant"):
        raise ValueError("training.scheduler must be cosine, plateau or constant.")
    warmup = train.get("warmup_epochs", 2)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("training.warmup_epochs must be a nonnegative integer.")
    minimum_phase = train.get("min_finetune_epochs", 1)
    if not isinstance(minimum_phase, int) or isinstance(minimum_phase, bool) or minimum_phase < 1:
        raise ValueError("training.min_finetune_epochs must be a positive integer.")
    if not 0 <= train.get("momentum", .9) < 1:
        raise ValueError("training.momentum must be in [0,1).")
    for key in ("weight_decay", "gradient_clip"):
        value = train.get(key, .0001 if key == "weight_decay" else 5.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative.")
    budget = cfg.get("budget_seconds", 7200)
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget_seconds must be finite and positive.")
    for key in ("epochs", "batch_size", "eval_batch_size", "accumulation_steps", "patience"):
        if not isinstance(train[key], int) or isinstance(train[key], bool) or train[key] < 1:
            raise ValueError(f"training.{key} must be an integer >=1.")
    for key in ("learning_rate", "head_learning_rate", "fine_tune_lr_factor", "max_class_weight"):
        if not isinstance(train[key], (int, float)) or not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive.")
    if train["max_class_weight"] < 1:
        raise ValueError("max_class_weight must be >=1.")
    if train.get("imbalance", "auto") not in ("auto", "none", "weighted_loss", "sampler"):
        raise ValueError("Unknown imbalance strategy.")
    for flag in ("pretrained", "independent_units_confirmed", "export_images", "ensembles", "fine_tune_previous", "continue_on_model_error"):
        if not isinstance(cfg[flag], bool):
            raise ValueError(f"{flag} must be YAML true/false, not a quoted string.")
    if Path(cfg["output_root"]).resolve().is_relative_to(Path(cfg["data_root"]).resolve()):
        raise ValueError("output_root must be outside data_root.")
    return cfg


def load_project(path, profile="laptop", task=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Project configuration must be a YAML mapping.")
    if "task" in raw and "tasks" not in raw:
        cfg = load_config(path, profile=profile)
        if task and cfg["task"] != task:
            raise ValueError("Requested task differs from single-task config.")
        cfg.setdefault("profile", profile)
        return [cfg]
    unknown = set(raw) - {"schema_version", "project_name", "paths", "defaults", "profiles", "tasks"}
    if unknown:
        raise ValueError(f"Unknown project keys: {sorted(unknown)}")
    if profile not in PROFILES:
        raise ValueError("Profile must be laptop, workstation or workstation_large.")
    paths = raw.get("paths", {})
    tasks = raw.get("tasks", {})
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Configure at least one task under tasks, with dataset_root.")
    if task and task not in tasks:
        raise ValueError(f"Task is not configured: {task}")
    result = []
    for name, options in tasks.items():
        if task and name != task:
            continue
        if not isinstance(options, dict) or not options.get("enabled", True):
            continue
        if not options.get("dataset_root"):
            raise ValueError(f"tasks.{name}.dataset_root is required.")
        # Profiles specify resource budgets; project defaults and task overrides are explicit experiments.
        cfg = merge_dict(DEFAULTS, PROFILES[profile])
        cfg = merge_dict(cfg, raw.get("defaults", {}))
        cfg = merge_dict(cfg, raw.get("profiles", {}).get(profile, {}))
        cfg = merge_dict(cfg, {k: v for k, v in options.items() if k not in ("dataset_root", "enabled")})
        cfg.update(task=name, profile=profile, project_name=raw.get("project_name", path.stem),
                   project_file=str(path), config_file=str(path),
                   data_root=resolve_path(options["dataset_root"], path.parent),
                   output_root=resolve_path(paths.get("output_root", "./results"), path.parent),
                   weights_dir=resolve_path(paths.get("weights_root", "./weights"), path.parent),
                   incoming_root=resolve_path(paths.get("incoming_root", "./incoming"), path.parent),
                   classification_output_root=resolve_path(paths.get("classification_output_root", "./classified"), path.parent))
        for key in ("previous_run", "groups_csv"):
            if cfg.get(key):
                cfg[key] = resolve_path(cfg[key], path.parent)
        validate_config(cfg)
        result.append(cfg)
    if not result:
        raise ValueError("No enabled tasks.")
    return result
