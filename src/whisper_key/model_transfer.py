# model_transfer.py
# Move a Whisper model between machines WITHOUT network access, for locked-down
# environments where huggingface.co is blocked or pending security review.
#
# `--export-model DEST` copies the already-downloaded model out of the HuggingFace
# cache into a plain folder (resolving the cache's symlinks into real files) that
# can go on a USB stick or a network share. `--import-model SRC` installs such a
# folder on the offline machine and points the config at it.
#
# Nothing here talks to the network. It leans on the fact that faster-whisper
# accepts a local directory as a model source, and that ModelRegistry already
# treats a filesystem path as an always-cached model.

import logging
import shutil
from pathlib import Path

from ruamel.yaml import YAML

from .utils import get_user_app_data_path

logger = logging.getLogger(__name__)

# Two model formats travel through here. A CTranslate2 Whisper model (the
# faster_whisper backend) is identified by model.bin + config.json; an OpenVINO
# IR model (the openvino backend) by its encoder/decoder XML pair. Tokenizer
# files vary between conversions, so they're copied when present, not demanded.
CT2_MARKERS = ("model.bin", "config.json")
OPENVINO_MARKERS = ("openvino_encoder_model.xml", "openvino_decoder_model.xml")
# One weight file per format proves a snapshot is complete, not a partial download
WEIGHT_FILES = ("model.bin", "openvino_encoder_model.bin")
USER_SETTINGS = "user_settings.yaml"


# Which model format a folder holds: 'ct2', 'openvino', or None if neither.
def _detect_model_format(folder: Path):
    if all((folder / f).exists() for f in CT2_MARKERS):
        return "ct2"
    if all((folder / f).exists() for f in OPENVINO_MARKERS):
        return "openvino"
    return None


def _hf_cache_root() -> Path:
    import os
    userprofile = os.environ.get("USERPROFILE")
    base = Path(userprofile) if userprofile else Path.home()
    return base / ".cache" / "huggingface" / "hub"


# Locate the snapshot directory for a cached model key (e.g. "base"). Returns
# None when the model hasn't been downloaded on this machine yet.
def _find_cached_snapshot(model_key: str, cache_folder: str):
    snapshots = _hf_cache_root() / cache_folder / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [d for d in snapshots.iterdir() if d.is_dir()]
    if not candidates:
        return None
    # Prefer a snapshot that actually contains the weights — a partial/aborted
    # download can leave an empty revision directory behind.
    for d in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if any((d / w).exists() for w in WEIGHT_FILES):
            return d
    return None


# HF-cache folder name for the OpenVINO variant of a model key, derived from
# the engine's own catalog so the two can't drift apart. None for keys with no
# OpenVINO twin (the distil-* models).
def _openvino_cache_folder(model_key: str, whisper_cfg: dict):
    from .whisper_engine_openvino import _PRECISION_SUFFIX, _SUPPORTED_MODELS

    base_name = _SUPPORTED_MODELS.get(model_key)
    if base_name is None:
        return None
    precision = _PRECISION_SUFFIX.get(whisper_cfg.get("compute_type", "int8"), "int8")
    return f"models--OpenVINO--{base_name}-{precision}-ov"


# Copy every file in the snapshot into `dest`. HF stores blobs once and symlinks
# each revision at them, so we resolve() before copying — otherwise the export
# would be a folder of dangling links on the target machine.
def _copy_snapshot(snapshot: Path, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in snapshot.iterdir():
        if item.is_dir():
            continue
        source = item.resolve()
        target = dest / item.name
        shutil.copy2(source, target)
        copied.append((item.name, target.stat().st_size))
    return copied


# Where to put an export when the user doesn't say. Desktop is the most likely
# place they'll actually find it; fall back to the home directory on the rare
# setup without one (redirected profiles, some non-English Windows installs).
def _default_export_dir() -> Path:
    desktop = Path.home() / "Desktop"
    return desktop if desktop.is_dir() else Path.home()


# Check we can actually write the export before copying 150 MB, and explain the
# problem in the user's terms. A raw "[WinError 3] cannot find the path" doesn't
# tell someone that they simply don't have the drive they typed.
def _validate_destination(parent: Path) -> str:
    import os
    import string

    anchor = parent.anchor  # 'D:\' on Windows, '/' on POSIX
    if anchor and not os.path.exists(anchor):
        drives = [d + ":" for d in string.ascii_uppercase if os.path.exists(d + ":" + os.sep)]
        detail = f"Drives on this machine: {', '.join(drives)}" if drives else ""
        return (f"There's no {anchor.rstrip(os.sep)} drive on this machine.\n"
                f"   {detail}\n"
                f"   Try a folder that exists, e.g.:  "
                f'whisper-local --export-model "{_default_export_dir()}"')
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return (f"No permission to write to {parent}.\n"
                f"   Pick a folder you own, e.g.:  "
                f'whisper-local --export-model "{_default_export_dir()}"')
    except OSError as e:
        return f"Can't use {parent} as a destination: {e}"
    return ""


# `--export-model [DEST]`: bundle the configured model into a portable folder.
# DEST defaults to the Desktop. Returns a process exit code.
def export_model(dest: str = None) -> int:
    from .config_manager import ConfigManager
    from .model_registry import ModelRegistry

    cm = ConfigManager(quiet=True)
    whisper_cfg = cm.get_whisper_config()
    model_key = whisper_cfg.get("model", "base")
    registry = ModelRegistry(whisper_models_config=whisper_cfg.get("models", {}))

    model = registry.get_model(model_key)
    if model and model.is_local_path:
        snapshot = Path(model.source)
        if _detect_model_format(snapshot) is None:
            print(f"ERROR: '{model_key}' points at {snapshot}, but no model files are there.")
            return 1
    else:
        # The active backend decides WHICH cached artifact "medium" means: the
        # CT2 conversion for faster_whisper, the OpenVINO IR for openvino.
        # Exporting what the machine actually runs is the only version that is
        # guaranteed present and guaranteed useful on the target machine.
        backend = whisper_cfg.get("backend", "faster_whisper")
        if backend == "openvino":
            cache_folder = _openvino_cache_folder(model_key, whisper_cfg)
            if cache_folder is None:
                print(f"ERROR: Model '{model_key}' has no OpenVINO variant to export.")
                return 1
        else:
            cache_folder = registry.get_cache_folder(model_key)
        snapshot = _find_cached_snapshot(model_key, cache_folder)
        if snapshot is None:
            print(f"ERROR: Model '{model_key}' isn't downloaded on this machine yet.")
            print("   Run Whisper Local once (or --selftest) on a machine WITH internet,")
            print("   then export from there.")
            return 1

    # DEST is always treated as a directory to place the bundle in, and created
    # if absent. No "does it look like a file?" guessing — a folder name that
    # happens to contain a dot (D:\transfer.v2, "OneDrive - Corp", a mktemp dir)
    # would be silently rewritten and the model would land somewhere else.
    parent = Path(dest).expanduser() if dest else _default_export_dir()
    problem = _validate_destination(parent)
    if problem:
        print(f"ERROR: {problem}")
        return 1
    target = parent / f"whisper-local-model-{model_key}"

    print(f"[export] Exporting model '{model_key}'")
    print(f"   from: {snapshot}")
    print(f"   to:   {target}")
    try:
        copied = _copy_snapshot(snapshot, target)
    except OSError as e:
        print(f"ERROR: Export failed: {e}")
        print(f"   Destination was {target}")
        return 1

    total = sum(size for _, size in copied)
    for name, size in copied:
        print(f"     {name:26s} {size / 1e6:8.2f} MB")
    print(f"   OK  {len(copied)} files, {total / 1e6:.0f} MB total")
    print()
    print("Next: copy that folder to the offline machine (USB or network share), then run:")
    print(f'   whisper-local --import-model "{target}"')
    return 0


# `--import-model SRC`: register a portable model folder and make it active.
# Copies into the app-data dir so the model survives the USB stick going away.
def import_model(src: str, keep_in_place: bool = False) -> int:
    source = Path(src).expanduser()
    if not source.is_dir():
        print(f"ERROR: Not a folder: {source}")
        return 1
    model_format = _detect_model_format(source)
    if model_format is None:
        print(f"ERROR: {source} doesn't look like a Whisper model.")
        print(f"   Expected the folder produced by --export-model, containing either")
        print(f"   {' + '.join(CT2_MARKERS)} (faster_whisper) or "
              f"{' + '.join(OPENVINO_MARKERS)} (openvino).")
        return 1

    model_name = source.name.replace("whisper-local-model-", "") or "imported"
    key = f"local-{model_name}"

    if keep_in_place:
        # Referencing a network share directly: nothing is copied, so IT can host
        # one canonical copy and every machine points at it.
        installed = source
        print(f"[import] Registering model in place: {installed}")
    else:
        installed = Path(get_user_app_data_path()) / "models" / model_name
        print(f"[import] Installing model to: {installed}")
        installed.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, installed / item.name)

    if not _register_model(key, str(installed), model_name):
        return 1

    print(f"   OK  Registered as '{key}' and set as the active model.")
    _warn_on_backend_mismatch(model_format)
    print("   Restart Whisper Local (tray menu -> Restart) to load it.")
    return 0


# Each model format loads on exactly one backend. Importing an OpenVINO bundle
# onto a faster_whisper machine (or vice versa) would otherwise fail only at the
# next launch, with a loader error that doesn't name the real cause — on the
# offline machines this feature targets, that's a dead end. Warn here instead.
_FORMAT_BACKENDS = {"ct2": "faster_whisper", "openvino": "openvino"}


def _warn_on_backend_mismatch(model_format: str):
    needed = _FORMAT_BACKENDS[model_format]
    settings = Path(get_user_app_data_path()) / USER_SETTINGS
    try:
        with open(settings, encoding="utf-8") as f:
            data = YAML().load(f) or {}
        configured = (data.get("whisper") or {}).get("backend", "faster_whisper")
    except Exception:
        configured = "faster_whisper"
    if configured != needed:
        print(f"   WARNING: this is a {model_format} model, but whisper.backend "
              f"is '{configured}'.")
        print(f"   Set 'backend: {needed}' under 'whisper:' in user_settings.yaml, "
              "or the model won't load.")


# Add the model to whisper.models in user_settings.yaml and select it. Written
# with round-trip YAML so the user's comments and other settings survive.
def _register_model(key: str, path: str, label: str) -> bool:
    settings = Path(get_user_app_data_path()) / USER_SETTINGS
    yaml = YAML()
    try:
        data = {}
        if settings.exists():
            with open(settings, encoding="utf-8") as f:
                data = yaml.load(f) or {}
        whisper = data.setdefault("whisper", {})
        models = whisper.setdefault("models", {})
        models[key] = {
            "source": path,
            "label": f"{label} (offline import)",
            "group": "custom",
            "enabled": True,
        }
        whisper["model"] = key
        settings.parent.mkdir(parents=True, exist_ok=True)
        with open(settings, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        logger.info(f"Registered offline model '{key}' -> {path}")
        return True
    except Exception as e:
        logger.error(f"Could not register model: {e}")
        print(f"ERROR: Could not write settings: {e}")
        return False
