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

# A CTranslate2 Whisper model is these files. model.bin and config.json are
# mandatory; the tokenizer/vocabulary pair varies slightly between conversions,
# so they're copied when present rather than demanded.
REQUIRED_FILES = ("model.bin", "config.json")
USER_SETTINGS = "user_settings.yaml"


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
        if (d / "model.bin").exists():
            return d
    return None


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


# `--export-model DEST`: bundle the configured model into a portable folder.
# Returns a process exit code.
def export_model(dest: str) -> int:
    from .config_manager import ConfigManager
    from .model_registry import ModelRegistry

    cm = ConfigManager(quiet=True)
    whisper_cfg = cm.get_whisper_config()
    model_key = whisper_cfg.get("model", "base")
    registry = ModelRegistry(whisper_models_config=whisper_cfg.get("models", {}))

    model = registry.get_model(model_key)
    if model and model.is_local_path:
        snapshot = Path(model.source)
        if not (snapshot / "model.bin").exists():
            print(f"ERROR: '{model_key}' points at {snapshot}, but no model.bin is there.")
            return 1
    else:
        snapshot = _find_cached_snapshot(model_key, registry.get_cache_folder(model_key))
        if snapshot is None:
            print(f"ERROR: Model '{model_key}' isn't downloaded on this machine yet.")
            print("   Run Whisper Local once (or --selftest) on a machine WITH internet,")
            print("   then export from there.")
            return 1

    # DEST is always treated as a directory to place the bundle in, and created
    # if absent. No "does it look like a file?" guessing — a folder name that
    # happens to contain a dot (D:\transfer.v2, "OneDrive - Corp", a mktemp dir)
    # would be silently rewritten and the model would land somewhere else.
    parent = Path(dest).expanduser()
    target = parent / f"whisper-local-model-{model_key}"

    print(f"[export] Exporting model '{model_key}'")
    print(f"   from: {snapshot}")
    print(f"   to:   {target}")
    try:
        copied = _copy_snapshot(snapshot, target)
    except OSError as e:
        print(f"ERROR: Export failed: {e}")
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
    missing = [f for f in REQUIRED_FILES if not (source / f).exists()]
    if missing:
        print(f"ERROR: {source} doesn't look like a Whisper model. Missing: {', '.join(missing)}")
        print("   Expected the folder produced by --export-model.")
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
    print("   Restart Whisper Local (tray menu -> Restart) to load it.")
    return 0


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
