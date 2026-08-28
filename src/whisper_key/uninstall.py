# uninstall.py
# `--uninstall`: remove everything Whisper Local leaves on the machine, because
# there is nothing to remove it otherwise. The standalone build is a portable
# .exe with no installer, so it never appears in Add/Remove Programs (issue #8),
# and a pip uninstall drops the package but leaves settings, logs, transcripts,
# an autostart entry, and potentially gigabytes of downloaded models behind.
#
# Safety is the whole design here. This is the only code in the project that
# deletes user data, so:
#   * nothing is removed without the user typing a confirmation;
#   * the plan, with real sizes, is printed before anything happens;
#   * models are a SEPARATE opt-in, because they are large and slow to refetch;
#   * the HuggingFace cache is shared with other tools (a diarization or CLIP
#     model may sit beside ours), so only directories matching our own Whisper
#     model naming are ever considered — never the cache as a whole.

import logging
import os
import shutil
import sys
from pathlib import Path

from .utils import get_user_app_data_path

logger = logging.getLogger(__name__)

# Cache directories we are willing to delete. Anything not matching these stays,
# including `.locks` and any other project's models.
_MODEL_DIR_PREFIXES = (
    "models--Systran--faster-whisper",
    "models--Systran--faster-distil-whisper",
    "models--distil-whisper--distil",
    "models--openai--whisper",
)


def _dir_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    except OSError:
        return 0


def _human(num_bytes: int) -> str:
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.1f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.0f} MB"
    return f"{num_bytes / 1e3:.0f} KB"


def _hf_hub_root() -> Path:
    userprofile = os.environ.get("USERPROFILE")
    base = Path(userprofile) if userprofile else Path.home()
    return base / ".cache" / "huggingface" / "hub"


# Only ever returns directories whose names match our own model naming. Anything
# else in the shared cache belongs to another tool and is left strictly alone.
def _our_cached_models() -> list:
    root = _hf_hub_root()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith(_MODEL_DIR_PREFIXES):
            found.append((child, _dir_size(child)))
    return found


def _remove(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError as e:
        print(f"   ! could not remove {path}: {e}")
        logger.warning(f"uninstall could not remove {path}: {e}")
        return False


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# `--uninstall`. Returns a process exit code. Interactive by design: it must not
# be possible to wipe settings or an 11 GB model cache from a script by accident.
def run_uninstall() -> int:
    config_dir = Path(get_user_app_data_path())
    models = _our_cached_models()

    try:
        from . import autostart
        autostart_on = autostart.is_enabled()
    except Exception:
        autostart_on = False

    print("Whisper Local - uninstall")
    print("=" * 40)
    print()
    print("This removes what the app has written to your machine. It does NOT")
    print("delete the program itself - see the note at the end.")
    print()

    if config_dir.is_dir():
        print(f"  Settings and data   {config_dir}   ({_human(_dir_size(config_dir))})")
        print("                      settings, logs, transcripts, stats, voice commands")
    else:
        print("  Settings and data   (none found)")
    print(f"  Start on login      {'enabled - will be removed' if autostart_on else 'not enabled'}")

    total_models = sum(size for _, size in models)
    if models:
        print(f"  Downloaded models   {len(models)} in the HuggingFace cache ({_human(total_models)})")
        for path, size in models:
            print(f"                        {path.name}  {_human(size)}")
        print("                      (asked about separately - other tools' models are never touched)")
    else:
        print("  Downloaded models   (none found)")
    print()

    if not config_dir.is_dir() and not models and not autostart_on:
        print("Nothing to remove - this machine is already clean.")
        return 0

    if not _confirm("Remove settings, data and the autostart entry? [y/N] "):
        print("Cancelled. Nothing was removed.")
        return 1

    if autostart_on:
        try:
            from . import autostart
            autostart.disable()
            print("   removed the start-on-login entry")
        except Exception as e:
            print(f"   ! could not remove autostart entry: {e}")

    if config_dir.is_dir() and _remove(config_dir):
        print(f"   removed {config_dir}")

    # Models are deliberately a second, explicit decision: they are the large
    # and slow-to-replace part, and a user may be reinstalling rather than
    # leaving for good.
    if models:
        print()
        print(f"The {len(models)} downloaded model(s) total {_human(total_models)}.")
        print("Keep them if you might reinstall - they'd have to be downloaded again.")
        if _confirm(f"Also delete {_human(total_models)} of models? [y/N] "):
            for path, _ in models:
                if _remove(path):
                    print(f"   removed {path.name}")
        else:
            print("   kept the models.")

    print()
    print("Done. To remove the program itself:")
    if os.environ.get("PYAPP"):
        print(f"   delete {os.environ['PYAPP']}")
    elif sys.executable.lower().endswith("whisper-local.exe"):
        print(f"   delete {sys.executable}")
    else:
        print("   pip uninstall whisper-local")
    return 0
