# platform/macos/icons.py
# Loads the menu-bar icon images for each app state (idle/recording/processing).
# Windows mirror: platform/windows/icons.py.
from pathlib import Path
from PIL import Image

from ...utils import resolve_asset_path

ASSETS_DIR = Path(resolve_asset_path("platform/macos/assets"))

def get_tray_icons() -> dict:
    return {
        "idle": Image.open(ASSETS_DIR / "tray_idle.png"),
        "recording": Image.open(ASSETS_DIR / "tray_recording.png"),
        "processing": Image.open(ASSETS_DIR / "tray_processing.png"),
    }
