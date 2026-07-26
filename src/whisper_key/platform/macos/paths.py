# platform/macos/paths.py
# Resolves the per-user config directory (~/.whisperkey).
# Windows mirror: platform/windows/paths.py (%APPDATA%\whisperkey).
import subprocess
from pathlib import Path

def get_app_data_path():
    return Path.home() / '.whisperkey'

def open_file(path):
    subprocess.run(['open', str(path)])
