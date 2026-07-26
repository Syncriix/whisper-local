# platform/windows/paths.py
# Resolves the per-user config directory (%APPDATA%\whisperkey).
# macOS mirror: platform/macos/paths.py (~/.whisperkey).
import os
from pathlib import Path

def get_app_data_path():
    return Path(os.getenv('APPDATA')) / 'whisperkey'

def open_file(path):
    os.startfile(path)
