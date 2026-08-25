# platform/windows/hotkeys.py
# Global hotkey registration via the `global-hotkeys` library, including the key
# name mapping from this app's config syntax to the library's expected format.
# macOS mirror: platform/macos/hotkeys.py (NSEvent taps).
from global_hotkeys import (clear_hotkeys, register_hotkeys, start_checking_hotkeys,
                             stop_checking_hotkeys)

# global-hotkeys library expects: 'control + window + shift' format
KEY_MAP = {
    'ctrl': 'control',
    'win': 'window',
    'windows': 'window',
    'cmd': 'window',
    'super': 'window',
    'esc': 'escape',
}

def _normalize_hotkey(hotkey_str: str) -> str:
    keys = hotkey_str.lower().split('+')
    converted = [KEY_MAP.get(k.strip(), k.strip()) for k in keys]
    return ' + '.join(converted)

# REPLACES the current bindings — it does not add to them. The macOS mirror
# rebinds its whole list on every call, and callers rely on that: the pause
# hotkey re-registers a reduced set and then the full set again on resume.
#
# global-hotkeys keeps its registrations across stop_checking_hotkeys(), so
# without clear_hotkeys() the second register() raises "already registered".
# That exception aborted the pause path before start() could run, leaving the
# listener stopped with every hotkey dead — including pause itself, so there
# was no way back without restarting the app (issue #6).
def register(bindings: list):
    clear_hotkeys()
    normalized = []
    for binding in bindings:
        hotkey_str = binding[0]
        normalized_binding = [_normalize_hotkey(hotkey_str)] + binding[1:]
        normalized.append(normalized_binding)
    register_hotkeys(normalized)

def start():
    start_checking_hotkeys()

def stop():
    stop_checking_hotkeys()
