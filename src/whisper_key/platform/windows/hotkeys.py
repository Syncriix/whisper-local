# platform/windows/hotkeys.py
# Global hotkey registration via the `global-hotkeys` library, including the key
# name mapping from this app's config syntax to the library's expected format.
# macOS mirror: platform/macos/hotkeys.py (NSEvent taps).
#
# Library caveats this wrapper papers over (global-hotkeys 0.1.7):
#   • Registrations persist across stop() — register() must clear first or a
#     re-register raises "already registered".
#   • The checker thread iterates a live view of the bindings dict and invokes
#     callbacks from inside that loop; stop() only sets a flag and never joins.
#     So registration changes need a settle wait after stop(), and must NEVER
#     be made from within a hotkey callback (that crashes the checker thread).
import time

from global_hotkeys import register_hotkeys, start_checking_hotkeys, stop_checking_hotkeys, clear_hotkeys

# The checker loop sleeps 20ms per pass; 0.2s comfortably outlives its final
# iteration after stop() flips the flag (the library itself uses 0.7s).
STOP_SETTLE_SECONDS = 0.2

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

def register(bindings: list):
    # Always replaces the full set (matching the macOS mirror). Callers must
    # stop() first and must not be running on the hotkey checker thread.
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
    # Let the checker thread's in-flight pass finish before callers mutate
    # bindings — stop_checking_hotkeys() sets a flag but does not join.
    time.sleep(STOP_SETTLE_SECONDS)
