# hardware_detection.py
# Thin platform-agnostic wrapper over the OS-specific GPU probe, so callers can
# ask about hardware without importing a platform backend directly.
from .platform import gpu as _platform_gpu


def detect_and_print(configured_device):
    return _platform_gpu.detect_and_print(configured_device)
