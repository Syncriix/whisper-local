# platform/windows/permissions.py
# Permission checks are a macOS concern (Accessibility/Input Monitoring);
# Windows needs none, so these stubs always report granted.
# macOS mirror: platform/macos/permissions.py (the real implementation).
def check_accessibility_permission() -> bool:
    return True


def handle_missing_permission(config_manager) -> bool:
    return True
