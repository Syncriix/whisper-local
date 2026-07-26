# platform/windows/app.py
# Windows app-lifecycle bits: single-keypress input (msvcrt) and the thread
# requirements the main loop must respect. Windows needs no special main-thread
# handling, so these are mostly trivial.
# macOS mirror: platform/macos/app.py (NSApplication run loop, which DOES
# require the main thread).
import msvcrt


def setup():
    pass

def run_event_loop(shutdown_event):
    while not shutdown_event.wait(timeout=0.1):
        pass

def getch():
    return msvcrt.getwch()
