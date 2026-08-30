# audio_feedback.py
# Short confirmation sounds for recording start/stop/complete — the audible half
# of the feedback the overlay provides visually, so the user knows a hotkey
# registered without looking. Playback is fire-and-forget on a background thread
# and any backend failure is swallowed: audio cues must never block dictation.
import logging
import os
import platform
import threading

from playsound3 import playsound

SOUND_BACKEND = "winmm" if platform.system() == "Windows" else None

from .utils import resolve_asset_path

class AudioFeedback:
    def __init__(self, enabled=True, transcription_complete_enabled=False,
                 ready_enabled=True, start_sound='', stop_sound='', cancel_sound='',
                 transcription_complete_sound='', ready_sound='',
                 send_phrase_sound=''):
        self.enabled = enabled
        self.transcription_complete_enabled = transcription_complete_enabled
        self.ready_enabled = ready_enabled
        self.logger = logging.getLogger(__name__)

        self.start_sound_path = resolve_asset_path(start_sound)
        self.stop_sound_path = resolve_asset_path(stop_sound)
        self.cancel_sound_path = resolve_asset_path(cancel_sound)
        self.transcription_complete_sound_path = resolve_asset_path(transcription_complete_sound)
        self.ready_sound_path = resolve_asset_path(ready_sound)
        self.send_phrase_sound_path = resolve_asset_path(send_phrase_sound) if send_phrase_sound else ''

        if not self.enabled:
            self.logger.info("Audio feedback disabled by configuration")
            print("   ✗ Audio feedback disabled")
        else:
            self._validate_sound_files()
            print("   ✓ Audio feedback enabled...")

    def _validate_sound_files(self):
        if self.ready_sound_path and not os.path.isfile(self.ready_sound_path):
            self.logger.warning(f"Ready sound file not found: {self.ready_sound_path}")

        if self.start_sound_path and not os.path.isfile(self.start_sound_path):
            self.logger.warning(f"Start sound file not found: {self.start_sound_path}")

        if self.stop_sound_path and not os.path.isfile(self.stop_sound_path):
            self.logger.warning(f"Stop sound file not found: {self.stop_sound_path}")

        if self.cancel_sound_path and not os.path.isfile(self.cancel_sound_path):
            self.logger.warning(f"Cancel sound file not found: {self.cancel_sound_path}")

        if self.send_phrase_sound_path and not os.path.isfile(self.send_phrase_sound_path):
            self.logger.warning(f"Send phrase sound file not found: {self.send_phrase_sound_path}")

        if self.transcription_complete_sound_path and not os.path.isfile(self.transcription_complete_sound_path):
            self.logger.warning(f"Transcription complete sound file not found: {self.transcription_complete_sound_path}")

    def _play_sound_file_async(self, file_path: str):
        def play():
            try:
                playsound(file_path, block=False, backend=SOUND_BACKEND)
            except Exception as e:
                self.logger.warning(f"Failed to play sound file {file_path}: {e}")

        threading.Thread(target=play, daemon=True).start()

    def play_start_sound(self):
        if self.enabled:
            self._play_sound_file_async(self.start_sound_path)

    def play_stop_sound(self):
        if self.enabled:
            self._play_sound_file_async(self.stop_sound_path)

    def play_cancel_sound(self):
        if self.enabled:
            self._play_sound_file_async(self.cancel_sound_path)

    def play_transcription_complete_sound(self):
        if self.enabled and self.transcription_complete_enabled:
            self._play_sound_file_async(self.transcription_complete_sound_path)

    # The send phrase was recognised and ENTER is on its way. Distinct from the
    # stop sound so the user can tell "sent" from "stopped" without looking.
    def play_send_phrase_sound(self):
        if self.enabled and self.send_phrase_sound_path:
            self._play_sound_file_async(self.send_phrase_sound_path)

    # Played once when startup finishes. Model loading can take a while on a cold
    # start and the app lives in the tray with no window to watch, so an audible
    # 'ready' is the clearest signal that the hotkey is now live.
    def play_ready_sound(self):
        if self.enabled and self.ready_enabled:
            self._play_sound_file_async(self.ready_sound_path)
