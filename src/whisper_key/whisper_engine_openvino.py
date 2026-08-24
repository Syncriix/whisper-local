# whisper_engine_openvino.py
# Opt-in Intel GPU backend (pip install 'whisper-local[openvino]'), selected via
# whisper.backend: openvino. Runs pre-converted Whisper models through
# openvino_genai.WhisperPipeline on Intel iGPU/Arc ("GPU"), CPU, or NPU.
# Mirrors WhisperEngine's public API so the rest of the app is backend-agnostic.

import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .utils import get_user_app_data_path

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# Pre-converted OpenVINO IR models published by the OpenVINO org on HuggingFace
# (https://huggingface.co/OpenVINO). Repo pattern: OpenVINO/{name}-{prec}-ov.
# The app's distil-* model keys have no OpenVINO twin and are rejected with a
# clear message rather than silently falling back to something else.
_SUPPORTED_MODELS = {
    "tiny": "whisper-tiny",
    "base": "whisper-base",
    "small": "whisper-small",
    "medium": "whisper-medium",
    "large": "whisper-large-v3",  # 'large' means large-v3, same as faster-whisper
    "large-v3-turbo": "whisper-large-v3-turbo",
    "tiny.en": "whisper-tiny.en",
    "base.en": "whisper-base.en",
    "small.en": "whisper-small.en",
    "medium.en": "whisper-medium.en",
}

# int8 is the recommended default (measured on Arc 140T: same accuracy as the
# CT2 engine, RTF 0.09). float32 has no IR variant upstream, so it gets fp16.
_PRECISION_SUFFIX = {
    "int8": "int8",
    "float16": "fp16",
    "float32": "fp16",
}

# Approximate int8 download sizes (fp16 is roughly double), shown before the
# first-run download so the wait is expected, not mysterious.
_MODEL_SIZE_HINTS = {
    "tiny": "~60 MB",
    "base": "~100 MB",
    "small": "~300 MB",
    "medium": "~790 MB",
    "large": "~1.6 GB",
    "large-v3-turbo": "~850 MB",
}

# Config `device` values → OpenVINO device strings. `cuda` is tolerated so a
# user switching backend with a leftover NVIDIA/AMD config still lands on GPU.
_DEVICE_MAP = {
    "gpu": "GPU",
    "cpu": "CPU",
    "npu": "NPU",
    "auto": "AUTO",
    "cuda": "GPU",
}


class WhisperEngineOpenVino:
    def __init__(self,
                 model_key: str = "base",
                 device: str = "gpu",
                 compute_type: str = "int8",
                 language: Optional[str] = None,
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: Optional[list] = None,
                 task: str = "transcribe",
                 vad_manager=None,
                 model_registry=None,
                 log_transcriptions: bool = False):

        try:
            import openvino_genai
        except ImportError as e:
            raise RuntimeError(
                "OpenVINO backend requested but openvino-genai is not installed. "
                "Install with: pip install 'whisper-local[openvino]' "
                "or pip install openvino-genai"
            ) from e

        self._genai = openvino_genai
        self.model_key = model_key
        self.device = _DEVICE_MAP.get((device or "gpu").lower(), "GPU")
        self.compute_type = compute_type
        self.language = None if language in (None, "auto") else language
        self.initial_prompt = initial_prompt or None
        self.hotwords = ", ".join(hotwords) if hotwords else None
        self.task = task if task in ("transcribe", "translate") else "transcribe"
        self.vad_manager = vad_manager
        self.registry = model_registry
        self.log_transcriptions = log_transcriptions
        self.logger = logging.getLogger(__name__)
        self.pipeline = None

        # Upstream limitation: num_beams >= 2 crashes WhisperPipeline
        # (openvino.genai#2069), so this backend always decodes greedily.
        if beam_size and beam_size > 1:
            self.logger.info(
                f"OpenVINO backend uses greedy decoding; ignoring beam_size={beam_size}"
            )

        # Compiled-model cache: first-ever GPU compile takes ~15s on an Arc
        # 140T, later launches ~2s. Kept in the app dir so it survives venv
        # rebuilds but goes away with an app-data reset.
        self._compile_cache_dir = os.path.join(get_user_app_data_path(), "openvino_cache")
        os.makedirs(self._compile_cache_dir, exist_ok=True)

        self._load_model()

    # ------------------------------------------------------------------
    # Model resolution & loading
    # ------------------------------------------------------------------

    # A registry entry pointing at a local directory (e.g. from --import-model)
    # wins over the HuggingFace catalog; otherwise the key must be in the
    # supported table.
    def _resolve_model_dir(self, model_key: str) -> str:
        if self.registry:
            definition = self.registry.get_model(model_key)
            if definition and definition.is_local_path:
                return definition.source

        base_name = _SUPPORTED_MODELS.get(model_key)
        if not base_name:
            supported = ", ".join(sorted(_SUPPORTED_MODELS))
            raise RuntimeError(
                f"Model '{model_key}' has no pre-converted OpenVINO variant. "
                f"Supported models for the openvino backend: {supported}"
            )

        precision = _PRECISION_SUFFIX.get(self.compute_type, "int8")
        repo_id = f"OpenVINO/{base_name}-{precision}-ov"
        return self._download_if_needed(repo_id, model_key)

    def _download_if_needed(self, repo_id: str, model_key: str) -> str:
        from huggingface_hub import snapshot_download

        # local_files_only succeeds iff the snapshot is already complete in the
        # HF cache — the cheap way to know whether to warn about a download.
        try:
            return snapshot_download(repo_id, local_files_only=True)
        except Exception:
            pass

        size = _MODEL_SIZE_HINTS.get(model_key.split(".")[0], "")
        size_note = f" ({size})" if size else ""
        print(f"⬇  Downloading the '{model_key}' OpenVINO model{size_note} — "
              "first run only. This can take a few minutes.")
        return snapshot_download(repo_id)

    def _load_model(self):
        print(f"🧠 Loading OpenVINO Whisper model [{self.model_key}]...")
        try:
            model_dir = self._resolve_model_dir(self.model_key)

            # An empty compile cache means OpenVINO will compile the model for
            # this device now — a one-time wait worth announcing.
            cache_is_cold = not os.path.isdir(self._compile_cache_dir) or \
                not os.listdir(self._compile_cache_dir)
            if cache_is_cold and self.device != "CPU":
                print(f"   ⏳ Preparing model for {self.device} — first launch only, "
                      "this can take a minute...")

            self.pipeline = self._genai.WhisperPipeline(
                model_dir, self.device, CACHE_DIR=self._compile_cache_dir
            )
            self._generation_config = self._build_generation_config()

            print(f"   ✓ OpenVINO model [{self.model_key}] ready!")
            print(f"   ✓ Running on {self.device} with {self.compute_type} precision")
            self._warmup()

        except Exception as e:
            self.logger.error(f"Failed to load OpenVINO Whisper model: {e}")
            raise

    def _build_generation_config(self):
        config = self.pipeline.get_generation_config()
        if self.language:
            # WhisperPipeline expects Whisper's token form, e.g. "<|de|>"
            config.language = f"<|{self.language}|>"
        config.task = self.task
        if self.initial_prompt:
            config.initial_prompt = self.initial_prompt
        if self.hotwords:
            config.hotwords = self.hotwords
        config.num_beams = 1
        return config

    # First generate after construction runs ~2.5x slower than warm (measured);
    # absorbing it here means the user's first real dictation is already fast.
    # Also triggers the device compile at startup rather than mid-workflow.
    def _warmup(self):
        try:
            silent = np.zeros(16000, dtype=np.float32)
            t0 = time.time()
            self.pipeline.generate(silent, self._generation_config)
            self.logger.info(f"OpenVINO warmup completed in {time.time() - t0:.2f}s")
        except Exception as e:
            self.logger.debug(f"OpenVINO warmup skipped: {e}")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    # Mono float32 @ 16 kHz in, plain text out (None when there's no speech) —
    # the same contract as WhisperEngine.transcribe_audio. The VAD pre-check is
    # load-bearing here: WhisperPipeline has no no_speech filtering at all and
    # reliably hallucinates ("you", "Thank you.") on silent input.
    def transcribe_audio(self, audio_data: Optional[np.ndarray]) -> Optional[str]:
        if self.pipeline is None:
            return None

        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        try:
            if self.vad_manager and self.vad_manager.is_available():
                if not self.vad_manager.check_audio_for_speech(audio_data):
                    print("   ✗ No speech detected, skipping transcription")
                    return None

            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            audio_data = audio_data.astype(np.float32)

            start_time = time.time()
            result = self._generate_with_watchdog(audio_data)
            if result is None:
                return None

            transcribed_text = " ".join(result.texts).strip() if result.texts else ""

            transcription_time = time.time() - start_time
            print(f"   ✓ Transcription completed in {transcription_time:.1f} seconds")

            if self.log_transcriptions:
                self.logger.info(f"Transcribed: '{transcribed_text}'")
            else:
                self.logger.info(f"Transcribed {len(transcribed_text)} chars")

            if transcribed_text:
                print(f"   ✓ Transcribed: '{transcribed_text}'")
                return transcribed_text
            self.logger.info("Transcription was empty")
            return None

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}", exc_info=True)
            print(f"❌ Transcription failed: {e}")
            return None

    # generate() can hang indefinitely on rare audio (openvino.genai#1950) and
    # offers no cancellation API, so it runs on a daemon worker with a timeout.
    # On timeout the worker thread is abandoned (it holds no Python locks); the
    # user gets an actionable message instead of a frozen dictation session.
    def _generate_with_watchdog(self, audio_data: np.ndarray):
        clip_seconds = len(audio_data) / 16000.0
        timeout = max(60.0, 4.0 * clip_seconds)

        result_box = {}

        def _run():
            try:
                result_box["result"] = self.pipeline.generate(
                    audio_data, self._generation_config
                )
            except Exception as e:  # surfaced below on the caller's thread
                result_box["error"] = e

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=timeout)

        if worker.is_alive():
            self.logger.error(
                f"OpenVINO generate() timed out after {timeout:.0f}s "
                f"(clip {clip_seconds:.1f}s) — known upstream hang, openvino.genai#1950"
            )
            print("❌ Transcription timed out. If this repeats, restart the app "
                  "or set whisper.device: cpu in your settings.")
            return None

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    # Synchronous like the whisper.cpp backend: OpenVINO loads are seconds, not
    # the multi-minute CT2 downloads that justified WhisperEngine's async path.
    def change_model(self,
                     new_model_key: str,
                     progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        if new_model_key == self.model_key:
            if progress_callback:
                progress_callback("Model already loaded")
            return False
        try:
            if progress_callback:
                progress_callback(f"Loading OpenVINO model {new_model_key}...")
            model_dir = self._resolve_model_dir(new_model_key)
            self.pipeline = self._genai.WhisperPipeline(
                model_dir, self.device, CACHE_DIR=self._compile_cache_dir
            )
            self._generation_config = self._build_generation_config()
            self.model_key = new_model_key
            self._warmup()
            if progress_callback:
                progress_callback(f"Model {new_model_key} ready")
            return True
        except Exception as e:
            self.logger.error(f"Model change failed: {e}")
            if progress_callback:
                progress_callback(f"Model change failed: {e}")
            return False

    def is_loading(self) -> bool:
        return False
