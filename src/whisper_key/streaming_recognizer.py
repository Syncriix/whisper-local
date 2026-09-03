# streaming_recognizer.py
# Wraps a sherpa-onnx streaming ASR model to produce partial transcripts while
# the user is still speaking. Powers the overlay's live preview and the opt-in
# type-as-you-speak delivery. Separate from whisper_engine because it optimises
# for latency (small model, incremental decode) rather than final accuracy.
import logging
import os
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soxr

if TYPE_CHECKING:
    from .model_registry import ModelRegistry

SHERPA_SAMPLE_RATE = 16000


# Upper-case, collapse whitespace, drop empties and duplicates, keep order.
def normalize_hotwords(phrases) -> list:
    out = []
    for p in phrases or []:
        if not p:
            continue
        p = " ".join(str(p).split()).upper()
        if p and p not in out:
            out.append(p)
    return out


# sherpa wants the BPE vocabulary as text ("piece<TAB>score" per line) but the
# models ship only the binary bpe.model. Export it once beside the model and
# reuse it; return None when the model has no BPE or sentencepiece is missing.
def ensure_bpe_vocab(model_path: str):
    vocab = os.path.join(model_path, "bpe.vocab")
    if os.path.exists(vocab):
        return vocab
    model = os.path.join(model_path, "bpe.model")
    if not os.path.exists(model):
        return None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model)
        with open(vocab, "w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
        return vocab
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not export BPE vocab from {model}: {e}")
        return None


class StreamingRecognizer:
    def __init__(self, model_type: str = "standard", recording_rate: int = 16000,
                 model_registry: "ModelRegistry" = None,
                 hotwords: list = None, hotwords_score: float = 1.5):
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.recording_rate = recording_rate
        self.model_registry = model_registry
        self.hotwords = list(hotwords or [])
        self.hotwords_score = float(hotwords_score)
        self.recognizer = None
        self.stream = None

    def load_model(self) -> bool:
        try:
            import sherpa_onnx
        except ImportError:
            self.logger.warning("sherpa-onnx not installed, streaming recognition unavailable")
            return False

        if not self.model_registry:
            self.logger.warning("No model registry provided for streaming recognizer")
            return False

        result = self.model_registry.get_streaming_model_path(self.model_type)
        if not result:
            self.logger.warning(f"Streaming model '{self.model_type}' not available")
            return False

        model_path, files = result

        encoder = os.path.join(model_path, files["encoder"])
        decoder = os.path.join(model_path, files["decoder"])
        joiner = os.path.join(model_path, files["joiner"])
        tokens = os.path.join(model_path, files["tokens"])

        for name, f in [("encoder", encoder), ("decoder", decoder), ("joiner", joiner), ("tokens", tokens)]:
            if not os.path.exists(f):
                self.logger.warning(f"Missing streaming model file: {name}")
                return False

        # Hotword biasing needs beam search plus the model's BPE vocabulary to
        # turn the phrases into token sequences. Without bpe.model we stay on
        # greedy search (faster) and say so once.
        base = dict(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=SHERPA_SAMPLE_RATE,
            feature_dim=80,
            provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
        )

        if self.hotwords:
            bpe_vocab = ensure_bpe_vocab(model_path)
            if bpe_vocab:
                try:
                    self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                        **base,
                        decoding_method="modified_beam_search",
                        hotwords_file=self._write_hotwords_file(),
                        hotwords_score=self.hotwords_score,
                        modeling_unit="bpe",
                        bpe_vocab=bpe_vocab,
                    )
                    self.logger.info(f"Streaming hotwords active ({len(self.hotwords)}): {self.hotwords}")
                except Exception as e:
                    # A bad hotword setup must never take the app down; the
                    # live preview still works, just without the bias.
                    self.logger.warning(f"Hotwords rejected by sherpa-onnx, using greedy search: {e}")
                    self.recognizer = None
            else:
                self.logger.warning(
                    f"Streaming model '{self.model_type}' has no BPE vocabulary; hotwords ignored")

        if self.recognizer is None:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                **base, decoding_method="greedy_search")

        self.stream = self.recognizer.create_stream()
        self.logger.info(f"Streaming recognizer loaded: {self.model_type}")
        return True

    # sherpa reads hotwords from a file, one phrase per line, in the model's
    # casing (these English zipformers emit upper case).
    def _write_hotwords_file(self) -> str:
        from .utils import get_user_app_data_path
        path = os.path.join(get_user_app_data_path(), "streaming_hotwords.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(normalize_hotwords(self.hotwords)) + "\n")
        return path

    # Warmup required to work-around clipping of first speech detected
    def warmup(self) -> bool:
        if not self.is_loaded():
            return False

        assets_dir = Path(__file__).parent / "assets" / "sounds"
        warmup_file = assets_dir / "streaming-recognizer-warmup.wav"

        if not warmup_file.exists():
            self.logger.warning(f"Warmup audio file not found: {warmup_file}")
            return False

        with wave.open(str(warmup_file), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio * (0.8 / max_val)

        if sample_rate != SHERPA_SAMPLE_RATE:
            audio = soxr.resample(audio, sample_rate, SHERPA_SAMPLE_RATE).astype(np.float32)

        chunk_size = 1600
        for i in range(0, len(audio), chunk_size):
            self.stream.accept_waveform(SHERPA_SAMPLE_RATE, audio[i:i + chunk_size])
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

        result = self.recognizer.get_result(self.stream).strip()
        self.logger.info(f"Warmup complete, recognized: '{result}'")
        self.recognizer.reset(self.stream)
        return True

    def is_loaded(self) -> bool:
        return self.recognizer is not None and self.stream is not None

    def process_chunk(self, audio_chunk: np.ndarray) -> None:
        if not self.is_loaded():
            return

        samples = audio_chunk.flatten().astype(np.float32)

        if self.recording_rate != SHERPA_SAMPLE_RATE:
            samples = soxr.resample(samples, self.recording_rate, SHERPA_SAMPLE_RATE).astype(np.float32)

        self.stream.accept_waveform(SHERPA_SAMPLE_RATE, samples)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def get_partial_result(self) -> str:
        if not self.is_loaded():
            return ""
        return self.recognizer.get_result(self.stream).strip()

    def is_endpoint(self) -> bool:
        if not self.is_loaded():
            return False
        return self.recognizer.is_endpoint(self.stream)

    def reset(self) -> None:
        if not self.is_loaded():
            return
        # sherpa's reset(stream) leaves decode-state residue behind (observed:
        # junk tokens prepended after a reset, compounding over hundreds of
        # endpoint resets into full gibberish). A fresh stream is cheap and
        # residue-free by construction, so never reuse one across utterances.
        self.stream = self.recognizer.create_stream()

    def set_recording_rate(self, rate: int) -> None:
        self.recording_rate = rate
