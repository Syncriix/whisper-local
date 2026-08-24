# Intel GPU Support: OpenVINO Backend

As a *user with an Intel Arc GPU (e.g. Arc 140T iGPU in Core Ultra 200H laptops)* I want **GPU-accelerated transcription via an OpenVINO backend** so I can run the `medium` model at comfortable speed instead of being stuck with `tiny`/`base` on CPU.

## Background

faster-whisper (CTranslate2) supports only CUDA on GPU. The AMD path in this
project works by swapping in a custom CTranslate2 ROCm wheel — that trick has no
Intel equivalent: CTranslate2 has no SYCL backend and no one is building one
(OpenNMT/CTranslate2#2032 sits unanswered). ZLUDA is AMD-only, PyTorch XPU costs
a 640 MB wheel for ~2.2x CPU, pywhispercpp ships CPU-only wheels, and DirectML is
deprecated with known whisper-medium failures on Intel iGPUs.

The viable path is **OpenVINO GenAI's `WhisperPipeline`**:

- `pip install openvino-genai` — torch-free, ~81 MB compressed, Windows wheels,
  Python 3.10–3.14. Runtime needs only the standard Intel graphics driver
  (no oneAPI, no Level Zero install).
- Pre-converted IR models on HF under the `OpenVINO` org (e.g.
  `OpenVINO/whisper-medium-int8-ov`, ~784 MB) — no torch needed for conversion.
- Takes 16 kHz mono float32 numpy directly (exactly what `audio_recorder.py`
  produces), auto-chunks >30 s audio, supports `initial_prompt` + `hotwords`,
  language forcing/auto-detect, and the translate task.
- Arc 140T = Arrow Lake-H Xe-LPG+ **with XMX** (~77 int8 TOPS), officially
  supported since OpenVINO 2025.0. Extrapolated: a 30 s utterance in ~5–8 s
  with medium-int8. **No published measurement exists — hence the spike phase.**

The repo already has the seam: `whisper.backend` config selects the engine in
`main.py:setup_whisper_engine()`, and `whisper_engine_cpp.py` proves the
mirror-the-API pattern. This plan adds a third engine the same way.

### Hard constraints discovered in research (design inputs, not options)

| Constraint | Source | Consequence |
|---|---|---|
| No `no_speech_prob` / temperature fallback / internal VAD | WhisperGenerationConfig has none | Our TEN-VAD pre-check is the only silence defense (already in the engine contract) |
| Beam search broken (`num_beams>=2` crashes) | openvino.genai#2069 | Greedy only; ignore `beam_size` config, log once |
| Rare unfixed `generate()` hang, no cancel API | openvino.genai#1950 | Watchdog thread + timeout around every generate |
| `openvino`/`-genai`/`-tokenizers` are ABI-locked | GenAI dependency docs, #407 | Pin `openvino-genai` exactly; verify pip resolves the matched trio |
| First GPU compile takes minutes, seconds after | #1992, Audacity plugin docs | `CACHE_DIR` property + one-time "preparing model" message; warmup at startup absorbs it |
| GPU plugin has history of silently-wrong output between releases | openvino#29131, #37419 | `device: cpu` stays a documented escape hatch; canary check in spike |

## Implementation Plan

1. **Spike: measure medium on the Arc 140T (GATE — do this before any app code)**
- [x] Write `.temp/openvino-spike.py`: throwaway venv, `pip install openvino-genai`, `huggingface_hub.snapshot_download("OpenVINO/whisper-medium-int8-ov")`, transcribe a real ~15 s speech WAV on `"GPU"`; print available devices, cold-compile time, warm time over 3 runs, and the transcript
  - ✅ Spike script + 17.1 s Windows-TTS test WAV with known text; venv in `.temp/ov-spike-venv`
- [x] Sanity-check transcript correctness (the silent-wrong-output failure mode) and behavior on 1 s of silence
  - ✅ Transcript essentially perfect on GPU, CPU, and faster-whisper (one TTS-artifact word)
  - ✅ Silence → hallucinates `'you'` on both devices (expected; TEN-VAD pre-check is the defense)
- [x] Record numbers in this plan's Status section
- [x] **Gate: PASSED** — GPU RTF 0.09, 5.4x faster than the current faster-whisper CPU engine (see Status)

2. **Engine: `whisper_engine_openvino.py`** ✅ complete, smoke-tested on the 140T
- [x] New class `WhisperEngineOpenVino` mirroring `WhisperEngine`'s public API exactly: same `__init__` signature, `transcribe_audio()`, `change_model()`, `is_loading()` (see Implementation Details)
- [x] Model-key → HF repo mapping table (`medium` → `OpenVINO/whisper-medium-int8-ov` etc.), `compute_type` selecting int8 vs fp16 repo variant; unmapped keys raise with a list of supported ones
  - ✅ Verified upstream catalog: ALL standard keys exist incl. multilingual `small` (flagged gap closed); only the app's three `distil-*` keys have no OpenVINO twin → clean error
- [x] Download via `huggingface_hub.snapshot_download` with the same "first run only" size-hint messaging as the other engines; resolve to the local snapshot dir
  - ✅ Cache detection via `local_files_only=True` probe; registry local-path entries win over the catalog (readies phase 4)
- [x] Build `WhisperPipeline(model_dir, device, CACHE_DIR=<appdata>/openvino_cache)`; print a one-time "Preparing model for GPU — first launch can take a few minutes" when the compile cache is cold
- [x] Generation config: language (`auto` → autodetect), task, `initial_prompt`, `hotwords` (join list like `WhisperEngine` does); clamp to greedy decoding regardless of `beam_size`
  - ✅ Fields verified live on `WhisperGenerationConfig` (2026.3.0); language passed in `<|xx|>` token form
- [x] Watchdog: run `generate()` on a worker thread, `join(timeout)`; on timeout log, print an actionable error (suggest restart / `device: cpu`), return None
- [x] `_warmup()` with 1 s silence so the GPU compile happens at startup, not on the first real dictation
- [x] `change_model()`: synchronous rebuild with progress callbacks (mirror `WhisperEngineCpp`)
  - ✅ Smoke test: medium→base switch incl. fresh download worked; base transcribes in 0.4 s
  - ⚠️ New finding: with `hotwords` configured, silence hallucinates the hotwords themselves ("OpenVINO, Whisper") — VAD pre-check is even more load-bearing than assumed

3. **Wiring: config, main, packaging** ✅ complete
- [x] `main.py:setup_whisper_engine()`: `backend == 'openvino'` branch, identical kwargs to the other two
- [x] Map config `device` for this backend: `gpu`→`"GPU"`, `cpu`→`"CPU"`, `npu`→`"NPU"`, `auto`→`"AUTO"`; treat a leftover `cuda` as `gpu`
  - ✅ Lives in the engine (`_DEVICE_MAP`), keeping main.py backend-agnostic
- [x] `config.defaults.yaml`: extend the `backend:` comment block with `openvino` and document the device values it accepts
- [x] `pyproject.toml`: `openvino = ["openvino-genai==2026.3.0.0"]` extra; confirm the resolved `openvino`/`openvino-tokenizers` versions match exactly (ABI lockstep)
  - ✅ Observed pip resolve the matched 2026.3.0 trio during the spike install

4. **Offline model transfer** ✅ complete, round-trip tested
- [x] `model_transfer.py`: recognize an OpenVINO IR snapshot (`openvino_encoder_model.xml/.bin`, `openvino_decoder_model.xml/.bin`, tokenizer files) alongside the CT2 `model.bin` check so `--export-model` / `--import-model` work for `OpenVINO/*-ov` models — the HF-blocked-network story is a headline feature of this fork and must not silently exclude the new backend
  - ✅ Format detection (`_detect_model_format`) covers both; export is backend-aware — with `backend: openvino` it exports the OpenVINO IR cache entry (folder derived from the engine's own catalog), otherwise CT2 as before
  - ✅ Round trip verified with scratch APPDATA: export 21 files / 784 MB (symlinks resolved), import registers `local-medium`, bogus folders rejected with a two-format message
- [x] `--import-model` writes the local dir into config the same way; engine accepts a local path as model source
  - ✅ Engine loaded the imported folder via the registry local-path branch and transcribed at full GPU speed (1.5 s)

5. **Docs & diagnostics** ✅ complete
- [x] `gpu-setup.md`: new **"Intel (OpenVINO)"** section — requirements (Intel iGPU/Arc + current graphics driver, nothing else), pip/pipx install of the extra, config snippet (`backend: openvino`, `device: gpu`, `compute_type: int8`), first-launch compile note, limitations (greedy decoding, `device: cpu` fallback), measured performance from the spike
  - ✅ Plus a pointer from the `device: cuda` intro note so Intel users don't follow the NVIDIA/AMD path
- [x] `doctor.py`: under the existing backend check, when `backend: openvino` report import/version and `openvino.Core().available_devices`
  - ✅ Also fails loudly when the configured gpu/npu device is absent from the device list (stale/missing driver) — verified live: reports 2026.3.0.0 and CPU, GPU, NPU
- [x] `docs/project-index.md`: component table row; `CHANGELOG.md` entry under `[Unreleased]`

6. **Onboarding auto-detect (last, explicitly cuttable)** ✅ complete
- [x] `platform/windows/gpu.py`: detect Intel GPUs (Win32_VideoController match on Intel Arc/Iris/UHD) when no NVIDIA/AMD card is found; new class `intel`
  - ✅ Classified only for Arc/Iris names (old UHD iGPUs are too slow to promote); runtime probe = openvino-genai importable + `'GPU'` in `available_devices` (catches stale drivers); verified live on the 140T: full System-check banner prints, returns `('intel', ..., True)`
- [x] `onboarding.py`: `intel` prompt variant — installs the pinned `openvino-genai`, sets `backend: openvino` + `device: gpu` + `compute_type: int8`; skip the CT2-specific checks for this class
  - ✅ `_enable_gpu_config()` centralizes the per-vendor config flip; install prompt offers "install OpenVINO — 90 MB download, 240 MB disk" (measured); `INTEL_PACKAGES` pin carries a keep-in-sync note pointing at pyproject
  - ✅ Beyond plan: the CUDA-failure recovery dialog now also covers openvino engine failures on `device: gpu` (re-run setup or continue on CPU — the openvino backend runs fine on CPU)
  - ⚠️ Interactive prompt paths not exercised (need a console); covered by the phase-7 manual test — reset `onboarding: gpu: pending` to see the prompt
- [x] macOS mirror stays a no-op (already is)

7. **Verification**
- [x] Unit tests: model-key mapping, import-error message when the extra isn't installed, device-string mapping (no openvino needed in CI — mirror the lean-env skip pattern from the model-transfer tests)
  - ✅ 10 new tests (catalog, device map, unsupported-model error, config docs, Intel classification, onboarding config flip, IR format detection, OV import round-trip, cache-folder derivation); full suite: 157 tests, 0 failures
- [x] Startup check with backend unset (no regression) — run natively (not WSL) against a scratch APPDATA: default config boots to "Whisper Local ready!" on CPU
  - ✅ Second launch with `backend: openvino`, `device: gpu`, `model: medium`: full app boots to ready — engine load, tray, hotkeys all live on the Arc GPU
- [x] **Manual test on the 140T machine (user):** full dictation loop on `medium` — hotkey, speak, text at cursor; model switch via tray; silence press; a >30 s dictation
  - ✅ Live dictation into VS Code with the user's real config: 2.6 s and 45.0 s utterances delivered complete; tray switch medium→base→medium with dictations on each
  - ✅ The first live >30 s dictation (35.3 s) EXPOSED A REAL BUG: openvino-genai's internal long-form mode drops/loops the short final window after the 30 s boundary. Fixed in-engine by splitting at quiet points into ≤30 s chunks decoded on the single-window path (commit 0843a0a, 3 regression tests); re-test at 45 s delivered the full text ending on a complete sentence
  - Silence tap not separately re-run live: the TEN-VAD short-circuit is backend-shared code, unchanged on this branch, and verified in the engine smoke test

## Outcome

Shipped. `pip install whisper-local[openvino]` + `backend: openvino`, `device: gpu`
runs Whisper medium on Intel Arc at RTF ~0.09 (5.4x the CPU baseline), with
first-launch auto-onboarding, offline model transfer, doctor checks, docs, and
a long-form fix for an upstream genai bug found during live testing.

Follow-ups (not in this branch): file the genai long-form bug upstream with the
minimal repro; watch pywhispercpp for Vulkan wheels as a potential second Intel
path; consider TEN-VAD-based cut points if RMS splitting ever misses.

## Implementation Details

### Backend selection stays dumb

`setup_whisper_engine()` grows one `elif`, config stays the single source of truth:

```yaml
whisper:
    backend: openvino   # faster_whisper | whisper_cpp | openvino
    device: gpu         # openvino: gpu (Intel iGPU/Arc), cpu, npu, auto
    compute_type: int8  # int8 (recommended) or float16 → picks the IR variant
    model: medium
```

### Engine skeleton (the non-obvious parts)

```python
# Model keys map to pre-converted IR repos — no torch, no conversion step.
_OPENVINO_REPOS = {
    # key: (int8_repo, fp16_repo)
    "medium": ("OpenVINO/whisper-medium-int8-ov", "OpenVINO/whisper-medium-fp16-ov"),
    "large":  ("OpenVINO/whisper-large-v3-int8-ov", ...),   # 'large' == large-v3, as elsewhere
    ...
}

def transcribe_audio(self, audio_data):
    # identical VAD pre-check + flatten/astype contract as the other engines
    ...
    # openvino.genai#1950: generate() can hang with no cancel API — watchdog it
    result_box = {}
    worker = threading.Thread(target=lambda: result_box.update(r=self.pipeline.generate(audio_data, self._gen_config)), daemon=True)
    worker.start()
    worker.join(timeout=self._generate_timeout(len(audio_data)))
    if worker.is_alive():
        ...  # log + user-facing error, return None
```

- **Timeout:** `max(60, 4 × clip_seconds)` — generous enough for a cold CPU
  fallback, tight enough that a hang surfaces within the user's patience.
- **Verify at implementation time** (research flagged as unconfirmed): exact
  `hotwords`/`initial_prompt` field types on `WhisperGenerationConfig`, whether
  a multilingual `small` pre-converted repo exists (if not: `small` maps to
  nothing and says so), and behavior on all-silence input.

### What this backend does NOT get

- No async model loading (mirror `WhisperEngineCpp`'s sync `change_model`) —
  the faster-whisper engine's background loader exists for big CT2 downloads;
  OpenVINO switches are rarer and the sync path is simpler.
- No beam search, no temperature fallback — upstream limitations, documented
  in gpu-setup.md rather than worked around.
- No NPU tuning — `device: npu` is passed through for the adventurous but the
  documented path is `gpu` (ARL-H NPU is 13 TOPS vs the iGPU's 77).

### Scope

| File | Changes |
|------|---------|
| `src/whisper_key/whisper_engine_openvino.py` | **new** — the engine |
| `src/whisper_key/main.py` | one `elif` in `setup_whisper_engine()` |
| `src/whisper_key/config.defaults.yaml` | backend/device comment docs |
| `pyproject.toml` | `[openvino]` extra, pinned |
| `src/whisper_key/model_transfer.py` | recognize IR snapshots |
| `src/whisper_key/doctor.py` | openvino branch in backend check |
| `docs/gpu-setup.md` | Intel section |
| `docs/project-index.md`, `CHANGELOG.md` | bookkeeping |
| `src/whisper_key/platform/windows/gpu.py`, `onboarding.py` | phase 6, cuttable |

## Success Criteria

- [ ] Spike numbers recorded; medium-int8 on 140T GPU clearly beats CPU medium
- [ ] `pip install whisper-local[openvino]` + 3-line config change = working GPU dictation on Intel, no other setup
- [ ] `medium` dictation round-trip on the 140T: speak ~15 s, text lands at cursor in a few seconds
- [ ] First launch shows the compile message once; second launch starts fast (cache hit)
- [ ] Silence press produces no hallucinated text (VAD short-circuit fires)
- [ ] Tray model switching works between mapped models; unmapped model gives a clear message, not a crash
- [ ] `--export-model` / `--import-model` round-trips an OpenVINO model between machines
- [ ] `--doctor` reports backend, version, and available OpenVINO devices
- [ ] Existing backends untouched: startup with default config unchanged (`/test-from-wsl`)

## Status

**Phase 1 complete — gate PASSED** (2026-08-24, Arc Pro 140T 16GB, driver 32.0.101.8517,
openvino/genai/tokenizers 2026.3.0 matched trio, Python 3.13.14, 17.1 s TTS clip,
whisper-medium int8):

| Engine | Device | Warm time | RTF | Notes |
|---|---|---|---|---|
| faster-whisper 1.2.1 (current app), beam 5 | CPU | 8.0–8.2 s | 0.47 | today's baseline |
| OpenVINO WhisperPipeline | CPU | 2.7–2.8 s | 0.16 | already 2.9x faster than baseline |
| OpenVINO WhisperPipeline | **GPU** | **1.4–1.6 s** | **0.09** | **5.4x faster than baseline** |

- Cold pipeline construction (first-ever compile): 15.0 s → **1.9 s** on relaunch with `CACHE_DIR`
- First `generate()` after construction: ~3.6–4.1 s, warm thereafter → startup warmup absorbs it
- Model download: 22 files, ~784 MB, resolved via `snapshot_download` into the standard HF cache
- `ov.Core().available_devices` = `['CPU', 'GPU', 'NPU']`
- Silence hallucinates `'you'` (0.3 s GPU / 1.4 s CPU) — VAD pre-check mandatory, as planned
- Beam-5 note: faster-whisper baseline ran beam 5 (app default); OpenVINO ran greedy
  (upstream limitation). Accuracy on the test clip was identical.

## Risks

| Risk | Mitigation |
|------|------------|
| Medium too slow on 140T despite XMX (no published benchmark exists) | Spike gate before any integration work |
| GPU plugin silently wrong output on this driver/release combo | Spike checks transcript correctness; `device: cpu` documented fallback |
| `openvino-genai` pin drifts out of ABI lockstep on future bumps | Single pinned extra; `/check-deps` flow catches bumps, doctor surfaces version |
| Multilingual `small` repo missing upstream | Verified in phase 2; key errors cleanly if absent |
| `generate()` hang (openvino.genai#1950) | Watchdog timeout, actionable error message |
| Windows DLL load errors in pipx venvs (#407 class) | Exact-pin trio; doctor check gives the diagnosis path |

## References

- Research briefs (this session): OpenVINO GenAI API surface, Arc 140T hardware, rejected-alternatives verification, pitfall inventory
- [OpenVINO GenAI speech recognition guide](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/)
- [OpenVINO/whisper-medium-int8-ov](https://huggingface.co/OpenVINO/whisper-medium-int8-ov)
- [openvino.genai#2069 beam search broken](https://github.com/openvinotoolkit/openvino.genai/issues/2069) · [#1950 generate hang](https://github.com/openvinotoolkit/openvino.genai/issues/1950) · [#407 tokenizers DLL](https://github.com/openvinotoolkit/openvino.genai/issues/407)
- [GenAI ABI/dependency rules](https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/genai-dependencies.html)
- Pattern precedent: `whisper_engine_cpp.py`, `docs/gpu-setup.md` AMD section
