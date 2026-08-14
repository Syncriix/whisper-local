# Using Whisper Local without HuggingFace access

Whisper Local downloads its speech model once, from `huggingface.co`. After that
it never touches the network again — transcription is fully local.

But many corporate networks block `huggingface.co`, or require it to go through
security review first. This guide covers every way to get the model onto such a
machine.

**Important:** hosting your own copy of the model *on HuggingFace* does **not**
help. If the block is on the `huggingface.co` domain, your repository is on that
same domain and is blocked identically. Pick one of the options below instead.

---

## Option 1 — Export from one machine, import on the other (recommended)

Best when you have a personal machine with internet and a work machine without.
No admin rights or IT involvement needed.

**On a machine WITH internet** (run Whisper Local once first so the model is
downloaded):

```bash
whisper-local --export-model
```

That writes to your Desktop — four files, about 148 MB for the default `base`
model. Copy the resulting `whisper-local-model-base` folder to a USB stick, or
anywhere the offline machine can read.

To send it somewhere else, pass a folder that exists on your machine:

```bash
whisper-local --export-model E:\
```

**On the offline machine**, point at the folder you copied over:

```bash
whisper-local --import-model "E:\whisper-local-model-base"
```

That installs the model into your app-data folder, registers it, and makes it
active. Restart Whisper Local (tray → Restart) and you're done. The app will
never try to reach HuggingFace for it again.

## Option 2 — One shared copy on a network drive

Best when IT can host the model once for a whole team. Nothing is copied per
machine — every user points at the same folder.

Put an exported model folder on a share, then on each machine:

```bash
whisper-local --import-model "\\fileserver\tools\whisper-model-base" --keep-in-place
```

`--keep-in-place` registers the share path directly instead of copying it. UNC
paths and mapped drives both work.

## Option 3 — Point at an internal HuggingFace mirror

Best when your company already runs an artifact proxy (Artifactory, Nexus, or a
self-hosted HF mirror). Set one environment variable and the normal download
works through it:

```bash
setx HF_ENDPOINT https://hf-mirror.your-company.com
```

Whisper Local doesn't implement this itself — it's honoured by the underlying
`huggingface_hub` library, so it applies to the model download as a whole.

## Option 4 — Ask IT to allowlist the exact download

If you'd rather get the domain approved, this is the minimum needed. It is a
plain HTTPS GET of static files, no account or API key:

| | |
|---|---|
| Host | `huggingface.co` (plus `cdn-lfs.huggingface.co` for the weights) |
| Repo | `Systran/faster-whisper-base` (or the model you choose) |
| Files | `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt` |
| Size | ~148 MB for `base` |
| When | Once, on first run. Never again. |

Nothing is uploaded, and no audio or text ever leaves the machine — before,
during, or after.

---

## Verifying you're actually offline

```bash
whisper-local --doctor
```

Look for `Model '<name>' cached` under **Whisper model cache**. If that's green,
no download will be attempted.

To prove it, set `HF_HUB_OFFLINE=1` in the environment and start the app — if it
transcribes, it isn't reaching the network for anything.

## Licensing

The Whisper models are MIT-licensed by OpenAI, and the CTranslate2 conversions
used here are MIT-licensed by Systran. Copying them onto company machines or an
internal share is permitted. (Redistributing them publicly is also allowed —
it just doesn't solve a domain block, per the note at the top.)
