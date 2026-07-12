import json
import logging
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_SENTENCE_END = ('.', '!', '?', '"', "'", ')', ']', ':', ';', ',', '…')

INLINE_FORMAT_REPLACEMENTS = [
    (r'\bnew paragraph\b', '\n\n'),
    (r'\bnew line\b', '\n'),
    # Trailing space baked in so absorb mode doesn't glue words together
    # ("hello comma world" → "hello, world", not "hello,world"). With absorb OFF
    # the extra space is normalized away by the cleanup pass below.
    (r'\b(?:full stop|period)\b', '. '),
    (r'\bcomma\b', ', '),
    (r'\bquestion mark\b', '? '),
    (r'\bexclamation (?:mark|point)\b', '! '),
    (r'\bcolon\b', ': '),
    (r'\bsemi[- ]?colon\b', '; '),
    (r'\bopen (?:quote|quotes)\b', ' "'),
    (r'\bclose (?:quote|quotes)\b', '" '),
    (r'\bopen paren(?:thesis)?\b', ' ('),
    (r'\bclose paren(?:thesis)?\b', ') '),
    (r'\bopen bracket\b', ' ['),
    (r'\bclose bracket\b', '] '),
    (r'\bdash\b', ' — '),
    (r'\bhyphen\b', '-'),
]


def postprocess(text: str, config: dict) -> str:
    if not text or not config:
        return text

    # Spoken editing commands ("scratch that") operate on the raw dictation flow,
    # so they run first — before any symbol/format rewriting.
    if config.get('voice_editing', False):
        text = _apply_voice_editing(text)

    if config.get('inline_formatting', False):
        text = _apply_inline_formatting(text, config)

    # Deterministic, offline symbol formatting (times / emails / URLs). Each
    # sub-toggle is off by default; only run the pass if at least one is on.
    smart_cfg = config.get('smart_formatting') or {}
    if any(smart_cfg.get(k) for k in ('times', 'emails', 'urls')):
        text = _apply_smart_formatting(text, smart_cfg)

    # User corrections (misrecognition fixes, e.g. "see translate two" →
    # "CTranslate2"). Applied late so they win over formatting, but before the
    # Ollama pass so the LLM sees already-corrected text. This is the backing
    # store for the history window's one-click "always fix this".
    replacements = config.get('replacements') or []
    if replacements:
        text = _apply_replacements(text, replacements)

    if config.get('strip_filler_words', False):
        text = _strip_fillers(text)

    if config.get('strip_trailing_period', False):
        text = _strip_trailing_period(text)

    if config.get('capitalize_first', False):
        text = _capitalize_first(text)

    if config.get('ensure_punctuation', False):
        text = _ensure_punctuation(text)

    ollama_cfg = config.get('ollama') or {}
    if ollama_cfg.get('enabled', False):
        polished = _ollama_polish(text, ollama_cfg)
        if polished:
            text = polished

    return text


def _strip_trailing_period(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return text
    trailing = text[len(stripped):]
    if stripped.endswith('.') and not stripped.endswith('..'):
        return stripped[:-1] + trailing
    return text


# Build the (pattern, replacement) list to apply. With no user config, this is
# just the built-in English map. A user can supply their own phrases via
# postprocess.inline_formatting_replacements — essential for non-English dictation
# (e.g. Polish), where Whisper won't emit the English trigger words. By default a
# user list REPLACES the English defaults; set inline_formatting_extend: true to
# append to them instead. User phrases are matched as whole, case-insensitive,
# regex-escaped words, so no regex injection or ReDoS is possible.
def _resolve_inline_replacements(config: dict):
    cfg = config or {}
    custom = cfg.get('inline_formatting_replacements') or []

    entries = []
    if not custom or cfg.get('inline_formatting_extend', False):
        entries.extend(INLINE_FORMAT_REPLACEMENTS)

    for item in custom:
        if not isinstance(item, dict):
            continue
        phrase = str(item.get('phrase', '')).strip()
        if not phrase:
            continue
        replacement = str(item.get('replacement', ''))
        entries.append((r'\b' + re.escape(phrase) + r'\b', replacement))
    return entries


def _apply_inline_formatting(text: str, config: dict = None) -> str:
    cfg = config or {}
    # When you SPEAK a cue word, Whisper also inserts its own punctuation around it
    # based on prosody (e.g. "hello comma world" → "Hello, comma, world."), so a bare
    # swap leaves artifacts ("Hello,, world."). With absorb on, each phrase also eats
    # the runs of commas/periods/whitespace hugging it, and the replacement's own
    # spacing wins — so define replacements like ", " or " → ". Off by default.
    absorb = cfg.get('inline_formatting_absorb_punctuation', False)
    for pattern, replacement in _resolve_inline_replacements(cfg):
        # Absorb only spaces/tabs/commas/periods around the cue — NOT newlines, so
        # "new paragraph"/"new line" (\n) breaks survive a following cue's absorb.
        effective = r'[ \t,.]*(?:' + pattern + r')[ \t,.]*' if absorb else pattern
        # Literal replacement via a function repl: avoids re interpreting \1, \g<>,
        # or stray backslashes in user-provided replacement strings.
        text = re.sub(effective, lambda _m, r=replacement: r, text, flags=re.IGNORECASE)
    text = re.sub(r' +([.,!?:;])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# =============================================================================
# Spoken editing commands
# =============================================================================

# "scratch that" / "delete that" / "strike that" — Wispr-style self-correction.
# Removes the command AND the clause it follows (back to the previous sentence
# terminator or newline), so "book the flight, scratch that, cancel it" →
# "cancel it". The char class stops at .!?\n, so it never eats across a prior
# sentence you meant to keep. A trailing command with nothing before it just
# disappears. Case-insensitive; a following comma/period artifact is absorbed.
# Trailing class is [ \t,.]* (NOT \s*): it must not eat the newline after the
# command, or "first scratch that\nsecond" would pull "second" up onto the first
# line — the leading scan already stops at \n, so the trailing side must too.
_VOICE_EDIT_SCRATCH = re.compile(
    r'[^.!?\n]*?\b(?:scratch|delete|strike)\s+that\b[ \t,.]*',
    flags=re.IGNORECASE,
)


def _apply_voice_editing(text: str) -> str:
    # Replace the removed clause+command with a single space (not nothing), so the
    # separator after a preceding sentence survives ("flight. scratch that go" →
    # "flight. go", not "flight.go"). Scratching the entire utterance yields "".
    cleaned = _VOICE_EDIT_SCRATCH.sub(' ', text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)         # no trailing space before a break
    cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)   # no space before punctuation
    return cleaned.strip()


# =============================================================================
# Deterministic smart formatting (times / emails / URLs)
# =============================================================================

# Known TLDs we're willing to collapse from speech. Kept deliberately small and
# common so "<word> dot <tld>" only fires on things that really look like a
# domain, not ordinary prose ("connect the dots" has no trailing TLD).
_TLDS = 'com|org|net|io|dev|co|edu|gov|ai|app|uk|us|ca|de|fr'

# "3pm" / "3 p.m." / "3:30 pm" → "3 PM" / "3:30 PM". Requires a leading digit,
# so it can't fire inside words like "spam".
# The trailing (?![A-Za-z0-9]) excludes a following letter OR digit, so "pm2.5"
# (an air-quality token) and "3 pm2" aren't mangled into a time.
_TIME_RE = re.compile(
    r'\b(\d{1,2}(?::\d{2})?)\s*([ap])\.?\s*m\.?(?![A-Za-z0-9])',
    flags=re.IGNORECASE,
)

# "john at example dot com" → "john@example.com". The "at ... dot <tld>" shape is
# a strong signal, so this rarely fires on prose ("meet me at noon" has no
# "dot <tld>"). Emails run before URLs so the domain isn't collapsed twice.
_EMAIL_RE = re.compile(
    r'\b([A-Za-z0-9][A-Za-z0-9._%+-]*)\s+at\s+([A-Za-z0-9][A-Za-z0-9.-]*)\s+dot\s+(' + _TLDS + r')\b',
    flags=re.IGNORECASE,
)

# "example dot com" → "example.com". Opt-in; can occasionally fire on
# "<word> dot <tld>" in prose, which is why it's its own toggle.
_URL_RE = re.compile(
    r'\b([A-Za-z0-9][A-Za-z0-9-]*)\s+dot\s+(' + _TLDS + r')\b',
    flags=re.IGNORECASE,
)


def _apply_smart_formatting(text: str, cfg: dict) -> str:
    if cfg.get('emails'):
        text = _EMAIL_RE.sub(lambda m: f"{m.group(1)}@{m.group(2)}.{m.group(3).lower()}", text)
    if cfg.get('urls'):
        text = _URL_RE.sub(lambda m: f"{m.group(1)}.{m.group(2).lower()}", text)
    if cfg.get('times'):
        text = _TIME_RE.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}M", text)
    return text


# =============================================================================
# User corrections (post-transcription replacements)
# =============================================================================

# Literal, whole-word, case-insensitive text corrections applied after
# transcription. Each item: {from, to, whole_word=true, case_sensitive=false,
# regex=false}. Literal replacement text is inserted verbatim (no backref
# interpretation), and a bad regex is skipped rather than crashing the pipeline.
def _apply_replacements(text: str, items: list) -> str:
    for item in items:
        if not isinstance(item, dict):
            continue
        frm = str(item.get('from', ''))
        if not frm.strip():
            continue
        to = str(item.get('to', ''))
        flags = 0 if item.get('case_sensitive', False) else re.IGNORECASE
        if item.get('regex', False):
            pattern = frm
        else:
            escaped = re.escape(frm)
            # Edge-aware boundaries, not \b…\b: \b needs a word char on the inside
            # edge, so a `from` like "C++", "C#" or ".NET" (non-word edge) would
            # never match. (?<!\w)…(?!\w) still enforces whole-word for normal
            # terms ("cat" won't hit "category") while allowing punctuation edges.
            pattern = r'(?<!\w)' + escaped + r'(?!\w)' if item.get('whole_word', True) else escaped
        try:
            text = re.sub(pattern, lambda _m, r=to: r, text, flags=flags)
        except re.error as e:
            logger.debug(f"Skipping invalid replacement {frm!r}: {e}")
    return text


def _strip_fillers(text: str) -> str:
    pattern = re.compile(
        r'\b(um|uh|erm|uhm|like|you know)\b[,]?\s*',
        flags=re.IGNORECASE,
    )
    cleaned = pattern.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned or text


def _capitalize_first(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return text
    leading = text[: len(text) - len(stripped)]
    return leading + stripped[0].upper() + stripped[1:]


def _ensure_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return text
    trailing = text[len(stripped):]
    if stripped.endswith(_SENTENCE_END):
        return text
    return stripped + '.' + trailing


def _ollama_polish(text: str, cfg: dict) -> str:
    endpoint = cfg.get('endpoint', 'http://localhost:11434').rstrip('/')
    model = cfg.get('model', 'llama3.2')
    prompt_template = cfg.get(
        'prompt',
        "Polish this dictation. Fix punctuation and capitalization only. Do not change wording or add anything. Output ONLY the polished text:\n\n{text}",
    )
    timeout = float(cfg.get('timeout', 5))

    if '{text}' in prompt_template:
        final_prompt = prompt_template.replace('{text}', text)
    else:
        final_prompt = f"{prompt_template}\n\n{text}"

    payload = {
        'model': model,
        'prompt': final_prompt,
        'stream': False,
    }

    try:
        req = urllib.request.Request(
            f"{endpoint}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        polished = (data.get('response') or '').strip()
        if polished:
            logger.debug("Ollama polish applied")
            return polished
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning(f"Ollama post-edit unavailable ({e}); using raw transcript")
    return ''
