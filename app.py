"""
Simple web UI for testing the personalized video messaging pipeline.

Drop this file into the same folder as main.py (personalized-video-messsging/python),
alongside the existing src/ folder and constants.py. Run with:

    pip install flask
    python app.py

Then open http://localhost:5000
"""
import os
import uuid
import threading
import traceback
import time
import json
import re
import shutil
import hashlib
import subprocess
import requests

from flask import Flask, render_template, request, jsonify

from src.service.LipSyncService import LipSyncProcessor
from sync.core.api_error import ApiError
from src.service.VoiceService import VoiceProcessor
from src.Processor.FileProcessor import FileProcessor

app = Flask(__name__)
application = app  # cPanel's Python App (Passenger/WSGI) looks for this name specifically

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base path when hosted under a subpath (e.g. james-harvey.com/pvm-tool).
# Primary source is Flask's request.script_root, which Passenger populates
# automatically per-request from the .htaccess PassengerBaseURI directive —
# no cPanel environment variable needed. APP_BASE_PATH env var is kept as a
# manual override/fallback only, in case script_root isn't available for
# some reason (e.g. a background thread with no request context).
APP_BASE_PATH_FALLBACK = os.environ.get('APP_BASE_PATH', '').rstrip('/')

# constants.py is optional now — only used as a local-dev fallback if present.
# On the server, constants.py doesn't need to exist at all; env vars cover everything.
try:
    import constants
except ImportError:
    constants = None

ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY') or (
    constants.ELEVENLABS_API_KEY if constants else None
)
SYNCLABS_API_KEY = os.environ.get('SYNCLABS_API_KEY') or (
    constants.SYNCLABS_API_KEY if constants else None
)

if not ELEVENLABS_API_KEY or not SYNCLABS_API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY and SYNCLABS_API_KEY must be set, either as "
        "environment variables (cPanel) or in constants.py (local dev)."
    )

file_processor = FileProcessor(ROOT_DIR)
voice_service = VoiceProcessor(ELEVENLABS_API_KEY)
lipsync_service = LipSyncProcessor(SYNCLABS_API_KEY)

# --- Simple password gate ---
# Set APP_USERNAME / APP_PASSWORD as environment variables in cPanel to enable this.
# If unset, the app runs without a login (fine for local testing, NOT fine once hosted).
APP_USERNAME = os.environ.get('APP_USERNAME')
APP_PASSWORD = os.environ.get('APP_PASSWORD')


@app.before_request
def require_login():
    if not APP_USERNAME or not APP_PASSWORD:
        return  # no credentials configured — auth disabled (local dev only)

    auth = request.authorization
    if not auth or auth.username != APP_USERNAME or auth.password != APP_PASSWORD:
        return (
            'Login required.',
            401,
            {'WWW-Authenticate': 'Basic realm="PVM Tool"'},
        )


@app.after_request
def no_cache_for_api(response):
    """
    The hosting domain's .htaccess applies a blanket 'cache everything for a
    month' rule (ExpiresDefault) meant for static webmail assets, which also
    caught our live polling endpoints — causing the browser to keep replaying
    a stale response instead of re-fetching current status. Explicitly
    override with no-store on every response so polling always gets fresh
    data, regardless of that domain-level rule.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Reuses the ANTHROPIC_API_KEY env var you already have set up for the H3 pipeline.
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fast/cheap, plenty for a short respelling task

# In-memory store of jobs so the browser can poll for status.
# job_id (our own uuid, NOT the sync.so job id) -> dict of state
JOBS = {}

# --- Persistent history (survives server restarts) ---
# Every completed/failed generation — from Generate, Batch, Grid, or Name List —
# gets appended here as one flat record, and the actual video file gets
# downloaded locally so the history doesn't depend on sync.so's result URL
# staying valid indefinitely.
MEDIA_DIR = os.path.join(ROOT_DIR, "Data", "Outputs")
HISTORY_FILE = os.path.join(ROOT_DIR, "history.json")
os.makedirs(MEDIA_DIR, exist_ok=True)

_history_lock = threading.Lock()


def find_cached_video(name: str, video: str, voice_id: str) -> dict:
    """
    Looks through persistent history for a previous COMPLETED generation with
    the same name, base video, and voice — the "identity" of a personalized
    video. Ignores tuning knobs (sync_mode, model, temperature) since those
    can be adjusted over time without changing which video is "the" video
    for that name/spokesperson combo.

    Returns the most recent match, or None if nothing found. This mirrors
    the production caching concept: if it's already been rendered, serve
    the existing file instead of generating again.
    """
    name_norm = name.strip().lower()
    matches = []
    for record in load_history():
        if record.get('status') != 'COMPLETED':
            continue
        entry = record.get('entry', {})
        if (entry.get('text', '').strip().lower() == name_norm
                and entry.get('video') == video
                and entry.get('voice_id') == voice_id
                and record.get('output_url')):
            matches.append(record)

    if not matches:
        return None
    matches.sort(key=lambda r: r.get('timestamp', 0), reverse=True)
    return matches[0]


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read history.json, starting fresh: {e}")
        return []


def save_video_locally(output_url: str, run_id: str, base_path: str = None) -> str:
    """
    Downloads the finished video from sync.so and saves it under Data/Outputs/,
    so it's still viewable later even if sync.so's own result URL eventually
    expires. Returns a local /media/... URL on success, or the original
    sync.so URL as a fallback if the download fails for any reason.
    """
    if base_path is None:
        base_path = APP_BASE_PATH_FALLBACK
    try:
        local_filename = f"{run_id}.mp4"
        local_path = os.path.join(MEDIA_DIR, local_filename)
        resp = requests.get(output_url, timeout=120)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return f"{base_path}/media/{local_filename}"
    except Exception as e:
        print(f"Failed to save video locally, falling back to sync.so URL: {e}")
        return output_url


def record_history(entry: dict, result: dict, source: str):
    """
    Appends one flat record to the persistent history file. Called for every
    single generation — whether it came from Generate, Batch, Grid, or Name
    List — so nothing is lost even across server restarts.
    """
    record = {
        'id': str(uuid.uuid4()),
        'source': source,  # 'single' / 'batch' / 'grid' / 'names'
        'timestamp': time.time(),
        'entry': entry,
        'status': result.get('status'),
        'output_url': result.get('output_url'),
        'error': result.get('error'),
        'duration_seconds': result.get('duration_seconds'),
        'phonetic_text': result.get('phonetic_text'),
        'length_bucket': result.get('length_bucket'),
        'validation': result.get('validation'),
        'slots': result.get('slots'),
        'stitched': result.get('stitched'),
        'parts': result.get('parts'),
        'render_seconds': result.get('render_seconds'),
        'stitch_seconds': result.get('stitch_seconds'),
    }

    with _history_lock:
        history = load_history()
        history.append(record)
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f)
        except Exception as e:
            print(f"Failed to write history.json: {e}")


import difflib


def validate_pronunciation(audio_bytes: bytes, intended_name: str) -> dict:
    """
    Transcribes the generated TTS audio via ElevenLabs Scribe and compares it
    against the intended name using a phonetic-ish fuzzy match (difflib's
    SequenceMatcher on lowercased text — not a true phonetic algorithm, but a
    reasonable first pass without adding a new dependency).

    Returns a dict with the transcript, a 0-1 similarity score, and a tier:
    'high' (>=0.85), 'borderline' (>=0.6), or 'low' (<0.6).

    This is a triage signal, not a verdict — Scribe can itself mishear a
    name, especially the same tricky ones TTS struggles with. Treat 'low' as
    "worth a human listening," not "definitely wrong."
    """
    try:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data={"model_id": "scribe_v1"},
            files={"file": ("audio.mp3", audio_bytes, "audio/mpeg")},
            timeout=30,
        )
        response.raise_for_status()
        transcript = response.json().get("text", "").strip()

        a = intended_name.strip().lower()
        b = transcript.strip().lower()
        similarity = difflib.SequenceMatcher(None, a, b).ratio()

        if similarity >= 0.85:
            tier = "high"
        elif similarity >= 0.6:
            tier = "borderline"
        else:
            tier = "low"

        return {"transcript": transcript, "similarity": round(similarity, 2), "tier": tier}
    except Exception as e:
        print(f"Pronunciation validation failed (non-fatal): {e}")
        return {"transcript": None, "similarity": None, "tier": "unknown", "error": str(e)}


def get_phonetic_spelling(name: str) -> str:
    """
    Uses Claude to turn a name into a plain-English phonetic respelling that
    reads more reliably through TTS (e.g. "Siobhan" -> "shi-VAWN").

    Falls back to the original name if the API key isn't set or the call fails,
    so a hiccup here never blocks the rest of the pipeline.
    """
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set — skipping phonetic respelling, using name as-is.")
        return name

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 30,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Give a simple phonetic respelling of this name for a text-to-speech "
                        "engine to pronounce correctly. Use plain lowercase English syllables "
                        "and hyphens between syllables — do NOT use capital letters to mark "
                        "stress, since TTS engines often misread short all-caps syllables (like "
                        "'EE' or 'SO') as acronyms and spell them out letter by letter instead "
                        "of pronouncing them. If a syllable is a long vowel sound, spell it out "
                        "with extra letters instead of capitalizing it (e.g. 'eefuh' not 'EE-fuh', "
                        "'sochee' not 'SO-chee'). Reply with ONLY the respelling, nothing else.\n\n"
                        f"Name: {name}"
                    ),
                }],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        phonetic = data["content"][0]["text"].strip()
        return phonetic if phonetic else name
    except Exception as e:
        print(f"Phonetic respelling failed, falling back to original name: {e}")
        return name


def classify_name_length(name: str) -> str:
    """
    Uses Claude to estimate syllable count and bucket the name into
    short / medium / long, for picking a matching pre-recorded video variant.

    Falls back to 'medium' if the API key isn't set or the call fails.
    """
    if not ANTHROPIC_API_KEY:
        return 'medium'

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 10,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Count the syllables in this name when spoken aloud naturally. "
                        "Reply with ONLY one word: 'short' for 1-2 syllables, "
                        "'medium' for 3 syllables, or 'long' for 4+ syllables. "
                        "Nothing else.\n\n"
                        f"Name: {name}"
                    ),
                }],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        bucket = data["content"][0]["text"].strip().lower()
        return bucket if bucket in ('short', 'medium', 'long') else 'medium'
    except Exception as e:
        print(f"Name length classification failed, defaulting to 'medium': {e}")
        return 'medium'


# --- Stitch mode: intro render + shared middle + outro render -> one video ---

def get_ffmpeg_exe() -> str:
    """System ffmpeg if present, otherwise the binary bundled by imageio-ffmpeg."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_clip(path: str) -> dict:
    """Reads width/height/fps/has-audio from `ffmpeg -i` output (no ffprobe needed)."""
    proc = subprocess.run([get_ffmpeg_exe(), '-hide_banner', '-i', path],
                          capture_output=True, text=True)
    info = proc.stderr
    vline = next((l for l in info.splitlines() if 'Video:' in l), '')
    size = re.search(r'(\d{2,5})x(\d{2,5})', vline)
    fps = re.search(r'([\d.]+) fps', vline)
    return {
        'width': int(size.group(1)) if size else None,
        'height': int(size.group(2)) if size else None,
        'fps': float(fps.group(1)) if fps else None,
        'has_audio': 'Audio:' in info,
    }


def download_to(url: str, path: str):
    resp = requests.get(url, timeout=180, stream=True)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def get_shared_clip(url: str) -> str:
    """Downloads the shared (nameless) middle video once and reuses the local copy.
    Cached by URL — if you replace the file at the same URL, delete Data/Outputs/shared_*.mp4."""
    key = hashlib.md5(url.encode()).hexdigest()[:16]
    path = os.path.join(MEDIA_DIR, f"shared_{key}.mp4")
    if not os.path.exists(path):
        download_to(url, path)
    return path


def normalize_clip(src: str, dst: str, W: int, H: int, fps: float):
    """Re-encodes ONE clip to the target size/fps/audio, one clip at a time so peak
    memory stays low (a three-input filter graph used ~400 MB and OOM'd a 512 MB host)."""
    cmd = [get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error', '-threads', '1',
           '-i', src,
           '-vf', (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                   f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"),
           '-af', 'aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
           '-x264-params', 'ref=1:rc-lookahead=0:bframes=0:threads=1',
           '-c:a', 'aac', '-ar', '44100', '-ac', '2', '-b:a', '192k',
           '-movflags', '+faststart', dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize failed: {proc.stderr[-600:]}")


def stitch_three(intro_path: str, middle_path: str, outro_path: str, out_path: str,
                 out_width: int = 0):
    """intro + shared middle + outro -> one MP4. Each clip is normalised to the intro's
    size/fps separately, then joined with a stream-copy concat (no re-encode, near-zero
    memory). The normalised middle is cached next to the original, so the long shared
    clip is only processed once."""
    for pth in (intro_path, middle_path, outro_path):
        i = probe_clip(pth)
        if not i['width'] or not i['has_audio']:
            raise ValueError(f"Clip {os.path.basename(pth)} is missing a video or audio stream.")
    ref = probe_clip(intro_path)
    W, H = ref['width'] - ref['width'] % 2, ref['height'] - ref['height'] % 2
    # Optional delivery size: scale everything to out_width (height keeps the intro's
    # aspect ratio). Smaller output = less memory and faster loading for end users;
    # 0 means match the intro exactly.
    if out_width and out_width < W:
        H = round(H * out_width / W / 2) * 2
        W = out_width - out_width % 2
    fps = ref['fps'] or 30

    mid_norm = f"{middle_path}.{W}x{H}x{fps}.norm.mp4"
    if not os.path.exists(mid_norm):
        tmp_mid = mid_norm + ".part.mp4"
        normalize_clip(middle_path, tmp_mid, W, H, fps)
        os.replace(tmp_mid, mid_norm)

    intro_norm = intro_path + ".norm.mp4"
    outro_norm = outro_path + ".norm.mp4"
    list_path = out_path + ".txt"
    try:
        normalize_clip(intro_path, intro_norm, W, H, fps)
        normalize_clip(outro_path, outro_norm, W, H, fps)
        with open(list_path, 'w') as f:
            for pth in (intro_norm, mid_norm, outro_norm):
                f.write("file '" + pth.replace("'", "'\\''") + "'\n")
        cmd = [get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'concat', '-safe', '0', '-i', list_path,
               '-c', 'copy', '-movflags', '+faststart', out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {proc.stderr[-600:]}")
    finally:
        for pth in (intro_norm, outro_norm, list_path):
            if os.path.exists(pth):
                os.remove(pth)


def render_entries(entries: list) -> dict:
    """Submits several lip-sync jobs together and waits for all of them.
    If the plan's concurrency limit (429) blocks a submission, that one waits
    until the others finish and is then submitted. Returns {index: poll_result}."""
    jobs, deferred = [], []
    for i, e in enumerate(entries):
        try:
            jobs.append((i, lipsync_service.process_lip_sync(e)['id']))
        except ApiError as err:
            if getattr(err, 'status_code', None) == 429:
                deferred.append((i, e))
            else:
                raise
    results = {}
    if jobs:
        for r in lipsync_service.poll_for_status(jobs, timeout=900, interval=5):
            results[r['idx']] = r
    for i, e in deferred:
        jid = lipsync_service.process_lip_sync(e)['id']
        results[i] = lipsync_service.poll_for_status([(i, jid)], timeout=900, interval=5)[0]
    return results



def execute_pipeline(entry: dict, source: str = 'single') -> dict:
    """
    Runs the voice + lipsync pipeline for a single entry and returns a result dict.
    Used by the single-generation, batch, grid, and name-list flows.

    `source` tags which flow triggered this run ('single'/'batch'/'grid'/'names'),
    purely for the persistent history log — it has no effect on the pipeline itself.
    """
    result = {'status': 'GENERATING_SPEECH'}
    started_at = time.time()

    try:
        # If "auto-select video by name length" is on, classify the name and
        # swap in the matching variant's video/segment timing before anything else.
        if entry.get('use_length_variants'):
            bucket = classify_name_length(entry['text'])
            result['length_bucket'] = bucket
            variant = entry.get('variants', {}).get(bucket)
            if variant and variant.get('video'):
                entry['video'] = variant['video']
                entry['segment_start'] = variant['segment_start']
                entry['segment_end'] = variant['segment_end']

        # If phonetic respelling is toggled on, generate speech using the respelled
        # version but keep entry['text'] (the original name) unchanged everywhere else.
        tts_entry = dict(entry)
        if entry.get('use_phonetic'):
            phonetic = get_phonetic_spelling(entry['text'])
            result['phonetic_text'] = phonetic
            tts_entry['text'] = phonetic

        # 1. Generate speech via ElevenLabs (reuses existing VoiceProcessor)
        input_audio = voice_service.generate_speech(tts_entry)

        # 1b. Optional: validate pronunciation by transcribing the TTS audio
        # and comparing it against the intended name (see the validator
        # design discussed — a triage signal, not a verdict).
        if entry.get('use_validation'):
            result['validation'] = validate_pronunciation(input_audio, entry['text'])

        # 2. Write to a temp mp3 and upload so Sync can fetch it via URL
        tmp_dir = os.path.join(ROOT_DIR, "Data", "Inputs")
        os.makedirs(tmp_dir, exist_ok=True)

        def _upload(audio_bytes):
            tmp = os.path.join(tmp_dir, f"webapp_gen_aud_{uuid.uuid4()}.mp3")
            with open(tmp, "wb") as f:
                f.write(audio_bytes)
            try:
                return file_processor.upload_file_uguu(tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        aud_url = _upload(input_audio)

        if not aud_url:
            result['status'] = 'FAILED'
            result['error'] = 'Failed to upload generated speech audio.'
            result['duration_seconds'] = round(time.time() - started_at, 1)
            record_history(entry, result, source)
            return result

        entry['audio'] = aud_url
        entry.pop('extra_slots', None)

        # 2b. Optional extra name windows (slots 2 and 3), all in the same render.
        # Each slot reuses the main audio unless it has its own take text
        # (different emphasis/intonation, e.g. "Tanya!" vs "Tanya.").
        extra = []
        for n in (2, 3):
            on = entry.get('double_name') if n == 2 else entry.get('third_name')
            st, en = entry.get(f'segment{n}_start', 0), entry.get(f'segment{n}_end', 0)
            if not on or en <= st:
                continue
            take = entry.get(f'text{n}')
            if take:
                tn = dict(tts_entry)
                tn['text'] = take
                url_n = _upload(voice_service.generate_speech(tn))
                if not url_n:
                    result['status'] = 'FAILED'
                    result['error'] = f'Failed to upload slot {n} take audio.'
                    result['duration_seconds'] = round(time.time() - started_at, 1)
                    record_history(entry, result, source)
                    return result
            else:
                url_n = aud_url
            extra.append({'audio': url_n, 'start': st, 'end': en})
        if extra:
            entry['extra_slots'] = extra
        result['slots'] = 1 + len(extra)

        # 3-STITCH. Intro + shared middle + outro: render intro and outro, then join.
        if entry.get('stitch'):
            entry.pop('extra_slots', None)
            result['slots'] = 1
            outro_audio = aud_url
            if entry.get('text_outro'):
                to = dict(tts_entry)
                to['text'] = entry['text_outro']
                outro_audio = _upload(voice_service.generate_speech(to))
                if not outro_audio:
                    raise ValueError('Failed to upload outro take audio.')

            intro_e = dict(entry, audio=aud_url)
            outro_e = dict(entry, audio=outro_audio,
                           video=entry['outro_video'],
                           segment_start=entry['outro_start'],
                           segment_end=entry['outro_end'])
            render_started = time.time()
            res = render_entries([intro_e, outro_e])
            result['render_seconds'] = round(time.time() - render_started, 1)
            if any(res.get(i, {}).get('status') != 'COMPLETED' for i in (0, 1)):
                bad = [str(res.get(i, {}).get('output_url', 'no result')) for i in (0, 1)
                       if res.get(i, {}).get('status') != 'COMPLETED']
                result['status'] = 'FAILED'
                result['error'] = 'Render failed: ' + '; '.join(bad)
            else:
                run_id = str(uuid.uuid4())
                base = entry.get('_base_path') if entry.get('_base_path') is not None else APP_BASE_PATH_FALLBACK
                intro_path = os.path.join(MEDIA_DIR, f"{run_id}_intro.mp4")
                outro_path = os.path.join(MEDIA_DIR, f"{run_id}_outro.mp4")
                download_to(res[0].get('output_url'), intro_path)
                download_to(res[1].get('output_url'), outro_path)
                middle_path = get_shared_clip(entry['middle_video'])
                final_path = os.path.join(MEDIA_DIR, f"{run_id}.mp4")
                stitch_started = time.time()
                stitch_three(intro_path, middle_path, outro_path, final_path,
                             out_width=int(entry.get('stitch_width') or 0))
                result['stitch_seconds'] = round(time.time() - stitch_started, 1)
                result['status'] = 'COMPLETED'
                result['stitched'] = True
                result['output_url'] = f"{base}/media/{run_id}.mp4"
                result['parts'] = {
                    'intro': f"{base}/media/{run_id}_intro.mp4",
                    'outro': f"{base}/media/{run_id}_outro.mp4",
                }
            result['duration_seconds'] = round(time.time() - started_at, 1)
            record_history(entry, result, source)
            return result

        # 3. Submit the lipsync job (reuses existing LipSyncProcessor, temperature-aware)
        response_json = lipsync_service.process_lip_sync(entry)
        sync_job_id = response_json['id']
        result['sync_job_id'] = sync_job_id

        # 4. Poll sync.so until it's done (reuses existing poll_for_status, single job)
        poll_results = lipsync_service.poll_for_status([(0, sync_job_id)], timeout=600, interval=5)
        poll_result = poll_results[0]

        if poll_result.get('status') == 'COMPLETED':
            result['status'] = 'COMPLETED'
            sync_output_url = poll_result.get('output_url') or poll_result.get('outputUrl')
            # Download and keep our own copy — don't depend on sync.so's URL
            # staying valid indefinitely.
            result['output_url'] = save_video_locally(
                sync_output_url, str(uuid.uuid4()), base_path=entry.get('_base_path')
            )
        else:
            result['status'] = 'FAILED'
            result['error'] = str(poll_result.get('output_url', 'Job failed or timed out.'))

    except Exception as e:
        traceback.print_exc()
        result['status'] = 'FAILED'
        result['error'] = str(e)

    result['duration_seconds'] = round(time.time() - started_at, 1)
    record_history(entry, result, source)
    return result


def run_generation(job_uuid: str, entry: dict, source: str = 'single'):
    """Runs a single generation and writes progress/result into JOBS for polling."""
    JOBS[job_uuid]['created_at'] = JOBS[job_uuid].get('created_at', time.time())
    result = execute_pipeline(entry, source=source)
    JOBS[job_uuid].update(result)
    JOBS[job_uuid]['completed_at'] = time.time()


# In-memory store of batch runs. batch_uuid -> dict with progress + list of run results.
BATCHES = {}


def run_batch(batch_uuid: str, entry: dict, count: int):
    """
    Runs the same entry `count` times, one after another (never in parallel —
    respects a 1-concurrent-job account limit). Updates BATCHES after each run
    so the frontend can show live progress and a running success ratio.
    """
    for i in range(count):
        BATCHES[batch_uuid]['current'] = i + 1
        result = execute_pipeline(dict(entry), source='batch')  # fresh copy each run
        result['run_number'] = i + 1
        BATCHES[batch_uuid]['runs'].append(result)

    BATCHES[batch_uuid]['status'] = 'DONE'


# In-memory store of grid sweeps. grid_uuid -> dict with progress + per-combo results.
GRIDS = {}


def run_grid(grid_uuid: str, entry: dict, models: list, modes: list, runs_per_combo: int):
    """
    Cycles through every (model, sync_mode) combination, running `runs_per_combo`
    generations for each — always one at a time, since the account is capped at
    1 concurrent job. Updates GRIDS after every single run for live progress.
    """
    combos = [(m, s) for m in models for s in modes]
    GRIDS[grid_uuid]['combos'] = [
        {'model': m, 'sync_mode': s, 'runs': []} for m, s in combos
    ]

    for combo_index, (model, mode) in enumerate(combos):
        for i in range(runs_per_combo):
            GRIDS[grid_uuid]['current'] += 1
            combo_entry = dict(entry)
            combo_entry['lipsync_model'] = model
            combo_entry['sync_mode'] = mode
            result = execute_pipeline(combo_entry, source='grid')
            result['run_number'] = i + 1
            GRIDS[grid_uuid]['combos'][combo_index]['runs'].append(result)

    GRIDS[grid_uuid]['status'] = 'DONE'


# In-memory store of name-list batches. batch_uuid -> dict with progress + per-name results.
NAME_BATCHES = {}


def run_names_batch(batch_uuid: str, entry: dict, names: list):
    """
    Runs the same settings across a whole list of different names, one after
    another. This is the "I've dialed in good settings, now check them against
    a real name list" step — not repeating one name, but sweeping many.
    """
    for i, name in enumerate(names):
        NAME_BATCHES[batch_uuid]['current'] = i + 1
        name_entry = dict(entry)
        name_entry['text'] = name
        result = execute_pipeline(name_entry, source='names')
        result['name'] = name
        NAME_BATCHES[batch_uuid]['runs'].append(result)

    NAME_BATCHES[batch_uuid]['status'] = 'DONE'


def build_entry_from_form(form) -> dict:
    """Builds the common entry dict shared by single/batch/grid runs."""
    return {
        'video': form.get('video', '').strip(),
        'text': form.get('text', '').strip(),
        'voice_id': form.get('voice_id', '').strip(),
        'segment_start': float(form.get('segment_start', 0)),
        'segment_end': float(form.get('segment_end', 1)),
        'double_name': form.get('double_name') == 'on',
        'segment2_start': float(form.get('segment2_start') or 0),
        'segment2_end': float(form.get('segment2_end') or 0),
        'text2': form.get('text2', '').strip(),
        'stitch': form.get('stitch') == 'on',
        'outro_video': form.get('outro_video', '').strip(),
        'outro_start': float(form.get('outro_start') or 0),
        'outro_end': float(form.get('outro_end') or 0),
        'middle_video': form.get('middle_video', '').strip(),
        'stitch_width': int(float(form.get('stitch_width') or 0)),
        'text_outro': form.get('text_outro', '').strip(),
        'third_name': form.get('third_name') == 'on',
        'segment3_start': float(form.get('segment3_start') or 0),
        'segment3_end': float(form.get('segment3_end') or 0),
        'text3': form.get('text3', '').strip(),
        'sync_mode': form.get('sync_mode', 'bounce'),
        'lipsync_model': form.get('lipsync_model', 'lipsync-2'),
        'tts_model': form.get('tts_model', 'eleven_multilingual_v2'),
        'temperature': float(form.get('temperature', 0.5)),
        'model_mode': form.get('model_mode', 'lips'),
        'emotion': form.get('emotion', ''),
        'use_phonetic': form.get('use_phonetic') == 'on',
        'use_validation': form.get('use_validation') == 'on',
        'use_length_variants': form.get('use_length_variants') == 'on',
        'variants': {
            'short': {
                'video': form.get('short_video', '').strip(),
                'segment_start': float(form.get('short_start') or 0),
                'segment_end': float(form.get('short_end') or 0),
            },
            'medium': {
                'video': form.get('medium_video', '').strip(),
                'segment_start': float(form.get('medium_start') or 0),
                'segment_end': float(form.get('medium_end') or 0),
            },
            'long': {
                'video': form.get('long_video', '').strip(),
                'segment_start': float(form.get('long_start') or 0),
                'segment_end': float(form.get('long_end') or 0),
            },
        },
    }


@app.route('/')
def index():
    return render_template('index.html', base_path=request.script_root)


@app.route('/generate-sync-diagnostic', methods=['POST'])
def generate_sync_diagnostic():
    """
    TEMPORARY diagnostic route — runs the full pipeline synchronously, inside
    this request, with NO background thread. Used to test whether hosting's
    Passenger setup is interfering with threads that outlive their parent
    request (a known issue on some managed/shared Passenger hosts). The
    browser will simply wait (up to ~1-2 min) for the full result instead of
    polling. Safe to remove once the threading question is settled.
    """
    entry = build_entry_from_form(request.form)
    entry['_base_path'] = request.script_root

    if not entry['video'] or not entry['text'] or not entry['voice_id']:
        return jsonify({'error': 'video, text, and voice_id are required.'}), 400

    result = execute_pipeline(entry, source='single')
    return jsonify(result)


@app.route('/generate', methods=['POST'])
def generate():
    entry = build_entry_from_form(request.form)
    entry['_base_path'] = request.script_root
    source = request.form.get('source', 'single')

    if not entry['video'] or not entry['text'] or not entry['voice_id']:
        return jsonify({'error': 'video, text, and voice_id are required.'}), 400
    if entry['stitch'] and (not entry['outro_video'] or not entry['middle_video']
                            or entry['outro_end'] <= entry['outro_start']):
        return jsonify({'error': 'Stitch mode needs an outro video, outro window, and middle video.'}), 400

    job_uuid = str(uuid.uuid4())
    JOBS[job_uuid] = {'status': 'STARTING', 'entry': entry, 'created_at': time.time()}

    thread = threading.Thread(target=run_generation, args=(job_uuid, entry, source), daemon=True)
    thread.start()

    return jsonify({'job_uuid': job_uuid})


@app.route('/generate-batch', methods=['POST'])
def generate_batch():
    entry = build_entry_from_form(request.form)
    entry['_base_path'] = request.script_root
    count = int(request.form.get('batch_count', 1))
    count = max(1, min(count, 20))  # sanity cap — this runs sequentially and costs real money

    if not entry['video'] or not entry['text'] or not entry['voice_id']:
        return jsonify({'error': 'video, text, and voice_id are required.'}), 400

    batch_uuid = str(uuid.uuid4())
    BATCHES[batch_uuid] = {
        'status': 'RUNNING',
        'entry': entry,
        'total': count,
        'current': 0,
        'runs': [],
        'created_at': time.time(),
    }

    thread = threading.Thread(target=run_batch, args=(batch_uuid, entry, count), daemon=True)
    thread.start()

    return jsonify({'batch_uuid': batch_uuid})


@app.route('/generate-grid', methods=['POST'])
def generate_grid():
    form = request.form
    entry = build_entry_from_form(form)
    entry['_base_path'] = request.script_root

    models = form.getlist('grid_models')
    modes = form.getlist('grid_modes')
    runs_per_combo = int(form.get('runs_per_combo', 1))
    runs_per_combo = max(1, min(runs_per_combo, 5))

    if not entry['video'] or not entry['text'] or not entry['voice_id']:
        return jsonify({'error': 'video, text, and voice_id are required.'}), 400
    if not models or not modes:
        return jsonify({'error': 'Select at least one model and one sync mode.'}), 400

    total_runs = len(models) * len(modes) * runs_per_combo
    if total_runs > 40:
        return jsonify({'error': f'That combination would trigger {total_runs} generations — narrow your selection (max 40).'}), 400

    grid_uuid = str(uuid.uuid4())
    GRIDS[grid_uuid] = {
        'status': 'RUNNING',
        'entry': entry,
        'total': total_runs,
        'current': 0,
        'combos': [],
        'created_at': time.time(),
    }

    thread = threading.Thread(
        target=run_grid, args=(grid_uuid, entry, models, modes, runs_per_combo), daemon=True
    )
    thread.start()

    return jsonify({'grid_uuid': grid_uuid})


@app.route('/grid/<grid_uuid>')
def grid_status(grid_uuid):
    grid = GRIDS.get(grid_uuid)
    if not grid:
        return jsonify({'error': 'Unknown grid run'}), 404

    response = dict(grid)
    for combo in response['combos']:
        completed = [r for r in combo['runs'] if r.get('status') == 'COMPLETED']
        combo['success_count'] = len(completed)
        combo['success_rate'] = round(len(completed) / len(combo['runs']) * 100, 1) if combo['runs'] else None
        durations = [r['duration_seconds'] for r in combo['runs'] if r.get('duration_seconds') is not None]
        combo['avg_duration'] = round(sum(durations) / len(durations), 1) if durations else None

    return jsonify(response)


@app.route('/batch/<batch_uuid>')
def batch_status(batch_uuid):
    batch = BATCHES.get(batch_uuid)
    if not batch:
        return jsonify({'error': 'Unknown batch'}), 404

    completed = [r for r in batch['runs'] if r.get('status') == 'COMPLETED']
    failed = [r for r in batch['runs'] if r.get('status') == 'FAILED']

    response = dict(batch)
    response['success_count'] = len(completed)
    response['fail_count'] = len(failed)
    response['success_rate'] = round(len(completed) / len(batch['runs']) * 100, 1) if batch['runs'] else None

    return jsonify(response)


@app.route('/generate-names-batch', methods=['POST'])
def generate_names_batch():
    form = request.form
    entry = build_entry_from_form(form)
    entry['_base_path'] = request.script_root

    raw_names = form.get('names_list', '')
    # Accept names separated by newlines or commas, ignore blank lines/whitespace.
    names = [n.strip() for n in raw_names.replace(',', '\n').splitlines() if n.strip()]
    # De-dupe while preserving order, in case the same name got pasted twice.
    seen = set()
    names = [n for n in names if not (n in seen or seen.add(n))]

    if not entry['video'] or not entry['voice_id']:
        return jsonify({'error': 'video and voice_id are required.'}), 400
    if not names:
        return jsonify({'error': 'Paste at least one name (one per line, or comma-separated).'}), 400
    if len(names) > 50:
        return jsonify({'error': f'{len(names)} names is a lot to run sequentially in one go — try 50 or fewer at a time.'}), 400

    batch_uuid = str(uuid.uuid4())
    NAME_BATCHES[batch_uuid] = {
        'status': 'RUNNING',
        'entry': entry,
        'total': len(names),
        'current': 0,
        'runs': [],
        'created_at': time.time(),
    }

    thread = threading.Thread(
        target=run_names_batch, args=(batch_uuid, entry, names), daemon=True
    )
    thread.start()

    return jsonify({'names_batch_uuid': batch_uuid})


@app.route('/names-batch/<batch_uuid>')
def names_batch_status(batch_uuid):
    batch = NAME_BATCHES.get(batch_uuid)
    if not batch:
        return jsonify({'error': 'Unknown names batch'}), 404

    completed = [r for r in batch['runs'] if r.get('status') == 'COMPLETED']
    failed = [r for r in batch['runs'] if r.get('status') == 'FAILED']

    response = dict(batch)
    response['success_count'] = len(completed)
    response['fail_count'] = len(failed)
    response['success_rate'] = round(len(completed) / len(batch['runs']) * 100, 1) if batch['runs'] else None

    return jsonify(response)


@app.route('/status/<job_uuid>')
def status(job_uuid):
    job = JOBS.get(job_uuid)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify(job)


@app.route('/jobs')
def list_jobs():
    """Returns every job run this session, newest first, for the history panel."""
    items = []
    for job_uuid, job in JOBS.items():
        item = dict(job)
        item['job_uuid'] = job_uuid
        items.append(item)
    items.sort(key=lambda j: j.get('created_at', 0), reverse=True)
    return jsonify(items)


@app.route('/media/<path:filename>')
def media(filename):
    """Serves locally-saved videos (Data/Outputs/) so history doesn't depend
    on sync.so's result URLs staying valid indefinitely."""
    from flask import send_from_directory
    return send_from_directory(MEDIA_DIR, filename)


@app.route('/history')
def history():
    """
    Returns the full persistent history — every generation ever run through
    Generate, Batch, Grid, or Name List — newest first. Survives server
    restarts, unlike the in-memory JOBS/BATCHES/GRIDS/NAME_BATCHES stores.
    """
    records = load_history()
    records.sort(key=lambda r: r.get('timestamp', 0), reverse=True)
    return jsonify(records)


@app.route('/check-cache')
def check_cache():
    """
    Used by the Customer Experience Preview: checks whether this exact
    name + video + voice combo has already been successfully rendered, so
    the preview can serve the existing file instantly instead of generating
    again — mirroring the production pre-rendered-cache concept.
    """
    name = request.args.get('name', '')
    video = request.args.get('video', '')
    voice_id = request.args.get('voice_id', '')

    if not name or not video or not voice_id:
        return jsonify({'found': False})

    match = find_cached_video(name, video, voice_id)
    if match:
        return jsonify({'found': True, 'output_url': match['output_url']})
    return jsonify({'found': False})


if __name__ == '__main__':
    # Works both locally and on Render:
    # - Locally: no PORT env var is set, so this defaults to 5000 as before.
    # - On Render: Render sets PORT itself and requires binding to 0.0.0.0
    #   (not just localhost), or the platform can't route traffic to the app.
    #
    # use_reloader=False is intentional: Flask's auto-reloader watches every
    # imported module's file, including third-party packages like syncsdk.
    # A file-timestamp touch on one of those (antivirus scan, pip cache, etc.)
    # can trigger a false restart mid-generation, wiping in-memory JOBS/BATCHES
    # state right when the browser is polling for it. Restart manually
    # (Ctrl+C, then `python app.py` again) after editing app.py or index.html.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
