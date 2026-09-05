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
import requests

from flask import Flask, render_template, request, jsonify

from src.service.LipSyncService import LipSyncProcessor
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
    }

    with _history_lock:
        history = load_history()
        history.append(record)
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f)
        except Exception as e:
            print(f"Failed to write history.json: {e}")


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

        # 2. Write to a temp mp3 and upload so Sync can fetch it via URL
        tmp_dir = os.path.join(ROOT_DIR, "Data", "Inputs")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_aud = os.path.join(tmp_dir, f"webapp_gen_aud_{uuid.uuid4()}.mp3")
        with open(tmp_aud, "wb") as f:
            f.write(input_audio)

        aud_url = file_processor.upload_file_uguu(tmp_aud)
        os.remove(tmp_aud)

        if not aud_url:
            result['status'] = 'FAILED'
            result['error'] = 'Failed to upload generated speech audio.'
            result['duration_seconds'] = round(time.time() - started_at, 1)
            record_history(entry, result, source)
            return result

        entry['audio'] = aud_url

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


def run_generation(job_uuid: str, entry: dict):
    """Runs a single generation and writes progress/result into JOBS for polling."""
    JOBS[job_uuid]['created_at'] = JOBS[job_uuid].get('created_at', time.time())
    result = execute_pipeline(entry, source='single')
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
        'sync_mode': form.get('sync_mode', 'bounce'),
        'lipsync_model': form.get('lipsync_model', 'lipsync-2'),
        'tts_model': form.get('tts_model', 'eleven_multilingual_v2'),
        'temperature': float(form.get('temperature', 0.5)),
        'model_mode': form.get('model_mode', 'lips'),
        'emotion': form.get('emotion', ''),
        'use_phonetic': form.get('use_phonetic') == 'on',
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

    if not entry['video'] or not entry['text'] or not entry['voice_id']:
        return jsonify({'error': 'video, text, and voice_id are required.'}), 400

    job_uuid = str(uuid.uuid4())
    JOBS[job_uuid] = {'status': 'STARTING', 'entry': entry, 'created_at': time.time()}

    thread = threading.Thread(target=run_generation, args=(job_uuid, entry), daemon=True)
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
