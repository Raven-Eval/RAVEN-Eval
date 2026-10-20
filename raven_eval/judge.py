#!/usr/bin/env python3
"""Anonymous, configurable LMM judge for pairwise AIVGM evaluation.

The source contains no local usernames, absolute storage paths, affiliation
information, credential files, or author-identifying defaults. Runtime paths
are supplied through command-line arguments, while API credentials are read
only from an environment variable. Persisted state and log metadata use paths
relative to ``--root`` and redact common credential or identity-bearing text.
"""
import argparse
import base64
import copy
import fcntl
import json
import hashlib
import tempfile
import math
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


ROOT = Path(".")
STATE_JSON = None
SUMMARY_JSON = None
LOG_DIR = None
FRAME_CACHE_DIR = None
API_URL = None
TASK_SPECS = {}
TARGET_MODELS = set()
TARGET_MODEL_REGEX = None
EXCLUDE_MODELS = set()
EXCLUDE_MODEL_REGEX = None
PAIR_SELECTION = "any"
SKIP_BASES = []
OUTPUT_BASE = "general_lmm_judge"
OVERWRITE_EXISTING = False
LABEL_RE = re.compile(r"^\s*(Better\b|Worse\b|Comparable\b|Similar\b|更好|更差|相近|相似)", re.IGNORECASE)
PROTOCOL_VERSION = "anonymous_uniform8_v1"
SAMPLING_MODE = "uniform"
FRAME_COUNT = 8
CURRENT_PROTOCOL = {}
SELECTED_TRIPLE_IDS = set()

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()


def read_openai_key(args):
    """Read the API key only from an environment variable.

    Keeping credentials out of source files, command histories, and local
    configuration files reduces both security and anonymity risks.
    """
    if args.allow_no_api_key:
        return ""
    value = os.environ.get(args.api_key_env, "").strip()
    if value:
        return value
    raise RuntimeError(
        f"API key environment variable {args.api_key_env!r} is not set."
    )


def triples_of(data):
    for key in ("pairwise_comparison_triples", "pairwise_triples", "triples", "pairs"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, list):
            return value
    return data if isinstance(data, list) else []


def has_label(value):
    if not isinstance(value, dict):
        return False
    text = value.get("output_text") or value.get("text") or value.get("output") or ""
    return isinstance(text, str) and bool(LABEL_RE.match(text.strip()))


def extract_label(text):
    if not isinstance(text, str):
        return ""
    labels = {"better": "Better", "worse": "Worse", "comparable": "Comparable", "similar": "Comparable"}
    return labels.get(text.strip().casefold(), "")


def direction_key(base, reverse):
    return f"{base}_{'reversed_evaluation' if reverse else 'evaluation'}"


def output_key(reverse):
    return direction_key(OUTPUT_BASE, reverse)


def skip_keys(reverse):
    return [direction_key(base, reverse) for base in SKIP_BASES]


def protocol_for(task_type):
    config = dict(CURRENT_PROTOCOL)
    config["guideline_sha256"] = config["guideline_sha256"][task_type]
    config["task_type"] = task_type
    return config


def existing_skip_reason(triple, reverse, task_type):
    for base, key in zip(SKIP_BASES, skip_keys(reverse)):
        if has_label(triple.get(key)):
            return f"existing_{base}"
    value = triple.get(output_key(reverse))
    if not OVERWRITE_EXISTING and has_label(value):
        if value.get("judge_config") != protocol_for(task_type):
            raise ValueError("Existing output uses a different or unknown protocol; choose another --output-base or explicitly --overwrite.")
        return "existing_current_output"
    return None


def triple_models(triple):
    return (
        (triple.get("generated_video_a") or {}).get("model"),
        (triple.get("generated_video_b") or {}).get("model"),
    )


def pair_selected(triple):
    if SELECTED_TRIPLE_IDS and triple.get("triple_id") not in SELECTED_TRIPLE_IDS:
        return False
    model_a, model_b = triple_models(triple)
    if not model_a or not model_b:
        return False

    model_a_cf = model_a.casefold()
    model_b_cf = model_b.casefold()
    if model_a_cf in EXCLUDE_MODELS or model_b_cf in EXCLUDE_MODELS:
        return False
    if EXCLUDE_MODEL_REGEX and (
        EXCLUDE_MODEL_REGEX.search(model_a) or EXCLUDE_MODEL_REGEX.search(model_b)
    ):
        return False

    if not TARGET_MODELS and TARGET_MODEL_REGEX is None:
        return True

    matches = []
    for model in (model_a, model_b):
        exact = model.casefold() in TARGET_MODELS
        regex_hit = bool(TARGET_MODEL_REGEX.search(model)) if TARGET_MODEL_REGEX else False
        matches.append(exact or regex_hit)
    return all(matches) if PAIR_SELECTION == "all" else any(matches)


def data_uri(path):
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix)
    if not mime:
        raise ValueError(f"unsupported image type: {path.suffix.lower() or '<none>'}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def video_duration_seconds(path):
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(output)


def extract_frame(video_path, out_path, timestamp=0.0, sseof=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sseof:
        commands = [
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-sseof",
                "-0.1",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_path),
            ]
        ]
    else:
        commands = [
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_path),
            ],
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-ss",
                f"{timestamp:.3f}",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_path),
            ],
        ]
    for command in commands:
        subprocess.run(command, check=False)
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        if out_path.exists():
            out_path.unlink()
    raise FileNotFoundError("failed to extract a requested video frame")


def ensure_reference_frames(task_type, data):
    mode = TASK_SPECS[task_type]["reference_mode"]
    if mode == "none":
        return []
    result = []
    for side in (["first"] if mode == "first" else ["first", "last"]):
        raw = data.get(f"reference_{side}_frame_path")
        explicit = resolve_path(ROOT, raw) if raw else None
        if explicit is not None and explicit.is_file() and explicit.stat().st_size > 0:
            frame = explicit
        else:
            raw_video = data.get("reference_video_path")
            video = resolve_path(ROOT, raw_video) if raw_video else None
            if video is None or not video.is_file():
                raise FileNotFoundError(f"Missing reference {side} frame and source video")
            stat = video.stat()
            identity = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            cache_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            frame = FRAME_CACHE_DIR / task_type / "reference_frames" / f"{cache_id}_{side}.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            with frame.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not frame.is_file() or not frame.stat().st_size:
                    if side == "first":
                        extract_frame(video, frame, 0.0)
                    else:
                        duration = video_duration_seconds(video)
                        try:
                            extract_frame(video, frame, max(duration - 0.05, 0.0))
                        except FileNotFoundError:
                            extract_frame(video, frame, sseof=True)
        result.append((f"{side.capitalize()} frame of the reference video (task input):", frame))
    return result


def extract_video_frames(video_path, task_type):
    """Sample eight chronological frames by default; 1 fps is an explicit legacy option."""
    import hashlib

    duration = video_duration_seconds(video_path)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video duration must be finite and positive")
    # Container duration can extend beyond the final decodable frame. Use actual
    # frame timestamps so the final sample remains valid for low/VFR frame rates.
    frame_times = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(video_path)], text=True)
    available = []
    for line in frame_times.splitlines():
        try:
            value = float(line.split(",", 1)[0])
            if math.isfinite(value):
                available.append(value)
        except ValueError:
            pass
    if not available:
        raise ValueError("Video contains no timestamped frames")
    last_time = max(0.0, max(available) - 0.001)
    timestamps = (list(range(math.ceil(duration))) if SAMPLING_MODE == "1fps" else [i * last_time / max(FRAME_COUNT - 1, 1) for i in range(FRAME_COUNT)])
    sampling_tag = "fps_1" if SAMPLING_MODE == "1fps" else f"uniform_{FRAME_COUNT}"
    stat = video_path.stat()
    identity = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{PROTOCOL_VERSION}:{SAMPLING_MODE}:{FRAME_COUNT}"
    cache_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    # Sampling policy, file identity and protocol are isolated in cache paths.
    cache_dir = FRAME_CACHE_DIR / task_type / cache_id / sampling_tag
    lock_dir = FRAME_CACHE_DIR / task_type / "_locks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_id}.{sampling_tag}.lock"
    expected = [cache_dir / f"frame_{index:02d}.jpg" for index in range(len(timestamps))]
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if all(path.exists() and path.stat().st_size > 0 for path in expected):
                return expected
            for path in expected:
                if path.exists():
                    path.unlink()
            for out_path, timestamp in zip(expected, timestamps):
                # Fail explicitly rather than silently substituting the last frame.
                extract_frame(video_path, out_path, timestamp)
            return expected
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_guidelines():
    return {
        name: spec["guideline"].read_text(encoding="utf-8")
        for name, spec in TASK_SPECS.items()
    }


def build_instruction(triple, guideline, args):
    # Generator model names are concealed from the judge by default: only
    # positional aliases Model A / Model B are included in the instruction.
    # --show-model-names explicitly opts out of this blinded evaluation protocol.
    return f"""{guideline.strip()}

Only compare the following two generated videos:
- First video: generated_video_a, model {triple['generated_video_a']['model'] if args.show_model_names else 'Model A'}
- Second video: generated_video_b, model {triple['generated_video_b']['model'] if args.show_model_names else 'Model B'}

prompt:
{triple['prompt']}

scoring_criteria:
{triple['scoring_criteria']}

Follow the guideline above strictly. Do not output reasoning, drafts, analysis, step-by-step decomposition, self-corrections, or descriptions of how you will answer.
Use the label_only output format. The final response must contain exactly one of these three labels: "Better", "Worse", or "Comparable". Do not provide reasons or any additional text."""


def append_image(content, text, image_path, detail):
    content.append({"type": "input_text", "text": text})
    content.append({"type": "input_image", "image_url": data_uri(image_path), "detail": detail})


def build_payload(task_type, data, triple, guideline, args):
    video_a = resolve_path(ROOT, triple["generated_video_a"]["video_path"])
    video_b = resolve_path(ROOT, triple["generated_video_b"]["video_path"])
    frames_a = extract_video_frames(video_a, task_type)
    frames_b = extract_video_frames(video_b, task_type)
    content = [{"type": "input_text", "text": build_instruction(triple, guideline, args)}]
    for text, frame_path in ensure_reference_frames(task_type, data):
        append_image(content, text, frame_path, args.image_detail)
    content.append(
        {
            "type": "input_text",
            "text": f"First video, generated_video_a: the following {len(frames_a)} frames are sampled {'at 1 fps' if SAMPLING_MODE == '1fps' else 'uniformly over the full video duration'} and arranged chronologically.",
        }
    )
    for index, frame in enumerate(frames_a, 1):
        append_image(content, f"Sampled frame {index}/{len(frames_a)} from the first video, generated_video_a:", frame, args.image_detail)
    content.append(
        {
            "type": "input_text",
            "text": f"Second video, generated_video_b: the following {len(frames_b)} frames are sampled {'at 1 fps' if SAMPLING_MODE == '1fps' else 'uniformly over the full video duration'} and arranged chronologically.",
        }
    )
    for index, frame in enumerate(frames_b, 1):
        append_image(content, f"Sampled frame {index}/{len(frames_b)} from the second video, generated_video_b:", frame, args.image_detail)
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": args.max_output_tokens,
    }
    if args.reasoning_effort != "default":
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def call_openai(payload, api_key, args):
    last_error = None
    for attempt in range(1, args.api_retries + 1):
        req = urllib.request.Request(
            args.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=args.api_timeout, context=SSL_CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = {"status": exc.code, "body": sanitize_text(body[:4000]), "attempt": attempt}
            if exc.code < 500 and exc.code not in {408, 409, 429}:
                raise RuntimeError(json.dumps(last_error, ensure_ascii=False))
        except Exception as exc:
            last_error = {"error": sanitize_text(repr(exc)), "attempt": attempt}
        time.sleep(2 * attempt)
    raise RuntimeError(json.dumps(last_error, ensure_ascii=False))


def extract_output_text(response):
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"].strip()
    parts = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def with_json_lock(path, callback):
    lock_path = path.with_suffix(path.suffix + ".general_lmm_judge.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = callback(data)
            write_json(path, data)
            return result
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_task_and_triple(task_type, task_json, triple_id):
    original = json.loads(task_json.read_text(encoding="utf-8"))
    for item in triples_of(original):
        if item.get("triple_id") == triple_id:
            triple = copy.deepcopy(item)
            data = copy.deepcopy(original) if isinstance(original, dict) else {}
            for key in ("reference_video_path", "reference_first_frame_path", "reference_last_frame_path"):
                if not data.get(key) and triple.get(key):
                    data[key] = triple[key]
            for side in ("generated_video_a", "generated_video_b"):
                value = triple[side].get("video_path")
                if not value:
                    raise ValueError(f"Missing {side}.video_path")
                triple[side]["video_path"] = str(resolve_path(ROOT, value))
            return data, triple
    raise KeyError(triple_id)


def write_result(task_json, triple_id, key, label, task_type):
    def update(data):
        for triple in triples_of(data):
            if triple.get("triple_id") == triple_id:
                triple[key] = {"output_text": label, "judge_config": protocol_for(task_type)}
                return True
        return False
    if not with_json_lock(task_json, update):
        raise KeyError(triple_id)


def collect_tasks(task_types, glob_pattern, task_paths=None):
    tasks = []
    seen = set()
    if task_paths and len(task_types) != 1:
        raise ValueError("--task-json requires exactly one task type")
    for task_type in task_types:
        pair_dir = TASK_SPECS[task_type]["pair_dir"]
        if not task_paths and not pair_dir.is_dir():
            raise FileNotFoundError(f"Pair directory does not exist: {public_path(pair_dir)}")
        paths = [resolve_path(ROOT, p) for p in task_paths] if task_paths else sorted(pair_dir.glob(glob_pattern))
        for path in paths:
            identity = (task_type, path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            data = json.loads(path.read_text(encoding="utf-8"))
            selected = [t for t in triples_of(data) if pair_selected(t)]
            ids = [t.get("triple_id") for t in selected]
            if any(not tid for tid in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Missing or duplicate triple_id: {public_path(path)}")
            if selected:
                tasks.append((task_type, path))
    return tasks


def public_path(path):
    """Return a path safe to store in logs/state files.

    Paths under --root are stored relative to that root. Paths outside it are
    reduced to their basename so that local usernames, mount points, and
    institutional storage layouts are never persisted.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except (ValueError, OSError):
        return path.name


def sanitize_text(value):
    """Redact common identity- and credential-bearing strings from logs."""
    text = str(value)
    for sensitive_root in (ROOT, Path.home()):
        try:
            raw = str(sensitive_root.resolve())
        except OSError:
            raw = str(sensitive_root)
        if raw and raw not in {"/", "."}:
            text = text.replace(raw, "<ROOT>")

    # Common absolute local paths. URLs are intentionally not matched.
    text = re.sub(
        r"(?<![A-Za-z0-9:])/(?:home|Users|mnt|srv|tmp|var|private|Volumes)/[^\s\"']+",
        "<PATH>",
        text,
    )
    text = re.sub(r"[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*", "<PATH>", text)
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "<EMAIL>",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<API_KEY>", text)
    text = re.sub(r"\borg-[A-Za-z0-9_-]{4,}\b", "<ORG_ID>", text)
    return text


def job_id(task_type, task_json, triple_id, key):
    return f"{task_type}::{public_path(task_json)}::{triple_id}::{key}"


def load_json(path, default):
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name, delete=False) as stream:
            temp = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temp.chmod(path.stat().st_mode & 0o777)
        temp.replace(path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def collect_jobs(tasks, state, max_attempts):
    jobs = []
    for task_type, task_json in tasks:
        data = json.loads(task_json.read_text(encoding="utf-8"))
        for triple in triples_of(data):
            if not pair_selected(triple):
                continue
            triple_id = triple.get("triple_id")
            if not triple_id:
                continue
            for reverse in (False, True):
                if existing_skip_reason(triple, reverse, task_type):
                    continue
                key = output_key(reverse)
                jid = job_id(task_type, task_json, triple_id, key)
                entry = state.get(jid, {})
                # Persisted judgments, not state alone, determine completion.
                attempts = 0 if entry.get("last_done") is True else int(entry.get("attempts", 0))
                if attempts >= max_attempts:
                    continue
                jobs.append(
                    {
                        "task_type": task_type,
                        "task_json": task_json,
                        "triple_id": triple_id,
                        "reverse": reverse,
                        "output_key": key,
                        "job_id": jid,
                        "attempts": attempts,
                    }
                )
    return jobs


def summarize(tasks, state, args):
    rows = []
    total = {key: 0 for key in ("target_pairs", "target_directions", "skipped_directions", "run_pending_directions", "exhausted_directions")}
    for task_type, task_json in tasks:
        data = json.loads(task_json.read_text(encoding="utf-8"))
        selected = [triple for triple in triples_of(data) if pair_selected(triple)]
        row = dict(task_type=task_type, task_json=public_path(task_json), target_pairs=len(selected), target_directions=2*len(selected), skipped_directions=0, run_pending_directions=0, exhausted_directions=0)
        for triple in selected:
            for reverse in (False, True):
                if existing_skip_reason(triple, reverse, task_type):
                    row["skipped_directions"] += 1
                else:
                    entry = state.get(job_id(task_type, task_json, triple["triple_id"], output_key(reverse)), {})
                    attempts = 0 if entry.get("last_done") is True else int(entry.get("attempts", 0))
                    row["exhausted_directions" if attempts >= args.max_attempts else "run_pending_directions"] += 1
        rows.append(row)
        for key in total:
            total[key] += row[key]
    summary = dict(created_at=time.strftime("%Y-%m-%d %H:%M:%S"), judge_model=args.model,
                   api="Responses-compatible API", output_base=OUTPUT_BASE, judge_config=CURRENT_PROTOCOL,
                   skip_existing_bases=SKIP_BASES, overwrite_existing=OVERWRITE_EXISTING,
                   target_models=sorted(TARGET_MODELS), target_model_regex=args.target_model_regex,
                   pair_selection=PAIR_SELECTION, task_count=len(tasks), task_rows=rows, **total)
    if not args.dry_run:
        write_json(SUMMARY_JSON, summary)
    return summary


def run_job(job, args, api_key, guidelines):
    data, triple = read_task_and_triple(job["task_type"], job["task_json"], job["triple_id"])
    reason = existing_skip_reason(triple, job["reverse"], job["task_type"])
    if reason:
        return True, reason
    request_triple = copy.deepcopy(triple)
    # Both forward and reversed evaluations conceal generator model names from
    # the judge by default (unless --show-model-names is explicitly enabled).
    # Reversal swaps the videos; Model A / Model B remain positional aliases,
    # not the generators' actual identities. Internal model metadata is retained.
    if job["reverse"]:
        request_triple["generated_video_a"], request_triple["generated_video_b"] = (
            request_triple["generated_video_b"],
            request_triple["generated_video_a"],
        )
    payload = build_payload(job["task_type"], data, request_triple, guidelines[job["task_type"]], args)
    last_response = None
    for attempt in range(1, args.output_attempts + 1):
        response = call_openai(payload, api_key, args)
        last_response = response
        output_text = extract_output_text(response)
        label = extract_label(output_text)
        if label:
            write_result(job["task_json"], job["triple_id"], job["output_key"], label, job["task_type"])
            return True, label
        time.sleep(2 * attempt)
    message = sanitize_text(json.dumps(last_response, ensure_ascii=False)[:1200]) if last_response else "empty"
    return False, message


def parse_task_types(value):
    task_types = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(task_types) - set(TASK_SPECS))
    if unknown:
        raise ValueError(f"unknown task types: {unknown}")
    return task_types


def slugify(value):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "general_lmm_judge"


def resolve_path(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main(argv=None, *, task_type=None, default_root=None):
    global ROOT, TASK_SPECS, STATE_JSON, SUMMARY_JSON, LOG_DIR, FRAME_CACHE_DIR, API_URL
    global TARGET_MODELS, TARGET_MODEL_REGEX, EXCLUDE_MODELS, EXCLUDE_MODEL_REGEX
    global PAIR_SELECTION, SKIP_BASES, OUTPUT_BASE, OVERWRITE_EXISTING, CURRENT_PROTOCOL, SELECTED_TRIPLE_IDS, SAMPLING_MODE, FRAME_COUNT

    parser = argparse.ArgumentParser(description="RAVEN-Eval anonymous pairwise judge (Responses-compatible API, eight uniform frames).")
    parser.add_argument("--root", type=Path, default=default_root or Path(__file__).resolve().parents[1])
    parser.add_argument("--model", required=True, help="Judge model identifier.")
    parser.add_argument("--api-url", required=True, help="Responses-compatible endpoint (not chat/completions).")
    parser.add_argument("--api-key-env", default="LMM_API_KEY")
    parser.add_argument("--allow-no-api-key", action="store_true")
    parser.add_argument("--output-base", default=None, help="Default: MODEL_blind_uniform8_v1; protocol metadata isolates historical results.")
    parser.add_argument("--skip-existing-base", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-model-names", action="store_true", help="Explicitly reveal generator identities; default is hidden.")
    parser.add_argument("--target-model", action="append", default=[])
    parser.add_argument("--target-model-regex", default=None)
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--exclude-model-regex", default=None)
    parser.add_argument("--pair-selection", choices=["any", "all"], default="any")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--task-types", default=task_type or "first_last,keyframe,t2v")
    parser.add_argument("--task-json", action="append", type=Path, help="Explicit task JSON, repeatable; one task type only.")
    parser.add_argument("--triple-id", action="append", default=[])
    parser.add_argument("--glob", default="*_pairwise_triples.json")
    parser.add_argument("--pair-dir", help="Override pair directory for a single task type.")
    parser.add_argument("--guideline-file", help="Override guideline for a single task type.")
    parser.add_argument("--first-last-pair-dir", default="runs/pairs/flt")
    parser.add_argument("--keyframe-pair-dir", default="runs/pairs/kft")
    parser.add_argument("--t2v-pair-dir", default="runs/pairs/t2v")
    parser.add_argument("--first-last-guideline", default="guidelines/flt.txt")
    parser.add_argument("--keyframe-guideline", default="guidelines/kft.txt")
    parser.add_argument("--t2v-guideline", default="guidelines/t2v.txt")
    for name in ("state-json", "summary-json", "log-dir", "frame-cache-dir"):
        parser.add_argument("--" + name, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--reasoning-effort", choices=["default", "none", "low", "medium", "high"], default="default")
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--output-attempts", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--sampling", choices=["uniform", "1fps"], default="uniform")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-detail", choices=["auto", "high", "low"], default="high")
    parser.add_argument("--dry-run", action="store_true", help="Read-only coverage preview; no API, frames, or output writes.")
    args = parser.parse_args(argv)
    for name in ("max_output_tokens", "api_timeout", "api_retries", "output_attempts", "max_attempts", "max_parallel"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_jobs < 0:
        parser.error("--max-jobs must be nonnegative")
    ROOT = args.root.expanduser().resolve()
    TASK_SPECS = {
        "first_last": dict(pair_dir=resolve_path(ROOT,args.first_last_pair_dir), guideline=resolve_path(ROOT,args.first_last_guideline), reference_mode="first_last"),
        "keyframe": dict(pair_dir=resolve_path(ROOT,args.keyframe_pair_dir), guideline=resolve_path(ROOT,args.keyframe_guideline), reference_mode="first"),
        "t2v": dict(pair_dir=resolve_path(ROOT,args.t2v_pair_dir), guideline=resolve_path(ROOT,args.t2v_guideline), reference_mode="none"),
    }
    task_types = list(dict.fromkeys(parse_task_types(args.task_types)))
    if not task_types:
        parser.error("At least one task type is required")
    if task_type and task_types != [task_type]:
        parser.error(f"This entry point is restricted to {task_type}")
    if args.pair_dir or args.guideline_file:
        if len(task_types) != 1:
            parser.error("--pair-dir/--guideline-file require one task type")
        if args.pair_dir:
            TASK_SPECS[task_types[0]]["pair_dir"] = resolve_path(ROOT,args.pair_dir)
        if args.guideline_file:
            TASK_SPECS[task_types[0]]["guideline"] = resolve_path(ROOT,args.guideline_file)
    API_URL = args.api_url
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")
    SAMPLING_MODE, FRAME_COUNT = args.sampling, args.num_frames
    sampling_tag = "1fps" if SAMPLING_MODE == "1fps" else f"uniform{FRAME_COUNT}"
    visibility = "named" if args.show_model_names else "blind"
    OUTPUT_BASE = slugify(args.output_base or f"{args.model}_{visibility}_{sampling_tag}_v1")
    SKIP_BASES = [slugify(base) for base in args.skip_existing_base]
    OVERWRITE_EXISTING = args.overwrite
    TARGET_MODELS = set() if args.all_models else {name.casefold() for name in args.target_model}
    TARGET_MODEL_REGEX = None if args.all_models or not args.target_model_regex else re.compile(args.target_model_regex,re.IGNORECASE)
    EXCLUDE_MODELS = {name.casefold() for name in args.exclude_model}
    EXCLUDE_MODEL_REGEX = re.compile(args.exclude_model_regex,re.IGNORECASE) if args.exclude_model_regex else None
    PAIR_SELECTION = args.pair_selection
    SELECTED_TRIPLE_IDS = set(args.triple_id)
    guidelines = {name: TASK_SPECS[name]["guideline"].read_text(encoding="utf-8") for name in task_types}
    CURRENT_PROTOCOL = dict(version=PROTOCOL_VERSION, judge_model=args.model, show_model_names=args.show_model_names,
                            sampling=SAMPLING_MODE, num_frames=FRAME_COUNT if SAMPLING_MODE == "uniform" else None, sampling_fps=1 if SAMPLING_MODE == "1fps" else None, image_detail=args.image_detail, reasoning_effort=args.reasoning_effort,
                            max_output_tokens=args.max_output_tokens,
                            guideline_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in guidelines.items()})
    summary_root = ROOT / ".run_artifacts"
    prefix = "general_" + OUTPUT_BASE + "_" + "_".join(task_types)
    STATE_JSON = resolve_path(ROOT,args.state_json) if args.state_json else summary_root / (prefix + "_state.json")
    SUMMARY_JSON = resolve_path(ROOT,args.summary_json) if args.summary_json else summary_root / (prefix + "_summary.json")
    LOG_DIR = resolve_path(ROOT,args.log_dir) if args.log_dir else summary_root / ("logs_" + prefix)
    FRAME_CACHE_DIR = resolve_path(ROOT,args.frame_cache_dir) if args.frame_cache_dir else summary_root / "frame_cache_anonymous"
    tasks = collect_tasks(task_types,args.glob,args.task_json)
    if not tasks:
        parser.error("No matching task pairs found")
    state = load_json(STATE_JSON,{})
    pending = collect_jobs(tasks,state,args.max_attempts)
    if args.max_jobs:
        pending = pending[:args.max_jobs]
    summary = summarize(tasks,state,args)
    print(json.dumps({k:v for k,v in summary.items() if k != "task_rows"} | {"selected_jobs":len(pending),"state_json":public_path(STATE_JSON),"summary_json":public_path(SUMMARY_JSON)},ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        return 0
    api_key = read_openai_key(args)
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    state_lock = Lock()
    def run_one(job):
        attempt_no = job["attempts"] + 1
        log_id = hashlib.sha256(job["job_id"].encode()).hexdigest()[:20]
        log_path = LOG_DIR / f"{log_id}_attempt{attempt_no}.log"
        with state_lock:
            entry = state.setdefault(job["job_id"],{})
            entry.update(task_type=job["task_type"],task_json=public_path(job["task_json"]),triple_id=job["triple_id"],
                         output_key=job["output_key"],reverse=job["reverse"],attempts=attempt_no,
                         last_started_at=time.strftime("%Y-%m-%d %H:%M:%S"),last_log=public_path(log_path),
                         last_done=None,judge_config=CURRENT_PROTOCOL)
            write_json(STATE_JSON,state)
        try:
            done,message = run_job(job,args,api_key,guidelines)
        except Exception as exc:
            done,message = False,f"{type(exc).__name__}: {sanitize_text(exc)}"
        log_path.write_text(sanitize_text(message)+"\n",encoding="utf-8")
        with state_lock:
            state[job["job_id"]].update(last_done=bool(done),last_error=None if done else sanitize_text(message),last_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            write_json(STATE_JSON,state)
        print(f"{'DONE' if done else 'ERROR'} {job['task_type']} {log_id} {job['output_key']}",flush=True)
        return done
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        results = [future.result() for future in as_completed([executor.submit(run_one,job) for job in pending])]
    # After an overwrite, the newly written protocol-matching labels count as done.
    OVERWRITE_EXISTING = False
    final = summarize(tasks,state,args)
    print(json.dumps({k:v for k,v in final.items() if k != "task_rows"},ensure_ascii=False,indent=2),flush=True)
    return 0 if all(results) and final["exhausted_directions"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
