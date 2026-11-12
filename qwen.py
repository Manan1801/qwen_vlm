"""
Single-file Qwen3-VL video analysis pipeline (merges run.py + benchmark_run.py).

Same output layout as the Gemini pipeline. For every run it creates:

Results/
  {model}-{timestamp}/
    prompt.txt
    Total_Generation_Time.csv
    Video-{name}/
      Given Action Codes/        (mode: fixed)
        timestamps.csv
        Description.txt
        Action-Codes Table.png
        Raw_Response.json        (every chunk response + the final merge response)
      Generated Action Codes/    (mode: generated)
        ... same files ...

Usage:
    python pipeline_qwen.py
    python pipeline_qwen.py --video-folder videos --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
"""

import argparse
import csv
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch

torch.backends.cudnn.enabled = False

try:
    import types
    import triton.runtime.autotuner as triton_autotuner
    import triton.runtime.jit as triton_jit
    import triton.runtime.cache as triton_cache
    import triton.compiler.compiler as triton_compiler

    if not hasattr(triton_autotuner, "JITFunction") and hasattr(triton_jit, "JITFunction"):
        triton_autotuner.JITFunction = triton_jit.JITFunction
    if not hasattr(triton_autotuner, "get_cache_invalidating_env_vars"):
        triton_autotuner.get_cache_invalidating_env_vars = triton_compiler.get_cache_invalidating_env_vars
    if not hasattr(triton_autotuner, "get_cache_manager"):
        triton_autotuner.get_cache_manager = triton_cache.get_cache_manager
    if not hasattr(triton_autotuner, "triton_key"):
        triton_autotuner.triton_key = triton_compiler.triton_key
    if not hasattr(triton_autotuner, "knobs"):
        triton_autotuner.knobs = types.SimpleNamespace(autotuning=types.SimpleNamespace(print=False))
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
DEFAULT_VIDEO_DIR = Path("videos")
DEFAULT_RESULTS_DIR = Path("Results")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODES = {
    "fixed_codes": {"folder": "Given Action Codes", "label": "fixed"},
    "open_discovery": {"folder": "Generated Action Codes", "label": "generated"},
}
MODES_TO_RUN = ["fixed_codes", "open_discovery"]

SAMPLE_EVERY_SECONDS = 0.5
CHUNK_SIZE = 8
MAX_IMAGE_SIZE = 640
MAX_NEW_TOKENS = 500
MAX_NEW_TOKENS_MERGE = 4096
GPU_MEMORY_GIB = 20
CPU_MEMORY_GIB = 48

ACTION_CODES = [
    "Meshing gears - correct",
    "Meshing gears - incorrect",
    "Stopper magnet - correct",
    "Stopper magnet - incorrect",
    "Axle and gear magnets - correct",
    "Axle and gear magnets - incorrect",
    "Grey plate and red gear - correct",
    "Grey plate and red gear - incorrect",
    "Look at the components - includes rotations, looking, search",
    "mount - correct",
    "mount - incorrect",
    "pairing - correct combo",
    "pairing - incorrect combo",
    "Orientation of the part - correct",
    "Orientation of the part - incorrect",
    "disassembling",
    "Correct pair & orientation",
    "Incorrect pair & correct orientation",
    "Correct pair & incorrect orientation",
    "Incorrect pair & orientation",
    "orienting",
    "testing - includes turning gears",
    "rotating",
]

CHUNK_PROMPT_FIXED_CODES = """
You are shown {n_frames} frames extracted from one chunk of a longer video,
in chronological order, labeled Frame 0 to Frame {max_index}.

Create a concise descriptor for what happens across these frames. Do not
describe each frame separately; infer the short sequence they represent.
Only report actions that are visibly supported by the frames. A score of 0
is fine when an action is not observed.

Classify this chunk using every action code below. For each code, return a
score from 0 to 100, where 0 means not observed and 100 means clearly
observed, supporting keywords/short phrases, a one-sentence evidence
description, and the list of 0-based frame indices (among the frames shown
here) where this action is visible. Return an empty frame_indices list when
the score is 0. Do not invent actions that are not visible.

Action codes:
{action_codes}

Return only valid JSON with this exact shape:
{{
    "descriptor": "one paragraph of approximately 50-100 words",
    "action_matches": [
        {{"action": "action code", "score": 0, "keywords": [], "evidence": "", "frame_indices": []}}
    ]
}}
""".strip()

CHUNK_PROMPT_OPEN_DISCOVERY = """
You are shown {n_frames} frames extracted from one chunk of a longer video,
in chronological order, labeled Frame 0 to Frame {max_index}.

Create a concise descriptor for what happens across these frames. Do not
describe each frame separately; infer the short sequence they represent.

Then, independently identify every significant action, activity, or movement
visible across these frames. Do not use any predefined list of actions.
Invent your own short, clear action labels based only on what you actually
observe (for example "picking up gear", "aligning magnet", "rotating part").
List every distinct action you can identify, however many there are.

For each action, return a confidence score from 0 to 100, supporting
keywords/short phrases, a one-sentence evidence description, and the list of
0-based frame indices (among the frames shown here) where this action is
visible.

Return only valid JSON with this exact shape:
{{
    "descriptor": "one paragraph of approximately 50-100 words",
    "action_matches": [
        {{"action": "your own invented label", "score": 0, "keywords": [], "evidence": "", "frame_indices": []}}
    ]
}}
""".strip()

MERGE_PROMPT_FIXED_CODES = """
Below are descriptors and detected action codes from consecutive chunks of
ONE continuous video, in chronological order. Each chunk lists its time
range, its descriptor, and every action code score observed in that chunk
along with the exact timestamps it was seen at.

Chunks:
{chunk_summaries}

Using all of this, write ONE single coherent descriptor paragraph
(approximately 100-150 words) for the entire video as a whole. Do not
describe it chunk by chunk; infer the overall story/sequence of events
across the whole video.

Then produce ONE final merged score (0-100) for every action code below,
reflecting how clearly it was observed anywhere in the full video. Use the
exact timestamp strings already given above when merging; do not invent new
ones. Return a merged, deduplicated list of those timestamps for every time
that action occurs, plus merged supporting keywords and a short evidence
summary.

Action codes:
{action_codes}

Return only valid JSON with this exact shape:
{{
    "descriptor": "one paragraph, approximately 100-150 words",
    "action_matches": [
        {{"action": "action code", "score": 0, "keywords": [], "evidence": "", "timestamps": []}}
    ]
}}
""".strip()

MERGE_PROMPT_OPEN_DISCOVERY = """
Below are descriptors and independently-detected actions from consecutive
chunks of ONE continuous video, in chronological order. Each chunk lists its
time range, its descriptor, and the actions it detected with their exact
timestamps. Each chunk invented its own action labels independently, so the
same real action may appear under slightly different wording in different
chunks.

Chunks:
{chunk_summaries}

Using all of this, write ONE single coherent descriptor paragraph
(approximately 100-150 words) for the entire video as a whole. Do not
describe it chunk by chunk; infer the overall story/sequence of events
across the whole video.

Then produce ONE final merged list of distinct actions observed anywhere in
the video. Merge entries from different chunks that refer to the same real
action under a single clear label, even if the chunks worded them
differently. For each final action, return a confidence score from 0 to 100,
merged supporting keywords, a short evidence summary, and a merged,
deduplicated list of timestamps for every time that action occurs. Use the
exact timestamp strings already given above; do not invent new ones.

Return only valid JSON with this exact shape:
{{
    "descriptor": "one paragraph, approximately 100-150 words",
    "action_matches": [
        {{"action": "final merged action label", "score": 0, "keywords": [], "evidence": "", "timestamps": []}}
    ]
}}
""".strip()

TIMESTAMP_FIELDS = [
    "video_name",
    "timestamp",
    "action_code",
    "mode",
    "score",
    "evidence",
    "keywords",
]

TIMING_FIELDS = [
    "video_name",
    "mode",
    "video_duration_seconds",
    "processing_seconds",
    "processing_minutes",
    "processing_min_per_video_sec",
    "status",
    "error",
]


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def safe_name(text: str) -> str:
    """Make a string safe to use as a file/folder name."""
    return re.sub(r'[<>:"/\\|?*\s]+', "_", text).strip("._") or "unnamed"


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_time(seconds: float) -> str:
    minutes, remaining = divmod(max(0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes:02d}:{remaining:05.2f}"


def find_videos(video_folder: Path) -> list[Path]:
    if not video_folder.exists():
        raise RuntimeError(f"Video folder does not exist: {video_folder}")
    if not video_folder.is_dir():
        raise RuntimeError(f"Video path is not a folder: {video_folder}")
    videos = sorted(
        p for p in video_folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise RuntimeError(f"No supported videos found in {video_folder}")
    return videos


# --------------------------------------------------------------------------- #
# Frames + model
# --------------------------------------------------------------------------- #

def extract_frames(video_path, sample_every_seconds=1.0, max_dimension=640):
    """Extract frames from the video and resize them before sending to Qwen.
    Returns (frames, timestamps, fps, duration_seconds)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_interval = max(1, int(fps * sample_every_seconds))

    frames, timestamps = [], []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            scale = min(max_dimension / width, max_dimension / height, 1.0)
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(Image.fromarray(frame))
            timestamps.append(frame_index / fps)
        frame_index += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted")
    duration = frame_count / fps if frame_count > 0 else timestamps[-1]
    return frames, timestamps, fps, duration


def infer_model_device_label(model):
    if hasattr(model, "hf_device_map"):
        return model.hf_device_map
    try:
        device = next(model.parameters()).device
        if device.type == "cuda":
            return {"model": f"cuda:{device.index}" if device.index is not None else "cuda"}
        return {"model": str(device)}
    except StopIteration:
        return {"model": "unknown"}


def load_model(model_name: str):
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)

    print("Loading model...", flush=True)
    if torch.cuda.is_available():
        max_memory = {0: f"{GPU_MEMORY_GIB}GiB", "cpu": f"{CPU_MEMORY_GIB}GiB"}
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        device_label = infer_model_device_label(model)
    else:
        print("CUDA is not available; falling back to CPU execution.")
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name, torch_dtype="auto", low_cpu_mem_usage=True,
        )
        model = model.to(torch.device("cpu"))
        device_label = {"model": "cpu"}

    model.eval()
    print(f"Model loaded. Device map: {device_label}", flush=True)
    return processor, model


def run_model(processor, model, content, max_new_tokens):
    """One chat-completion call (image+text or text-only); returns decoded text."""
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    first_device = next(model.parameters()).device
    for key in inputs:
        if hasattr(inputs[key], "to"):
            inputs[key] = inputs[key].to(first_device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    text = processor.decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs, generated_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


# --------------------------------------------------------------------------- #
# Prompts + parsing (from run.py)
# --------------------------------------------------------------------------- #

def build_chunk_prompt(mode: str, n_frames: int) -> str:
    if mode == "fixed_codes":
        return CHUNK_PROMPT_FIXED_CODES.format(
            n_frames=n_frames,
            max_index=n_frames - 1,
            action_codes="\n".join(f"- {a}" for a in ACTION_CODES),
        )
    if mode == "open_discovery":
        return CHUNK_PROMPT_OPEN_DISCOVERY.format(n_frames=n_frames, max_index=n_frames - 1)
    raise ValueError(f"Unknown mode: {mode}")


def build_merge_prompt(mode: str, chunk_summaries: str) -> str:
    if mode == "fixed_codes":
        return MERGE_PROMPT_FIXED_CODES.format(
            chunk_summaries=chunk_summaries,
            action_codes="\n".join(f"- {a}" for a in ACTION_CODES),
        )
    if mode == "open_discovery":
        return MERGE_PROMPT_OPEN_DISCOVERY.format(chunk_summaries=chunk_summaries)
    raise ValueError(f"Unknown mode: {mode}")


def parse_model_json(text):
    """Parse JSON even when a model surrounds it with a markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    recovered = recover_partial_model_json(cleaned)
    if recovered:
        recovered["parse_error"] = "Model returned incomplete JSON; recovered complete fields."
        return recovered

    return {
        "descriptor": cleaned,
        "action_matches": [],
        "parse_error": "Model did not return valid JSON.",
    }


def recover_partial_model_json(text):
    """Recover complete top-level fields and action objects from truncated JSON."""
    decoder = json.JSONDecoder()
    recovered = {}

    descriptor_key = text.find('"descriptor"')
    if descriptor_key >= 0:
        value_start = text.find(":", descriptor_key + len('"descriptor"')) + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        try:
            descriptor, _ = decoder.raw_decode(text, value_start)
            if isinstance(descriptor, str):
                recovered["descriptor"] = descriptor
        except (json.JSONDecodeError, ValueError):
            pass

    actions_key = text.find('"action_matches"')
    if actions_key >= 0:
        array_start = text.find("[", actions_key + len('"action_matches"'))
        if array_start >= 0:
            matches = []
            position = array_start + 1
            while position < len(text):
                while position < len(text) and (text[position].isspace() or text[position] == ","):
                    position += 1
                if position >= len(text) or text[position] == "]":
                    break
                try:
                    match, next_position = decoder.raw_decode(text, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(match, dict):
                    break
                matches.append(match)
                position = next_position
            recovered["action_matches"] = matches

    return recovered


def safe_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def safe_int_list(value):
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def safe_str_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize(result, mode, index_key, index_parser):
    if mode == "fixed_codes":
        by_action = {
            m.get("action"): m
            for m in result.get("action_matches", [])
            if isinstance(m, dict) and m.get("action")
        }
        source = [(a, by_action.get(a, {})) for a in ACTION_CODES]
    else:
        source = [
            (str(m["action"]).strip(), m)
            for m in result.get("action_matches", [])
            if isinstance(m, dict) and m.get("action")
        ]

    matches = [
        {
            "action": action,
            "score": safe_score(m.get("score", 0)),
            "keywords": safe_str_list(m.get("keywords", [])),
            "evidence": str(m.get("evidence", "")),
            index_key: index_parser(m.get(index_key, [])),
        }
        for action, m in source
    ]
    return {
        "descriptor": str(result.get("descriptor", "")).strip(),
        "action_matches": matches,
        **({"parse_error": result["parse_error"]} if result.get("parse_error") else {}),
    }


def normalize_chunk_result(result, mode):
    return _normalize(result, mode, "frame_indices", safe_int_list)


def normalize_merge_result(result, mode):
    return _normalize(result, mode, "timestamps", safe_str_list)


def apply_chunk_timestamps(chunk_result, chunk_frame_timestamps):
    """Convert each match's local frame_indices into absolute video timestamps."""
    for match in chunk_result["action_matches"]:
        indices = match.pop("frame_indices", [])
        stamps = [
            format_time(chunk_frame_timestamps[i])
            for i in indices
            if 0 <= i < len(chunk_frame_timestamps)
        ]
        match["timestamps"] = sorted(set(stamps))
    return chunk_result


def build_chunk_summaries(chunk_results):
    lines = []
    for chunk in chunk_results:
        lines.append(f"Chunk {chunk['chunk_number']} ({chunk['start_time']} - {chunk['end_time']}):")
        lines.append(f"Descriptor: {chunk['descriptor']}")
        lines.append("Actions:")
        for match in chunk["action_matches"]:
            if match["score"] <= 0 and not match["timestamps"]:
                continue
            lines.append(
                f"- {match['action']}: score={match['score']}, "
                f"timestamps=[{', '.join(match['timestamps'])}], "
                f"keywords=[{', '.join(match['keywords'])}], "
                f"evidence={match['evidence']}"
            )
        lines.append("")
    return "\n".join(lines)


def describe_chunk(processor, model, frames, chunk_number, mode):
    """Returns (normalized_chunk_result, raw_model_text)."""
    print(f"    chunk {chunk_number} ({len(frames)} frames)", flush=True)
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": build_chunk_prompt(mode, len(frames))})
    text = run_model(processor, model, content, MAX_NEW_TOKENS)
    return normalize_chunk_result(parse_model_json(text), mode), text


def merge_video_results(processor, model, chunk_results, mode):
    """One text-only call that merges all chunk results. Returns (result, raw_text)."""
    print(f"    merging {len(chunk_results)} chunks", flush=True)
    prompt = build_merge_prompt(mode, build_chunk_summaries(chunk_results))
    text = run_model(processor, model, [{"type": "text", "text": prompt}], MAX_NEW_TOKENS_MERGE)
    return normalize_merge_result(parse_model_json(text), mode), text


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_prompt_file(run_dir: Path, model_name: str) -> None:
    parts = [
        f"Model: {model_name}",
        f"Frame sampling: every {SAMPLE_EVERY_SECONDS}s, resized to max {MAX_IMAGE_SIZE}px",
        f"Chunk size: {CHUNK_SIZE} frames",
        f"Max new tokens: chunk={MAX_NEW_TOKENS}, merge={MAX_NEW_TOKENS_MERGE} (greedy decoding)",
        "Placeholders in braces ({n_frames}, {max_index}, {action_codes}, {chunk_summaries}) are filled at run time.",
        "",
    ]
    templates = {
        "fixed_codes": (CHUNK_PROMPT_FIXED_CODES, MERGE_PROMPT_FIXED_CODES),
        "open_discovery": (CHUNK_PROMPT_OPEN_DISCOVERY, MERGE_PROMPT_OPEN_DISCOVERY),
    }
    for mode in MODES_TO_RUN:
        chunk_t, merge_t = templates[mode]
        parts += [
            "=" * 70, f"MODE: {mode}  ({MODES[mode]['folder']})", "=" * 70,
            "--- CHUNK PROMPT ---", chunk_t, "",
            "--- MERGE PROMPT ---", merge_t, "",
        ]
        if mode == "fixed_codes":
            parts += ["--- ACTION CODES ---", *[f"- {c}" for c in ACTION_CODES], ""]
    (run_dir / "prompt.txt").write_text("\n".join(parts), encoding="utf-8")


def write_timestamps_csv(path: Path, video_name: str, mode: str, matches: list[dict]) -> None:
    """One row per (action, timestamp). Actions with no timestamps get one row with a blank timestamp."""
    label = MODES[mode]["label"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMESTAMP_FIELDS)
        writer.writeheader()
        for match in matches:
            base = {
                "video_name": video_name,
                "action_code": match.get("action", ""),
                "mode": label,
                "score": match.get("score", ""),
                "evidence": match.get("evidence", ""),
                "keywords": "; ".join(as_list(match.get("keywords"))),
            }
            for ts in as_list(match.get("timestamps")) or [""]:
                writer.writerow({**base, "timestamp": ts})


def write_description(path: Path, parsed: dict, chunk_results: list[dict]) -> None:
    """Final merged description first, then every chunk's own description with its time range."""
    lines = [
        "FINAL MERGED DESCRIPTION",
        "=" * 70,
        str(parsed.get("descriptor", "")).strip(),
        "",
        "CHUNK DESCRIPTIONS",
        "=" * 70,
    ]
    for chunk in chunk_results:
        lines.append(f"Chunk {chunk['chunk_number']} ({chunk['start_time']} - {chunk['end_time']}):")
        lines.append(str(chunk.get("descriptor", "")).strip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_table_png(path: Path, title: str, matches: list[dict]) -> None:
    headers = ["Action code", "Score", "Keywords", "Evidence", "Timestamps"]
    wrap_chars = [28, 6, 34, 44, 22]

    rows = []
    for m in matches:
        cells = [
            str(m.get("action", "")),
            str(m.get("score", "")),
            ", ".join(as_list(m.get("keywords"))) or "-",
            str(m.get("evidence", "")) or "-",
            ", ".join(as_list(m.get("timestamps"))) or "-",
        ]
        rows.append([textwrap.wrap(c, w) or [""] for c, w in zip(cells, wrap_chars)])
    if not rows:
        rows.append([["No action matches returned"]] + [[""] for _ in headers[1:]])

    char_w, pad, line_h, title_h = 0.085, 0.08, 0.17, 0.4
    col_w = [w * char_w + 2 * pad for w in wrap_chars]
    total_w = sum(col_w)
    header_h = line_h + 2 * pad
    row_h = [max(len(c) for c in r) * line_h + 2 * pad for r in rows]
    total_h = title_h + header_h + sum(row_h)

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    ax.text(pad, title_h / 2, title, fontsize=10, fontweight="bold", va="center", ha="left")

    xs = [sum(col_w[:i]) for i in range(len(col_w))]
    y = title_h
    ax.add_patch(plt.Rectangle((0, y), total_w, header_h, color="#dfe6ee"))
    for x, h in zip(xs, headers):
        ax.text(x + pad, y + pad, h, fontsize=8, fontweight="bold", va="top", ha="left")
    y += header_h

    for i, (cells, h) in enumerate(zip(rows, row_h)):
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y), total_w, h, color="#f4f6f8"))
        for x, lines in zip(xs, cells):
            ax.text(x + pad, y + pad, "\n".join(lines), fontsize=8, va="top", ha="left")
        y += h
        ax.plot([0, total_w], [y, y], color="#c9d1d9", linewidth=0.5)

    fig.savefig(path, facecolor="white")
    plt.close(fig)


def timing_row(video_name, mode, duration, seconds, status, error="") -> dict:
    per_sec = (seconds / 60) / duration if duration else 0.0
    return {
        "video_name": video_name,
        "mode": MODES[mode]["label"],
        "video_duration_seconds": round(duration, 3),
        "processing_seconds": round(seconds, 3),
        "processing_minutes": round(seconds / 60, 4),
        "processing_min_per_video_sec": round(per_sec, 6),
        "status": status,
        "error": error,
    }


def write_timing_csv(path: Path, rows: list[dict]) -> None:
    """Per-pair rows followed by an AVERAGE row over non-error rows."""
    ok = [r for r in rows if r["status"] != "error"]
    out = list(rows)
    if ok:
        n = len(ok)
        avg_sec = sum(r["processing_seconds"] for r in ok) / n
        out.append({
            "video_name": "AVERAGE",
            "mode": "",
            "video_duration_seconds": round(sum(r["video_duration_seconds"] for r in ok) / n, 3),
            "processing_seconds": round(avg_sec, 3),
            "processing_minutes": round(avg_sec / 60, 4),
            "processing_min_per_video_sec": round(
                sum(r["processing_min_per_video_sec"] for r in ok) / n, 6
            ),
            "status": f"over {n} video-mode pairs",
            "error": "",
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(out)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def process_mode(processor, model, model_name, frames, timestamps, fps,
                 video_name, video_dir, mode, duration) -> dict:
    """Run chunk inference + merge for one video-mode pair, write its files, return the timing row."""
    mode_dir = video_dir / MODES[mode]["folder"]
    mode_dir.mkdir(parents=True, exist_ok=True)

    cuda_sync()
    start = time.perf_counter()
    chunk_results, raw_chunks = [], []
    try:
        for start_frame in range(0, len(frames), CHUNK_SIZE):
            end_frame = min(start_frame + CHUNK_SIZE, len(frames))
            chunk_number = start_frame // CHUNK_SIZE + 1

            chunk_result, raw_text = describe_chunk(
                processor, model, frames[start_frame:end_frame], chunk_number, mode,
            )
            chunk_result = apply_chunk_timestamps(chunk_result, timestamps[start_frame:end_frame])
            end_time = (
                timestamps[end_frame - 1] + 1 / fps
                if end_frame == len(frames)
                else timestamps[end_frame]
            )
            chunk_record = {
                "chunk_number": chunk_number,
                "start_time": format_time(timestamps[start_frame]),
                "end_time": format_time(end_time),
                **chunk_result,
            }
            chunk_results.append(chunk_record)
            raw_chunks.append({
                "chunk_number": chunk_number,
                "start_time": chunk_record["start_time"],
                "end_time": chunk_record["end_time"],
                "raw_response": raw_text,
                "parsed_response": chunk_result,
            })

        merged, merge_raw = merge_video_results(processor, model, chunk_results, mode)
        cuda_sync()
        elapsed = time.perf_counter() - start
    except Exception as exc:
        cuda_sync()
        return timing_row(video_name, mode, duration, time.perf_counter() - start, "error", str(exc))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matches = merged["action_matches"]
    write_timestamps_csv(mode_dir / "timestamps.csv", video_name, mode, matches)
    write_description(mode_dir / "Description.txt", merged, chunk_results)
    render_table_png(
        mode_dir / "Action-Codes Table.png",
        f"{video_name} - {MODES[mode]['folder']}",
        matches,
    )
    (mode_dir / "Raw_Response.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "video": video_name,
                "mode": MODES[mode]["label"],
                "chunks": raw_chunks,
                "merge": {"raw_response": merge_raw, "parsed_response": merged},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return timing_row(video_name, mode, duration, elapsed,
                      "parse_error" if merged.get("parse_error") else "success")


def run(args) -> Path:
    video_paths = find_videos(args.video_folder)

    run_dir = args.results_dir / f"{safe_name(args.model)}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_prompt_file(run_dir, args.model)
    print(f"Run folder: {run_dir}", flush=True)

    timing_rows: list[dict] = []
    timing_csv = run_dir / "Total_Generation_Time.csv"

    try:
        processor, model = load_model(args.model)
    except Exception as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        for vp in video_paths:
            for mode in MODES_TO_RUN:
                timing_rows.append(timing_row(vp.stem, mode, 0.0, 0.0, "error", f"model load failed: {exc}"))
        write_timing_csv(timing_csv, timing_rows)
        raise SystemExit(1)

    for index, video_path in enumerate(video_paths, start=1):
        name = video_path.stem
        video_dir = run_dir / f"Video-{safe_name(name)}"
        print(f"\n[{index}/{len(video_paths)}] {video_path.name}", flush=True)

        try:
            frames, timestamps, fps, duration = extract_frames(
                video_path, SAMPLE_EVERY_SECONDS, MAX_IMAGE_SIZE,
            )
        except Exception as exc:
            print(f"  failed before inference: {exc}", flush=True)
            for mode in MODES_TO_RUN:
                timing_rows.append(timing_row(name, mode, 0.0, 0.0, "error", str(exc)))
            write_timing_csv(timing_csv, timing_rows)
            continue

        for mode in MODES_TO_RUN:
            print(f"  mode={MODES[mode]['label']}", flush=True)
            row = process_mode(processor, model, args.model, frames, timestamps, fps,
                               name, video_dir, mode, duration)
            timing_rows.append(row)
            write_timing_csv(timing_csv, timing_rows)
            print(f"  -> {row['processing_seconds']:.2f}s ({row['status']})", flush=True)

    ok = [r for r in timing_rows if r["status"] != "error"]
    if ok:
        avg = sum(r["processing_seconds"] for r in ok) / len(ok)
        print(f"\nAverage processing time: {avg:.2f} s ({avg / 60:.3f} min)")
    else:
        print("\nNo video-mode pairs completed successfully.")
    print(f"Results saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3-VL video analysis pipeline.")
    parser.add_argument("--video-folder", type=Path, default=DEFAULT_VIDEO_DIR,
                        help="Folder containing videos (default: videos)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Root folder for run outputs (default: Results)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model name (default: {DEFAULT_MODEL})")
    run(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)