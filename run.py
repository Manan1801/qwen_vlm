import argparse
import html
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

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

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


MODEL_NAME = os.environ.get(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
)

VIDEO_FOLDER = Path("videos")
OUTPUT_FOLDER = Path("resultsNew")

MODES = ["fixed_codes", "open_discovery"]

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


def print_system_info():

    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    if torch.cuda.is_available():

        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {total_vram:.2f} GiB")
        print(f"VRAM used:    {allocated:.2f} GiB")
        print(f"VRAM reserved:{reserved:.2f} GiB")
        print(f"CUDA:         {torch.version.cuda}")

    else:
        print("CUDA is NOT available.")

    print("=" * 60)


SAMPLE_EVERY_SECONDS = 0.5
CHUNK_SIZE = 8
MAX_IMAGE_SIZE = 640
MAX_NEW_TOKENS = 500
MAX_NEW_TOKENS_MERGE = 4096
GPU_MEMORY_GIB = 20
CPU_MEMORY_GIB = 48



def extract_frames(
    video_path,
    sample_every_seconds=1.0,
    max_dimension=640,
):
    """Extract frames from the video and resize them before sending to Qwen."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    frame_interval = max(1, int(fps * sample_every_seconds))

    frames = []
    timestamps = []
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
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_AREA,
                )

            frames.append(Image.fromarray(frame))
            timestamps.append(frame_index / fps)

        frame_index += 1

    cap.release()

    return frames, timestamps, fps


def infer_model_device_label(model):
    """Return a user-friendly device label even when hf_device_map is absent."""
    if hasattr(model, "hf_device_map"):
        return model.hf_device_map

    try:
        first_param = next(model.parameters())
        device = first_param.device
        if device.type == "cuda":
            return {"model": f"cuda:{device.index}" if device.index is not None else "cuda"}
        return {"model": str(device)}
    except StopIteration:
        return {"model": "unknown"}


def load_model():

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Loading model...")

    if torch.cuda.is_available():
        max_memory = {
            0: f"{GPU_MEMORY_GIB}GiB",
            "cpu": f"{CPU_MEMORY_GIB}GiB",
        }

        print("\nMemory configuration:")
        print(f"GPU:  {GPU_MEMORY_GIB} GiB")
        print(f"CPU:  {CPU_MEMORY_GIB} GiB")

        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        device_label = infer_model_device_label(model)
    else:
        print("CUDA is not available; falling back to CPU execution.")
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model = model.to(torch.device("cpu"))
        device_label = {"model": "cpu"}

    model.eval()

    print("\nModel loaded.")
    print("\nDevice map:")
    print(device_label)

    return processor, model


def run_model(processor, model, content, max_new_tokens):
    """Run one chat-completion call (image+text or text-only) and return decoded text."""

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
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    text = processor.decode(generated_tokens, skip_special_tokens=True)

    del inputs
    del outputs
    del generated_tokens

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text


def describe_chunk(processor, model, frames, chunk_number, mode):

    print(f"\nProcessing chunk {chunk_number} ({len(frames)} frames) [{mode}]")

    content = [{"type": "image", "image": frame} for frame in frames]

    if mode == "fixed_codes":
        prompt_text = CHUNK_PROMPT_FIXED_CODES.format(
            n_frames=len(frames),
            max_index=len(frames) - 1,
            action_codes="\n".join(f"- {action}" for action in ACTION_CODES),
        )
    elif mode == "open_discovery":
        prompt_text = CHUNK_PROMPT_OPEN_DISCOVERY.format(
            n_frames=len(frames),
            max_index=len(frames) - 1,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    content.append({"type": "text", "text": prompt_text})

    text = run_model(processor, model, content, MAX_NEW_TOKENS)

    return normalize_chunk_result(parse_model_json(text), mode)


def parse_model_json(text):
    """Parse JSON even when a model surrounds it with a markdown fence."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
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


def normalize_chunk_result(result, mode):
    """Ensure a predictable shape for a single chunk's result, keyed by frame_indices."""

    if mode == "fixed_codes":
        matches_by_action = {
            match.get("action"): match
            for match in result.get("action_matches", [])
            if isinstance(match, dict) and match.get("action")
        }

        matches = []
        for action in ACTION_CODES:
            match = matches_by_action.get(action, {})
            matches.append({
                "action": action,
                "score": safe_score(match.get("score", 0)),
                "keywords": safe_str_list(match.get("keywords", [])),
                "evidence": str(match.get("evidence", "")),
                "frame_indices": safe_int_list(match.get("frame_indices", [])),
            })
    else:
        matches = []
        for match in result.get("action_matches", []):
            if not isinstance(match, dict) or not match.get("action"):
                continue
            matches.append({
                "action": str(match["action"]).strip(),
                "score": safe_score(match.get("score", 0)),
                "keywords": safe_str_list(match.get("keywords", [])),
                "evidence": str(match.get("evidence", "")),
                "frame_indices": safe_int_list(match.get("frame_indices", [])),
            })

    return {
        "descriptor": str(result.get("descriptor", "")).strip(),
        "action_matches": matches,
        **({"parse_error": result["parse_error"]} if result.get("parse_error") else {}),
    }


def normalize_merge_result(result, mode):
    """Ensure a predictable shape for the final merged video-level result, keyed by timestamps."""

    if mode == "fixed_codes":
        matches_by_action = {
            match.get("action"): match
            for match in result.get("action_matches", [])
            if isinstance(match, dict) and match.get("action")
        }

        matches = []
        for action in ACTION_CODES:
            match = matches_by_action.get(action, {})
            matches.append({
                "action": action,
                "score": safe_score(match.get("score", 0)),
                "keywords": safe_str_list(match.get("keywords", [])),
                "evidence": str(match.get("evidence", "")),
                "timestamps": safe_str_list(match.get("timestamps", [])),
            })
    else:
        matches = []
        for match in result.get("action_matches", []):
            if not isinstance(match, dict) or not match.get("action"):
                continue
            matches.append({
                "action": str(match["action"]).strip(),
                "score": safe_score(match.get("score", 0)),
                "keywords": safe_str_list(match.get("keywords", [])),
                "evidence": str(match.get("evidence", "")),
                "timestamps": safe_str_list(match.get("timestamps", [])),
            })

    return {
        "descriptor": str(result.get("descriptor", "")).strip(),
        "action_matches": matches,
        **({"parse_error": result["parse_error"]} if result.get("parse_error") else {}),
    }


def apply_chunk_timestamps(chunk_result, chunk_frame_timestamps):
    """Convert each match's frame_indices (local to this chunk) into real,
    absolute video timestamps using the chunk's known per-frame times."""

    for match in chunk_result["action_matches"]:
        indices = match.pop("frame_indices", [])
        stamps = []
        for index in indices:
            if 0 <= index < len(chunk_frame_timestamps):
                stamps.append(format_time(chunk_frame_timestamps[index]))
        match["timestamps"] = sorted(set(stamps))

    return chunk_result


def build_chunk_summaries(chunk_results):
    lines = []
    for chunk in chunk_results:
        lines.append(
            f"Chunk {chunk['chunk_number']} ({chunk['start_time']} - {chunk['end_time']}):"
        )
        lines.append(f"Descriptor: {chunk['descriptor']}")
        lines.append("Actions:")
        for match in chunk["action_matches"]:
            if match["score"] <= 0 and not match["timestamps"]:
                continue
            keywords = ", ".join(match["keywords"])
            timestamps = ", ".join(match["timestamps"])
            lines.append(
                f"- {match['action']}: score={match['score']}, "
                f"timestamps=[{timestamps}], keywords=[{keywords}], "
                f"evidence={match['evidence']}"
            )
        lines.append("")
    return "\n".join(lines)


def merge_video_results(processor, model, chunk_results, mode):
    """One text-only call that synthesizes all of a video's chunk results
    (for a single mode) into one descriptor and one merged action list."""

    print(f"\nMerging {len(chunk_results)} chunks for mode={mode}")

    chunk_summaries = build_chunk_summaries(chunk_results)

    if mode == "fixed_codes":
        prompt_text = MERGE_PROMPT_FIXED_CODES.format(
            chunk_summaries=chunk_summaries,
            action_codes="\n".join(f"- {action}" for action in ACTION_CODES),
        )
    elif mode == "open_discovery":
        prompt_text = MERGE_PROMPT_OPEN_DISCOVERY.format(chunk_summaries=chunk_summaries)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    content = [{"type": "text", "text": prompt_text}]
    text = run_model(processor, model, content, MAX_NEW_TOKENS_MERGE)

    return normalize_merge_result(parse_model_json(text), mode)


def format_time(seconds):
    minutes, remaining = divmod(max(0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes:02d}:{remaining:05.2f}"


def browser_video_path(video_path, report_path):
    """Create a cached H.264 copy for browsers that lack MPEG-4 Part 2."""
    source = Path(video_path)
    if not source.exists():
        return video_path

    web_folder = report_path.parent / "web_videos"
    web_folder.mkdir(parents=True, exist_ok=True)
    converted = web_folder / f"{source.stem}_browser.mp4"

    if not converted.exists() or converted.stat().st_mtime < source.stat().st_mtime:
        print(f"Creating browser-compatible video: {converted}")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(source),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(converted),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"Could not convert {source}: {error}")
            return str(source.resolve())

    return str(converted.resolve())


MODE_LABELS = {
    "fixed_codes": "Fixed action codes",
    "open_discovery": "Open discovery",
}


def build_html_report(results, output_path):
    sections = []

    for video in results:
        video_file = Path(browser_video_path(video["video_path"], output_path))
        relative_video_path = os.path.relpath(video_file, start=output_path.parent)
        video_url = quote(relative_video_path.replace(os.sep, "/"), safe="/:")

        mode_cards = []
        for mode in MODES:
            mode_result = video.get("modes", {}).get(mode)
            if not mode_result:
                continue

            rows = []
            for match in mode_result["action_matches"]:
                if match["score"] <= 0 and not match["timestamps"]:
                    continue
                keywords = ", ".join(match["keywords"])
                timestamps = ", ".join(match["timestamps"])
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(match['action'])}</td>"
                    f"<td><strong>{match['score']}</strong></td>"
                    f"<td>{html.escape(keywords)}</td>"
                    f"<td>{html.escape(match['evidence'])}</td>"
                    f"<td>{html.escape(timestamps)}</td>"
                    "</tr>"
                )

            mode_cards.append(
                "<article class='mode-block'>"
                f"<h3>{html.escape(MODE_LABELS.get(mode, mode))}</h3>"
                f"<p>{html.escape(mode_result['descriptor'])}</p>"
                "<table><thead><tr><th>Action</th><th>Score</th>"
                "<th>Keywords</th><th>Evidence</th><th>Timestamps</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></article>"
            )

        sections.append(
            "<section class='video'>"
            f"<h2>{html.escape(video['video_name'])}</h2>"
            f"<video controls preload='metadata' src='{html.escape(video_url)}'></video>"
            f"<p class='meta'>Duration: {format_time(video['duration_seconds'])}</p>"
            f"{''.join(mode_cards)}</section>"
        )

    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video action analysis</title>
<style>
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #18202a; }
main { max-width: 1500px; margin: auto; padding: 24px; }
h1 { margin-top: 0; }
.video { background: white; padding: 20px; margin: 24px 0; border-radius: 8px; box-shadow: 0 2px 10px #0001; }
video { width: min(100%, 900px); max-height: 520px; background: #111; }
.meta { color: #52606d; }
.mode-block { border-top: 1px solid #d9dee5; padding-top: 16px; margin-top: 24px; }
.mode-block h3 { display: inline-block; background: #4a6fa5; color: white; padding: 2px 10px; border-radius: 999px; font-size: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; vertical-align: top; padding: 8px; border-bottom: 1px solid #e5e7eb; }
th { background: #eef2f5; }
th:nth-child(2), td:nth-child(2) { width: 70px; text-align: center; }
td:nth-child(3) { width: 18%; }
td:nth-child(5) { width: 18%; }
@media (max-width: 800px) { main { padding: 12px; } table { display: block; overflow-x: auto; white-space: normal; } }
</style>
</head>
<body><main><h1>Video action analysis</h1>
<p>Each video shows one merged descriptor and one merged action list per mode, covering the full video.</p>
""" + "\n".join(sections) + "\n</main></body></html>"

    output_path.write_text(document, encoding="utf-8")


def load_saved_results(results_path):
    if not results_path.exists():
        return []

    try:
        saved = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read existing results file {results_path}: {error}"
        ) from error

    if not isinstance(saved, list):
        raise RuntimeError(f"Expected a list in {results_path}")

    for video in saved:
        for mode, mode_result in video.get("modes", {}).items():
            raw_descriptor = mode_result.get("descriptor", "")
            if not mode_result.get("parse_error") or not raw_descriptor.lstrip().startswith("{"):
                continue
            recovered = parse_model_json(raw_descriptor)
            if recovered.get("descriptor") and not recovered["descriptor"].lstrip().startswith("{"):
                video["modes"][mode] = normalize_merge_result(recovered, mode)

    return saved


def save_results(results, results_path, report_path):
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_html_report(results, report_path)


def main():

    parser = argparse.ArgumentParser(
        description="Analyze every video in a folder with Qwen3-VL."
    )
    parser.add_argument(
        "--video-folder", type=Path, default=VIDEO_FOLDER,
        help="Folder containing video files (default: videos)",
    )
    parser.add_argument(
        "--output-folder", type=Path, default=OUTPUT_FOLDER,
        help="Folder for JSON and HTML output (default: resultsNew)",
    )
    args = parser.parse_args()

    video_paths = sorted(
        path for path in args.video_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        }
    )

    if not video_paths:
        raise RuntimeError(f"No supported videos found in {args.video_folder}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    results_path = args.output_folder / "video_results.json"
    report_path = args.output_folder / "video_report.html"

    print_system_info()

    results = load_saved_results(results_path)
    results_by_video = {
        result.get("video_name"): result
        for result in results
        if result.get("video_name")
    }
    results = list(results_by_video.values())
    save_results(results, results_path, report_path)
    processor = None
    model = None

    for video_path in video_paths:
        print(f"\n{'=' * 60}\nProcessing {video_path}\n{'=' * 60}")

        frames, timestamps, fps = extract_frames(
            str(video_path),
            sample_every_seconds=SAMPLE_EVERY_SECONDS,
            max_dimension=MAX_IMAGE_SIZE,
        )

        if not frames:
            print("No frames extracted; skipping video.")
            continue

        cap = cv2.VideoCapture(str(video_path))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_seconds = frame_count / fps if frame_count > 0 else timestamps[-1]

        video_result = results_by_video.get(video_path.name, {
            "video_name": video_path.name,
            "video_path": video_path.as_posix(),
            "duration_seconds": duration_seconds,
            "sample_every_seconds": SAMPLE_EVERY_SECONDS,
            "chunk_size_frames": CHUNK_SIZE,
            "chunks": {},
            "modes": {},
        })
        video_result["video_path"] = video_path.as_posix()
        video_result["duration_seconds"] = duration_seconds
        video_result.setdefault("chunks", {})
        video_result.setdefault("modes", {})

        expected_chunks = len(range(0, len(frames), CHUNK_SIZE))

        already_done = all(
            mode in video_result["modes"]
            and not video_result["modes"][mode].get("parse_error")
            and len(video_result["chunks"].get(mode, [])) >= expected_chunks
            for mode in MODES
        )
        if already_done:
            print("Already complete; keeping saved results.")
            continue

        if processor is None or model is None:
            processor, model = load_model()

        for mode in MODES:
            video_result["chunks"].setdefault(mode, [])
            chunks_by_number = {
                chunk["chunk_number"]: chunk
                for chunk in video_result["chunks"][mode]
            }

            for start in range(0, len(frames), CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, len(frames))
                chunk_number = start // CHUNK_SIZE + 1

                if chunk_number in chunks_by_number:
                    continue

                start_time = timestamps[start]
                end_time = (
                    timestamps[end - 1] + 1 / fps
                    if end == len(frames)
                    else timestamps[end]
                )

                chunk_result = describe_chunk(
                    processor, model, frames[start:end],
                    chunk_number=chunk_number, mode=mode,
                )
                chunk_result = apply_chunk_timestamps(chunk_result, timestamps[start:end])
                chunk_result = {
                    "chunk_number": chunk_number,
                    "start_time_seconds": start_time,
                    "end_time_seconds": end_time,
                    "start_time": format_time(start_time),
                    "end_time": format_time(end_time),
                    **chunk_result,
                }

                video_result["chunks"][mode].append(chunk_result)
                video_result["chunks"][mode].sort(key=lambda c: c["chunk_number"])
                results_by_video[video_path.name] = video_result
                results = list(results_by_video.values())
                save_results(results, results_path, report_path)

                print(f"Chunk {chunk_number} [{mode}]: {chunk_result['start_time']} - {chunk_result['end_time']}")
                print(chunk_result["descriptor"])

            if mode in video_result["modes"] and not video_result["modes"][mode].get("parse_error"):
                continue

            video_result["modes"][mode] = merge_video_results(
                processor, model, video_result["chunks"][mode], mode,
            )
            results_by_video[video_path.name] = video_result
            results = list(results_by_video.values())
            save_results(results, results_path, report_path)

        print(f"Saved {results_path} and {report_path}")

    print(f"\nDone. Processed {len(results)} video(s).")


if __name__ == "__main__":
    main()