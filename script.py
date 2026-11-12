import argparse
import html
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

# Must be set BEFORE importing torch
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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"

VIDEO_FOLDER = Path("videos")
OUTPUT_FOLDER = Path("resultsNew")

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

VIDEO_PROMPT = """
Analyze these frames as one continuous video chunk.

Create a concise but useful descriptor for this chunk. Include what is
happening, people, objects, important actions, environment/location, visible
text, and the sequence of events. Do not describe each frame separately.
Only report actions that are visibly supported by the frames. It is acceptable
to give a score of 0 when an action is not observed.

Classify the chunk using every action code below. For each code, return a score
from 0 to 100, where 0 means not observed and 100 means clearly observed.
Include exact keywords or short phrases from the descriptor that support the
score. Do not invent actions that are not visible.

Action codes:
{action_codes}

Return only valid JSON with this exact shape:
{{
    "descriptor": "one paragraph of approximately 50-100 words",
    "action_matches": [
        {{"action": "action code", "score": 0, "keywords": [], "evidence": ""}}
    ]
}}
""".strip()

# Take one frame every N seconds
SAMPLE_EVERY_SECONDS = 0.5

# Number of frames sent to Qwen at once
#
# For an RTX 4060 Ti 8GB:
# Start with 4.
#
# If it works comfortably, try 6 or 8.
CHUNK_SIZE = 8

# Maximum dimension of each image.
#
# Reducing this significantly reduces vision encoder VRAM usage.
MAX_IMAGE_SIZE = 640

# Maximum tokens generated for the descriptor
MAX_NEW_TOKENS = 500

# GPU memory reserved for the model.
#
# RTX 4060 Ti reports approximately 7.64 GiB.
# Keep some VRAM available for vision processing.
GPU_MEMORY_GIB = 6

# CPU RAM available for model offloading.
#
# CHANGE THIS according to your machine.
# For a 32GB RAM machine, 24GiB is reasonable.
CPU_MEMORY_GIB = 24


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def print_system_info():

    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    if torch.cuda.is_available():

        props = torch.cuda.get_device_properties(0)

        total_vram = props.total_memory / (1024 ** 3)

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {total_vram:.2f} GiB")
        print(f"VRAM used:    {allocated:.2f} GiB")
        print(f"VRAM reserved:{reserved:.2f} GiB")
        print(f"CUDA:         {torch.version.cuda}")

    else:
        print("CUDA is NOT available.")

    print("=" * 60)


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(
    video_path,
    sample_every_seconds=1.0,
    max_dimension=640,
):
    """
    Extract frames from the video and resize them
    before sending them to Qwen.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    frame_interval = max(
        1,
        int(fps * sample_every_seconds)
    )

    frames = []
    timestamps = []

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_interval == 0:

            # BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            height, width = frame.shape[:2]

            # Resize while preserving aspect ratio
            scale = min(
                max_dimension / width,
                max_dimension / height,
                1.0,
            )

            if scale < 1.0:

                new_width = int(
                    width * scale
                )

                new_height = int(
                    height * scale
                )

                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            frames.append(
                Image.fromarray(frame)
            )
            timestamps.append(frame_index / fps)

        frame_index += 1

    cap.release()

    return frames, timestamps, fps


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    if torch.cuda.is_available():
        # Allow Transformers/Accelerate to distribute
        # the model between GPU VRAM and CPU RAM.
        max_memory = {
            0: f"{GPU_MEMORY_GIB}GiB",
            "cpu": f"{CPU_MEMORY_GIB}GiB",
        }

        print("\nMemory configuration:")
        print(f"GPU:  {GPU_MEMORY_GIB} GiB")
        print(f"CPU:  {CPU_MEMORY_GIB} GiB")

        model = AutoModelForMultimodalLM.from_pretrained(

            MODEL_NAME,

            # Keep the model's native FP8 representation
            torch_dtype="auto",

            # Automatically distribute model layers
            device_map="auto",

            # GPU first, CPU RAM if required
            max_memory=max_memory,

            low_cpu_mem_usage=True,
        )
        device_label = model.hf_device_map
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


# ============================================================
# PROCESS ONE VIDEO CHUNK
# ============================================================

def describe_chunk(
    processor,
    model,
    frames,
    chunk_number,
    start_time,
    end_time,
):

    print(
        f"\nProcessing chunk {chunk_number}"
        f" ({len(frames)} frames)"
    )

    # --------------------------------------------------------
    # Build multimodal content
    # --------------------------------------------------------

    content = []

    for frame in frames:

        content.append({
            "type": "image",
            "image": frame,
        })

    content.append({
        "type": "text",
        "text": VIDEO_PROMPT.format(
            action_codes="\n".join(
                f"- {action}" for action in ACTION_CODES
            )
        )
    })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,

        add_generation_prompt=True,

        tokenize=True,

        return_dict=True,

        return_tensors="pt",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # When using device_map="auto", don't use:
    #
    # inputs.to(model.device)
    #
    # because the model is distributed between
    # GPU and CPU.
    # --------------------------------------------------------

    first_device = next(
        model.parameters()
    ).device

    for key in inputs:

        if hasattr(inputs[key], "to"):

            inputs[key] = inputs[key].to(
                first_device
            )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

        )

    # --------------------------------------------------------
    # Remove prompt tokens
    # --------------------------------------------------------

    input_length = (
        inputs["input_ids"]
        .shape[-1]
    )

    generated_tokens = (
        outputs[0][input_length:]
    )

    description = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del inputs
    del outputs
    del generated_tokens

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return parse_model_json(description)


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

    return {
        "descriptor": cleaned,
        "action_matches": [],
        "parse_error": "Model did not return valid JSON.",
    }


def normalize_result(result):
    """Ensure every requested action has a predictable result entry."""
    matches_by_action = {
        match.get("action"): match
        for match in result.get("action_matches", [])
        if isinstance(match, dict) and match.get("action")
    }

    matches = []
    for action in ACTION_CODES:
        match = matches_by_action.get(action, {})
        try:
            score = int(match.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        matches.append({
            "action": action,
            "score": max(0, min(100, score)),
            "keywords": match.get("keywords", []),
            "evidence": match.get("evidence", ""),
        })

    return {
        "descriptor": str(result.get("descriptor", "")).strip(),
        "action_matches": matches,
        **({"parse_error": result["parse_error"]}
           if result.get("parse_error") else {}),
    }


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


def build_html_report(results, output_path):
    sections = []

    for video in results:
        chunks = []
        video_file = Path(
            browser_video_path(video["video_path"], output_path)
        )
        relative_video_path = os.path.relpath(
            video_file,
            start=output_path.parent,
        )
        video_url = quote(
            relative_video_path.replace(os.sep, "/"),
            safe="/:",
        )

        for chunk in video["chunks"]:
            rows = []
            for match in chunk["action_matches"]:
                keywords = ", ".join(match["keywords"])
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(match['action'])}</td>"
                    f"<td><strong>{match['score']}</strong></td>"
                    f"<td>{html.escape(keywords)}</td>"
                    f"<td>{html.escape(match['evidence'])}</td>"
                    "</tr>"
                )

            chunks.append(
                "<article class='chunk'>"
                f"<h3>Chunk {chunk['chunk_number']} | "
                f"{html.escape(chunk['start_time'])} - "
                f"{html.escape(chunk['end_time'])}</h3>"
                f"<p>{html.escape(chunk['descriptor'])}</p>"
                "<table><thead><tr><th>Action</th><th>Score</th>"
                "<th>Keywords</th><th>Evidence</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></article>"
            )

        sections.append(
            "<section class='video'>"
            f"<h2>{html.escape(video['video_name'])}</h2>"
            f"<video controls preload='metadata' src='{html.escape(video_url)}'></video>"
            f"<p class='meta'>Duration: {format_time(video['duration_seconds'])} | "
            f"Chunks: {len(video['chunks'])}</p>"
            f"{''.join(chunks)}</section>"
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
.chunk { border-top: 1px solid #d9dee5; padding-top: 16px; margin-top: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; vertical-align: top; padding: 8px; border-bottom: 1px solid #e5e7eb; }
th { background: #eef2f5; }
th:nth-child(2), td:nth-child(2) { width: 70px; text-align: center; }
td:nth-child(3) { width: 20%; }
@media (max-width: 800px) { main { padding: 12px; } table { display: block; overflow-x: auto; white-space: normal; } }
</style>
</head>
<body><main><h1>Video action analysis</h1>
<p>Each chunk includes its time range, descriptor, action score, supporting keywords, and evidence.</p>
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

    return saved


def save_results(results, results_path, report_path):
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_html_report(results, report_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze every video in a folder with Qwen3-VL."
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default=VIDEO_FOLDER,
        help="Folder containing video files (default: videos)",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Folder for JSON and HTML output (default: results)",
    )
    args = parser.parse_args()

    video_paths = sorted(
        path for path in args.video_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        }
    )

    if not video_paths:
        raise RuntimeError(
            f"No supported videos found in {args.video_folder}"
        )

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
    build_html_report(results, report_path)
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
            "chunks": [],
        })
        video_result["video_path"] = video_path.as_posix()
        video_result["duration_seconds"] = duration_seconds

        chunks_by_number = {
            chunk.get("chunk_number"): chunk
            for chunk in video_result.get("chunks", [])
            if chunk.get("chunk_number")
        }
        expected_chunks = (
            len(range(0, len(frames), CHUNK_SIZE))
        )

        if len(chunks_by_number) >= expected_chunks:
            print("Already complete; keeping saved results.")
            results_by_video[video_path.name] = video_result
            continue

        if processor is None or model is None:
            processor, model = load_model()

        for start in range(0, len(frames), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(frames))
            chunk_number = start // CHUNK_SIZE + 1

            if chunk_number in chunks_by_number:
                print(f"Chunk {chunk_number}: already saved; skipping.")
                continue

            start_time = timestamps[start]
            end_time = (
                timestamps[end - 1] + 1 / fps
                if end == len(frames)
                else timestamps[end]
            )

            result = describe_chunk(
                processor,
                model,
                frames[start:end],
                chunk_number=chunk_number,
                start_time=start_time,
                end_time=end_time,
            )
            result = normalize_result(result)
            chunk_result = {
                "chunk_number": chunk_number,
                "start_time_seconds": start_time,
                "end_time_seconds": end_time,
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                **result,
            }
            video_result["chunks"].append(chunk_result)
            video_result["chunks"].sort(
                key=lambda chunk: chunk["chunk_number"]
            )
            results_by_video[video_path.name] = video_result
            results = list(results_by_video.values())
            save_results(results, results_path, report_path)

            print(
                f"Chunk {chunk_number}: {chunk_result['start_time']} - "
                f"{chunk_result['end_time']}"
            )
            print(chunk_result["descriptor"])

        results_by_video[video_path.name] = video_result
        results = list(results_by_video.values())
        save_results(results, results_path, report_path)
        print(f"Saved {results_path} and {report_path}")

    print(f"\nDone. Processed {len(results)} video(s).")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()