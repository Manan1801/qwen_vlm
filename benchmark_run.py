import argparse
import csv
import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import torch

import run


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SCRIPT_DIR = Path(__file__).resolve().parent


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_csv_row(csv_file, writer, row):
    writer.writerow(row)
    csv_file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark video processing without saving generated results."
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default=SCRIPT_DIR / "videos",
        help="Folder containing videos (default: videos)",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_DIR / "processing_benchmark_with_gpu.csv",
        help="CSV file for timing results (default: processing_benchmark_with_gpu.csv)",
    )
    args = parser.parse_args()

    video_folder = args.video_folder.resolve()
    csv_output = args.csv_output.resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        path for path in video_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ) if video_folder.is_dir() else []

    if not video_paths:
        raise RuntimeError(f"No supported videos found in {video_folder}")

    processor = None
    model = None
    timings = []

    with csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_name", "processing_seconds", "status"])

        try:
            with redirect_stdout(io.StringIO()):
                processor, model = run.load_model()
        except Exception as error:
            for video_path in video_paths:
                write_csv_row(
                    csv_file,
                    writer,
                    [video_path.name, "", f"error: model load failed: {error}"],
                )
            writer.writerow(["AVERAGE", "", "error: model load failed"])
            print(f"Model load failed: {error}")
            return 1

        for video_path in video_paths:
            print(f"Processing {video_path.name}...")
            cuda_sync()
            start = time.perf_counter()

            try:
                with redirect_stdout(io.StringIO()):
                    frames, timestamps, fps = run.extract_frames(
                        str(video_path),
                        sample_every_seconds=run.SAMPLE_EVERY_SECONDS,
                        max_dimension=run.MAX_IMAGE_SIZE,
                    )

                    if not frames:
                        raise RuntimeError("No frames extracted")

                    for mode in run.MODES:
                        chunk_results = []

                        for start_frame in range(0, len(frames), run.CHUNK_SIZE):
                            end_frame = min(start_frame + run.CHUNK_SIZE, len(frames))
                            chunk_number = start_frame // run.CHUNK_SIZE + 1
                            chunk_result = run.describe_chunk(
                                processor,
                                model,
                                frames[start_frame:end_frame],
                                chunk_number=chunk_number,
                                mode=mode,
                            )
                            chunk_result = run.apply_chunk_timestamps(
                                chunk_result,
                                timestamps[start_frame:end_frame],
                            )
                            chunk_results.append({
                                "chunk_number": chunk_number,
                                "start_time": run.format_time(timestamps[start_frame]),
                                "end_time": run.format_time(
                                    timestamps[end_frame - 1] + 1 / fps
                                    if end_frame == len(frames)
                                    else timestamps[end_frame]
                                ),
                                **chunk_result,
                            })

                        run.merge_video_results(
                            processor,
                            model,
                            chunk_results,
                            mode,
                        )

                cuda_sync()
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                write_csv_row(
                    csv_file,
                    writer,
                    [video_path.name, f"{elapsed:.3f}", "success"],
                )
                print(f"{video_path.name}: {elapsed:.3f} seconds")

            except Exception as error:
                cuda_sync()
                elapsed = time.perf_counter() - start
                write_csv_row(
                    csv_file,
                    writer,
                    [video_path.name, f"{elapsed:.3f}", f"error: {error}"],
                )
                print(f"{video_path.name}: ERROR ({error})")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        average = sum(timings) / len(timings) if timings else 0.0
        write_csv_row(
            csv_file,
            writer,
            ["AVERAGE", f"{average:.3f}", "success" if timings else "no successful videos"],
        )
        print(f"Average: {average:.3f} seconds")
        print(f"Saved timings to {csv_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
