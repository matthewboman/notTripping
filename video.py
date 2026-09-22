import argparse
from pathlib import Path
import shutil
import subprocess
import time

import cv2

from deepdream import DeepDream
from dream_utils import (
  blend_frames,
  frame_to_tensor,
  get_device,
  resize_to_dimensions,
  tensor_to_frame,
  warp_feedback,
)


EFAULT_OUTPUT_WIDTH = 0

DEFAULT_DREAM_STEPS = 3
DEFAULT_STEP_SIZE = 0.018
DEFAULT_FEEDBACK = 0.35

DEFAULT_OCTAVES = 2
DEFAULT_OCTAVE_SCALE = 2.0

DEFAULT_FLOW_WIDTH = 480
DEFAULT_MAX_WHOLE_FRAME_WIDTH = 768
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 96

DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"


def parse_args():
  parser = argparse.ArgumentParser(
    description="High-resolution multi-octave DeepDream video renderer",
  )

  parser.add_argument("input", help="Input video path")
  parser.add_argument("output", help="Output MP4 path")

  parser.add_argument(
    "--output-width",
    type=int,
    default=DEFAULT_OUTPUT_WIDTH,
    help="Final AND neural-processing width. 0 preserves source resolution.",
  )

  parser.add_argument(
    "--steps",
    type=int,
    default=DEFAULT_DREAM_STEPS,
    help="Dream iterations performed at every octave.",
  )

  parser.add_argument(
    "--step-size",
    type=float,
    default=DEFAULT_STEP_SIZE,
    help="Strength of each gradient-ascent update.",
  )

  parser.add_argument(
    "--feedback",
    type=float,
    default=DEFAULT_FEEDBACK,
    help="Amount of motion-aligned previous dream carried into the next frame.",
  )

  parser.add_argument(
    "--octaves",
    type=int,
    default=DEFAULT_OCTAVES,
    help="Number of scales used to build large hallucinated structure.",
  )

  parser.add_argument(
    "--octave-scale",
    type=float,
    default=DEFAULT_OCTAVE_SCALE,
    help="Resolution multiplier between successive dream octaves.",
  )

  parser.add_argument(
    "--flow-width",
    type=int,
    default=DEFAULT_FLOW_WIDTH,
    help="Reduced width used only for optical-flow calculation.",
  )

  parser.add_argument(
    "--max-whole-frame-width",
    type=int,
    default=DEFAULT_MAX_WHOLE_FRAME_WIDTH,
    help=(
      "Final-octave frames wider than this use overlapping VGG tiles. "
      "Set 0 to force whole-frame VGG processing."
    ),
  )

  parser.add_argument(
    "--tile-size",
    type=int,
    default=DEFAULT_TILE_SIZE,
  )

  parser.add_argument(
    "--tile-overlap",
    type=int,
    default=DEFAULT_TILE_OVERLAP,
  )

  parser.add_argument(
    "--start",
    type=float,
    default=0.0,
    help="Start time in seconds.",
  )

  parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Render only this many seconds. 0 renders to the end.",
  )

  parser.add_argument(
    "--model",
    choices=["vgg16", "googlenet"],
    default="vgg16",
  )

  parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
  parser.add_argument("--preset", default=DEFAULT_PRESET)

  return parser.parse_args()


def output_dimensions(frame, output_width):
  """Calculate even output dimensions while preserving aspect ratio."""
  source_height, source_width = frame.shape[:2]

  if output_width <= 0:
    width = source_width
    height = source_height
  else:
    width = output_width
    height = round(source_height * (width / source_width))

  width -= width % 2
  height -= height % 2

  return width, height


def open_encoder(
  input_path,
  output_path,
  width,
  height,
  fps,
  start,
  duration,
  crf,
  preset,
):
  """Encode processed BGR frames as H.264 and preserve source audio."""
  ffmpeg = shutil.which("ffmpeg")

  if ffmpeg is None:
    raise RuntimeError(
      "ffmpeg is required. Install it with: sudo apt install ffmpeg"
    )

  command = [
    ffmpeg,
    "-y",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}",
    "-r", str(fps),
    "-i", "-",
  ]

  if start > 0:
    command += ["-ss", str(start)]

  command += ["-i", str(input_path)]

  if duration > 0:
    command += ["-t", str(duration)]

  command += [
    "-map", "0:v:0",
    "-map", "1:a:0?",
    "-c:v", "libx264",
    "-profile:v", "high",
    "-preset", preset,
    "-crf", str(crf),
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    "-movflags", "+faststart",
    str(output_path),
  ]

  return subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
  )


def format_duration(seconds):
  """Format elapsed/remaining render time."""
  if seconds is None:
    return "--:--:--"

  seconds = max(0, int(seconds))
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)

  return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
  args = parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)

  if not input_path.exists():
    raise FileNotFoundError(input_path)

  if input_path.resolve() == output_path.resolve():
    raise ValueError("Input and output paths must be different")

  device = get_device()
  capture = cv2.VideoCapture(str(input_path))

  if not capture.isOpened():
    raise RuntimeError(f"Could not open input video: {input_path}")

  fps = capture.get(cv2.CAP_PROP_FPS)

  if not fps or fps <= 0:
    capture.release()
    raise RuntimeError("Could not determine source video frame rate")

  start_frame = round(args.start * fps)

  if start_frame > 0:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

  success, first_frame = capture.read()

  if not success:
    capture.release()
    raise RuntimeError("Could not read the first requested video frame")

  output_width, output_height = output_dimensions(
    first_frame,
    args.output_width,
  )

  max_frames = (
    round(args.duration * fps)
    if args.duration > 0
    else 0
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)

  encoder = open_encoder(
    input_path=input_path,
    output_path=output_path,
    width=output_width,
    height=output_height,
    fps=fps,
    start=args.start,
    duration=args.duration,
    crf=args.crf,
    preset=args.preset,
  )

  dreamer = DeepDream(
    device=device,
    steps=args.steps,
    step_size=args.step_size,
    octaves=args.octaves,
    octave_scale=args.octave_scale,
    max_whole_frame_width=args.max_whole_frame_width,
    tile_size=args.tile_size,
    tile_overlap=args.tile_overlap,
    model_name=args.model,
  )

  previous_dream = None
  previous_frame = None

  rendered_frames = 0
  started_at = time.perf_counter()
  pending_frame = first_frame

  print(f"Input: {input_path}")
  print(f"Output: {output_path}")
  print(f"Device: {device}")
  print(f"Processing/output resolution: {output_width}x{output_height}")
  print(f"FPS: {fps:.3f}")
  print(f"Octaves: {args.octaves}")
  print(f"Octave scale: {args.octave_scale}")
  print(f"Steps per octave: {args.steps}")
  print(f"Step size: {args.step_size}")
  print(f"Feedback: {args.feedback}")
  print()

  try:
    while True:
      if pending_frame is not None:
        frame = pending_frame
        pending_frame = None
        success = True
      else:
        success, frame = capture.read()

      if not success:
        break

      if max_frames > 0 and rendered_frames >= max_frames:
        break

      # Source, feedback, DeepDream, and encoded output all use this exact
      # resolution. There is no hidden low-resolution dream stage.
      frame = resize_to_dimensions(
        frame,
        output_width,
        output_height,
      )

      current = frame_to_tensor(frame, device)
      feedback = None

      if previous_dream is not None and previous_frame is not None:
        feedback = warp_feedback(
          previous_dream,
          previous_frame,
          frame,
          device,
          flow_width=args.flow_width,
        )

      dream_input = blend_frames(
        current,
        feedback,
        args.feedback,
      )

      dreamed = dreamer.dream(dream_input)

      previous_dream = dreamed
      previous_frame = frame.copy()

      output_frame = tensor_to_frame(dreamed)

      output_lab = cv2.cvtColor(output_frame, cv2.COLOR_BGR2LAB)
      source_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

      output_lab[:, :, 1:] = source_lab[:, :, 1:]

      output_frame = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

      encoder.stdin.write(output_frame.tobytes())

      rendered_frames += 1

      elapsed = time.perf_counter() - started_at
      seconds_per_frame = elapsed / rendered_frames

      if max_frames > 0:
        remaining_frames = max(0, max_frames - rendered_frames)
        eta = seconds_per_frame * remaining_frames
        progress = rendered_frames / max_frames * 100
        progress_text = f"{progress:6.2f}%"
      else:
        eta = None
        progress_text = f"{rendered_frames} frames"

      print(
        f"\r{progress_text} | "
        f"{seconds_per_frame:.2f}s/frame | "
        f"ETA {format_duration(eta)}",
        end="",
        flush=True,
      )

  finally:
    capture.release()

    if encoder.stdin:
      encoder.stdin.close()

    return_code = encoder.wait()

  print()

  if return_code != 0:
    raise RuntimeError(
      f"ffmpeg failed with exit code {return_code}"
    )

  elapsed = time.perf_counter() - started_at

  print(
    f"Rendered {rendered_frames} frames in "
    f"{format_duration(elapsed)}"
  )


if __name__ == "__main__":
  main()