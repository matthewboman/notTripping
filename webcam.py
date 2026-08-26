import argparse
import platform
import time

import cv2
import torch

from deepdream import DeepDream
from dream_utils import (
  blend_frames,
  frame_to_tensor,
  get_device,
  tensor_to_frame,
  warp_feedback,
)


WINDOW_TITLE = "DreamCam"

DEFAULT_CAMERA_INDEX = 0
DEFAULT_PROCESSING_WIDTH = 192
DEFAULT_DREAM_STEPS = 3
DEFAULT_STEP_SIZE = 0.018
DEFAULT_FEEDBACK = 0.42


def resize_to_width(frame, width):
  """Resize webcam processing while preserving aspect ratio."""
  height, original_width = frame.shape[:2]
  scale = width / original_width
  resized_height = max(1, round(height * scale))

  return cv2.resize(
    frame,
    (width, resized_height),
    interpolation=cv2.INTER_AREA,
  )


def open_camera(camera_index):
  """Open the requested webcam."""
  if platform.system() == "Darwin":
    capture = cv2.VideoCapture(
      camera_index,
      cv2.CAP_AVFOUNDATION,
    )
  else:
    capture = cv2.VideoCapture(camera_index)

  if not capture.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}")

  capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  return capture


def draw_status(frame, device, fps, paused):
  status = f"{device.type.upper()} | {fps:.1f} FPS"

  if paused:
    status += " | PAUSED"

  cv2.putText(
    frame,
    status,
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
  )


def parse_args():
  parser = argparse.ArgumentParser(
    description="Real-time recursive DeepDream webcam",
  )

  parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
  parser.add_argument("--width", type=int, default=DEFAULT_PROCESSING_WIDTH)
  parser.add_argument("--steps", type=int, default=DEFAULT_DREAM_STEPS)
  parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
  parser.add_argument("--feedback", type=float, default=DEFAULT_FEEDBACK)
  parser.add_argument("--model", choices=["vgg16", "googlenet"], default="vgg16")
  parser.add_argument("--octaves", type=int, default=1)
  parser.add_argument("--octave-scale", type=float, default=2.0)
  parser.add_argument("--mirror", action="store_true")

  return parser.parse_args()


def main():
  args = parse_args()
  device = get_device()

  # Webcam remains single-octave for speed and preserves the behavior already
  # established during live experimentation.
  dreamer = DeepDream(
    device=device,
    steps=args.steps,
    step_size=args.step_size,
    octaves=args.octaves,
    octave_scale=args.octave_scale,
    max_whole_frame_width=0,
    model_name=args.model,
  )

  print(f"Requested model: {args.model}")
  print(f"Loaded model: {type(dreamer.model).__name__}")

  capture = open_camera(args.camera)

  previous_dream = None
  previous_frame = None
  paused = False
  last_output = None

  fps = 0.0
  fps_frames = 0
  fps_started = time.perf_counter()

  print(f"PyTorch: {torch.__version__}")
  print(f"Device: {device}")
  print(f"Processing width: {args.width}")
  print(f"Dream steps: {args.steps}")

  try:
    while True:
      if not paused:
        success, frame = capture.read()

        if not success:
          break

        if args.mirror:
          frame = cv2.flip(frame, 1)

        display_height, display_width = frame.shape[:2]
        processed_frame = resize_to_width(frame, args.width)

        current = frame_to_tensor(processed_frame, device)
        feedback = None

        if previous_dream is not None and previous_frame is not None:
          feedback = warp_feedback(
            previous_dream,
            previous_frame,
            processed_frame,
            device,
            flow_width=processed_frame.shape[1],
          )

        dream_input = blend_frames(
          current,
          feedback,
          args.feedback,
        )

        dreamed = dreamer.dream(dream_input)

        previous_dream = dreamed
        previous_frame = processed_frame.copy()

        output = tensor_to_frame(dreamed)

        last_output = cv2.resize(
          output,
          (display_width, display_height),
          interpolation=cv2.INTER_CUBIC,
        )

        fps_frames += 1
        now = time.perf_counter()
        elapsed = now - fps_started

        if elapsed >= 1.0:
          fps = fps_frames / elapsed
          fps_frames = 0
          fps_started = now

      if last_output is not None:
        display = last_output.copy()
        draw_status(display, device, fps, paused)
        cv2.imshow(WINDOW_TITLE, display)

      key = cv2.waitKey(1) & 0xFF

      if key == ord("q") or key == 27:
        break

      if key == ord(" "):
        paused = not paused

      if key == ord("r"):
        previous_dream = None
        previous_frame = None

  finally:
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()