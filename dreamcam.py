import argparse
import platform
import time
from functools import lru_cache

import cv2
import numpy as np
import torch

from deepdream import DeepDream

WINDOW_TITLE = "DreamCam"

DEFAULT_CAMERA_INDEX = 0
DEFAULT_PROCESSING_WIDTH = 192
DEFAULT_DREAM_STEPS = 2
DEFAULT_STEP_SIZE = 0.022
DEFAULT_FEEDBACK = 0.42

STATUS_POSITION = (20, 35)
STATUS_FONT_SCALE = 0.7
STATUS_THICKNESS = 2

# Farneback optical-flow settings. These values favor responsiveness at the
# small processing resolutions used by DreamCam.
FLOW_PYRAMID_SCALE = 0.5
FLOW_LEVELS = 2
FLOW_WINDOW_SIZE = 15
FLOW_ITERATIONS = 2
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2


def get_device():
  """Select the fastest PyTorch backend available on the current machine."""
  if torch.cuda.is_available():
    return torch.device("cuda")

  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")

  if hasattr(torch, "xpu") and torch.xpu.is_available():
    return torch.device("xpu")

  return torch.device("cpu")


def frame_to_tensor(frame, device):
  """Convert an OpenCV BGR frame into a normalized NCHW RGB tensor."""
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
  tensor = tensor.to(device=device, dtype=torch.float32)
  tensor.div_(255.0)

  if device.type == "cpu":
    tensor = tensor.contiguous(memory_format=torch.channels_last)

  return tensor


def tensor_to_frame(tensor):
  """Convert a DreamCam RGB tensor back into an OpenCV BGR uint8 frame."""
  image = tensor.detach().squeeze(0)
  image = image.permute(1, 2, 0)
  image = image.clamp(0.0, 1.0)
  image = image.cpu().numpy()

  image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

  # RGB -> BGR without cvtColor, which also avoids CV_8S issues
  return image[:, :, ::-1].copy()


def resize_for_processing(frame, width):
  """Resize a camera frame while preserving its aspect ratio."""
  height, original_width = frame.shape[:2]

  scale = width / original_width
  resized_height = int(height * scale)

  return cv2.resize(
    frame,
    (width, resized_height),
    interpolation=cv2.INTER_AREA,
  )


def blend_frames(current, feedback, amount):
  """
  Mix fresh video with the motion-aligned previous dream.

  Lower feedback reacts faster to live video. Higher feedback preserves more of
  the evolving hallucination between frames.
  """
  if feedback is None or amount <= 0:
    return current

  return ( current * (1.0 - amount) + feedback * amount).clamp(0.0, 1.0)


@lru_cache(maxsize=4)
def remap_grid(height, width):
  """Cache the static pixel-coordinate grid used by optical-flow remapping."""
  grid_x, grid_y = np.meshgrid(
    np.arange(width, dtype=np.float32),
    np.arange(height, dtype=np.float32),
  )

  return grid_x, grid_y


def warp_feedback(previous_dream, previous_frame, current_frame, device):
  """
  Move the previous hallucination with motion in the live camera image.

  Optical flow is calculated from the previous webcam frame to the current
  webcam frame. The previous dream is then remapped along that motion before it
  is mixed into the new frame, reducing stationary ghost trails.
  """
  previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
  current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  flow = cv2.calcOpticalFlowFarneback(
    previous_gray,
    current_gray,
    None,
    FLOW_PYRAMID_SCALE,
    FLOW_LEVELS,
    FLOW_WINDOW_SIZE,
    FLOW_ITERATIONS,
    FLOW_POLY_N,
    FLOW_POLY_SIGMA,
    0,
  )

  height, width = current_gray.shape
  grid_x, grid_y = remap_grid(height, width)

  map_x = grid_x - flow[:, :, 0]
  map_y = grid_y - flow[:, :, 1]

  previous_image = tensor_to_frame(previous_dream)

  warped = cv2.remap(
    previous_image,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT,
  )

  return frame_to_tensor(warped, device)


def draw_status(frame, device, fps, paused):
  """Draw the active compute backend and measured dream-update FPS."""
  status = f"{device.type.upper()} | {fps:.1f} FPS"

  if paused:
    status += " | PAUSED"

  cv2.putText(
    frame,
    status,
    STATUS_POSITION,
    cv2.FONT_HERSHEY_SIMPLEX,
    STATUS_FONT_SCALE,
    (255, 255, 255),
    STATUS_THICKNESS,
    cv2.LINE_AA,
  )


def open_camera(camera_index):
  """Open the requested webcam with the appropriate macOS backend when needed."""
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


def parse_args():
  parser = argparse.ArgumentParser(
    description="Real-time recursive DeepDream webcam",
  )

  parser.add_argument(
    "--camera",
    type=int,
    default=DEFAULT_CAMERA_INDEX,
    help="OpenCV camera index.",
  )

  parser.add_argument(
    "--width",
    type=int,
    default=DEFAULT_PROCESSING_WIDTH,
    help="Width used for neural processing. Lower values are much faster.",
  )

  parser.add_argument(
    "--steps",
    type=int,
    default=DEFAULT_DREAM_STEPS,
    help="DeepDream gradient iterations performed for each video update.",
  )

  parser.add_argument(
    "--step-size",
    type=float,
    default=DEFAULT_STEP_SIZE,
    help="Strength of each individual DeepDream gradient update.",
  )

  parser.add_argument(
    "--feedback",
    type=float,
    default=DEFAULT_FEEDBACK,
    help="Fraction of the motion-aligned previous dream mixed into the new frame.",
  )

  parser.add_argument(
    "--mirror",
    action="store_true",
    help="Mirror the webcam image horizontally.",
  )

  return parser.parse_args()


def main():
  args = parse_args()
  device = get_device()

  print(f"PyTorch: {torch.__version__}")
  print(f"Device: {device}")
  print(f"Processing width: {args.width}")
  print(f"Dream steps: {args.steps}")
  print()
  print("Controls:")
  print("  q / Esc quit")
  print("  Space   pause")
  print("  r       reset feedback")

  dreamer = DeepDream(
    device=device,
    steps=args.steps,
    step_size=args.step_size,
  )

  capture = open_camera(args.camera)

  previous_dream = None
  previous_frame = None
  paused = False
  last_output = None

  fps = 0.0
  fps_frames = 0
  fps_started = time.perf_counter()

  try:
    while True:
      if not paused:
        success, frame = capture.read()

        if not success:
          print("Failed to read webcam frame.")
          break

        if args.mirror:
          frame = cv2.flip(frame, 1)

        original_height, original_width = frame.shape[:2]
        processed_frame = resize_for_processing(frame, args.width)

        # Convert the current webcam frame exactly once.
        current = frame_to_tensor(processed_frame, device)

        feedback = None

        if previous_dream is not None and previous_frame is not None:
          feedback = warp_feedback(
            previous_dream,
            previous_frame,
            processed_frame,
            device,
          )

        dream_input = blend_frames(
          current,
          feedback,
          args.feedback,
        )

        # Classic DeepDream directly amplifies features in the recurrent image.
        # The live webcam remains coupled through dream_input and optical flow.
        dreamed = dreamer.dream(dream_input)

        previous_dream = dreamed
        previous_frame = processed_frame.copy()

        output = tensor_to_frame(dreamed)
        last_output = cv2.resize(
          output,
          (original_width, original_height),
          interpolation=cv2.INTER_LINEAR,
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

        draw_status(
          display,
          device,
          fps,
          paused,
        )

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