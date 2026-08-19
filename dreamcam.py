import argparse
import platform
import time

import cv2
import numpy as np
import torch

from deepdream import DeepDream

def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda")

  if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    return torch.device("mps")

  if (hasattr(torch, "xpu") and torch.xpu.is_available()):
    return torch.device("xpu")

  return torch.device("cpu")

def frame_to_tensor(frame, device):
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  tensor = torch.from_numpy(frame)
  tensor = tensor.permute(2, 0, 1)
  tensor = tensor.unsqueeze(0)
  tensor = tensor.float() / 255.0

  return tensor.to(device)

def tensor_to_frame(tensor):
  image = tensor.detach().squeeze(0)
  image = image.permute(1, 2, 0)
  image = image.clamp(0.0, 1.0)
  image = image.cpu().numpy()

  image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

  return image[:, :, ::-1].copy()

def resize_for_processing(frame, width):
  height, original_width = frame.shape[:2]

  scale = width / original_width
  resized_height = int(height * scale)

  return cv2.resize(
    frame,
    (width, resized_height),
    interpolation=cv2.INTER_AREA
  )

def blend_frames(current, feedback, amount):
  if feedback is None or amount <= 0:
    return current

  return (current * (1.0 - amount) + feedback * amount).clamp(0.0, 1.0)


def warp_dream(dreamed, current_frame, device, amount):
    if amount <= 0:
        return dreamed

    dream_frame = tensor_to_frame(dreamed)

    dream_gray = cv2.cvtColor(
        dream_frame,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32) / 255.0

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32) / 255.0

    residual = dream_gray - current_gray

    residual = cv2.GaussianBlur(
        residual,
        (0, 0),
        2.5,
    )

    gradient_x = cv2.Sobel(
        current_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        current_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = np.sqrt(
        gradient_x ** 2
        + gradient_y ** 2
    )

    normal_x = gradient_x / (magnitude + 1e-6)
    normal_y = gradient_y / (magnitude + 1e-6)

    edge_scale = np.percentile(magnitude, 90)

    edge_mask = np.clip(
        magnitude / (edge_scale + 1e-6),
        0.0,
        1.0,
    )

    edge_mask = cv2.GaussianBlur(
        edge_mask,
        (0, 0),
        2.0,
    )

    displacement = (
        np.tanh(residual * 6.0)
        * edge_mask
        * amount
    )

    height, width = current_gray.shape

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    map_x = grid_x + normal_x * displacement
    map_y = grid_y + normal_y * displacement

    warped = cv2.remap(
        dream_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return frame_to_tensor(
        warped,
        device,
    )

def warp_feedback(previous_dream, previous_frame, current_frame, device):
  previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
  current_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  flow = cv2.calcOpticalFlowFarneback(
    previous_gray,
    current_gray,
    None,
    0.5,
    3,
    15,
    3,
    5,
    1.2,
    0,
  )

  height, width = current_gray.shape

  grid_x, grid_y = np.meshgrid(
    np.arange(width),
    np.arange(height),
  )

  map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
  map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

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
    cv2.LINE_AA
  )

def open_camera(camera_index):
  if platform.system() == "Darwin":
    capture = cv2.VideoCapture(
      camera_index,
      cv2.CAP_AVFOUNDATION
    )
  else:
    capture = cv2.VideoCapture(camera_index)

  if not capture.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}")

  capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  return capture

def parse_args():
  parser = argparse.ArgumentParser(
    description="Real-time DeepDream webcam"
  )

  parser.add_argument(
    "--camera",
    type=int,
    default=0,
  )

  parser.add_argument(
    "--width",
    type=int,
    default=320,
  )

  # how many DeepDream gradient-ascent passes happen per video frame
  parser.add_argument(
    "--steps",
    type=int,
    default=1,
  )

  # 0.005  subtle
  # 0.015  noticeable
  # 0.025  strong
  # 0.05   aggressive
  parser.add_argument(
    "--step-size",
    type=float,
    default=0.02,
  )

  parser.add_argument(
    "--layer",
    type=int,
    default=23,
  )

  # how much of the previous dreamed frame survives into the next frame versus how much fresh webcam imagery comes in
  parser.add_argument(
    "--feedback",
    type=float,
    default=0.15,
  )

  parser.add_argument(
    "--feedback-gain",
    type=float,
    default=1.15,
  )

  parser.add_argument(
    "--warp",
    type=float,
    default=4.0,
  )

  parser.add_argument(
    "--mirror",
    action="store_true"
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
    step_size=args.step_size
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

        tensor = frame_to_tensor(processed_frame, device)

        feedback = None

        if previous_dream is not None and previous_frame is not None:
          feedback = warp_feedback(
            previous_dream,
            previous_frame,
            processed_frame,
            device,
          )

        tensor = blend_frames(
          tensor,
          feedback,
          args.feedback,
        )

        dreamed = dreamer.dream(
          tensor,
          frame_to_tensor(
            processed_frame,
            device,
          ),
        )
        dreamed = warp_dream(
          dreamed,
          processed_frame,
          device,
          args.warp,
        )
        previous_dream = dreamed.detach()
        previous_frame = processed_frame.copy()

        output = tensor_to_frame(dreamed)
        output = cv2.resize(
          output,
          (original_width, original_height),
          interpolation=cv2.INTER_LINEAR
        )
        last_output = output

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
            paused
          )

          cv2.imshow("DreamCam", display)

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