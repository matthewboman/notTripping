from functools import lru_cache

import cv2
import numpy as np
import torch


FLOW_PYRAMID_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINDOW_SIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# Flow is intentionally cheaper than the neural pass. Motion is calculated at a
# reduced width and scaled back to the full video frame.
DEFAULT_FLOW_WIDTH = 480


def get_device():
  """Select the fastest PyTorch backend available."""
  if torch.cuda.is_available():
    return torch.device("cuda")

  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")

  if hasattr(torch, "xpu") and torch.xpu.is_available():
    return torch.device("xpu")

  return torch.device("cpu")


def frame_to_tensor(frame, device):
  """Convert OpenCV BGR uint8 data to an RGB float tensor."""
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
  tensor = tensor.to(device=device, dtype=torch.float32)
  tensor.div_(255.0)

  if device.type == "cpu":
    tensor = tensor.contiguous(memory_format=torch.channels_last)

  return tensor


def tensor_to_frame(tensor):
  """Convert an RGB float tensor back to OpenCV BGR uint8."""
  image = tensor.detach().squeeze(0)
  image = image.permute(1, 2, 0)
  image = image.clamp(0.0, 1.0)
  image = image.cpu().numpy()

  image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

  return image[:, :, ::-1].copy()


def resize_to_dimensions(frame, width, height):
  """Resize a source frame to the exact requested render dimensions."""
  if frame.shape[1] == width and frame.shape[0] == height:
    return frame

  interpolation = (
    cv2.INTER_AREA
    if width < frame.shape[1]
    else cv2.INTER_CUBIC
  )

  return cv2.resize(
    frame,
    (width, height),
    interpolation=interpolation,
  )


def blend_frames(current, feedback, amount):
  """Blend current video with the motion-aligned previous dreamed frame."""
  if feedback is None or amount <= 0:
    return current

  return (
    current * (1.0 - amount)
    + feedback * amount
  ).clamp(0.0, 1.0)


def warp_feedback(
  previous_dream,
  previous_frame,
  current_frame,
  device,
  flow_width=DEFAULT_FLOW_WIDTH,
):
  """
  Warp the prior full-resolution dream along source-video motion.

  Motion estimation is reduced-resolution for speed. The flow vectors are then
  resized and scaled to the full processing resolution.
  """
  full_height, full_width = current_frame.shape[:2]

  if flow_width > 0 and full_width > flow_width:
    scale = flow_width / full_width
    flow_height = max(1, round(full_height * scale))

    previous_flow_frame = cv2.resize(
      previous_frame,
      (flow_width, flow_height),
      interpolation=cv2.INTER_AREA,
    )

    current_flow_frame = cv2.resize(
      current_frame,
      (flow_width, flow_height),
      interpolation=cv2.INTER_AREA,
    )
  else:
    previous_flow_frame = previous_frame
    current_flow_frame = current_frame

  previous_gray = cv2.cvtColor(
    previous_flow_frame,
    cv2.COLOR_BGR2GRAY,
  )

  current_gray = cv2.cvtColor(
    current_flow_frame,
    cv2.COLOR_BGR2GRAY,
  )

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

  flow_height, flow_width_actual = flow.shape[:2]

  if flow_height != full_height or flow_width_actual != full_width:
    scale_x = full_width / flow_width_actual
    scale_y = full_height / flow_height

    flow = cv2.resize(
      flow,
      (full_width, full_height),
      interpolation=cv2.INTER_LINEAR,
    )

    flow[:, :, 0] *= scale_x
    flow[:, :, 1] *= scale_y

  grid_x, grid_y = remap_grid(full_height, full_width)

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


@lru_cache(maxsize=8)
def remap_grid(height, width):
  """Cache the coordinate grid used for optical-flow remapping."""
  return np.meshgrid(
    np.arange(width, dtype=np.float32),
    np.arange(height, dtype=np.float32),
  )