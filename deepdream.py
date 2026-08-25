import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16, GoogLeNet_Weights, googlenet

# https://distill.pub/2017/feature-visualization
GOOGLENET_DREAM_LAYER_WEIGHTS = {
  "mixed3a": 0.25,
  "mixed4a": 0.15,
  "mixed4d": 0.35,
  "mixed4e": 0.25,
}

# VGG16 activation layers.
VGG_RELU2_3 = 8
VGG_RELU3_3 = 15
VGG_RELU4_3 = 22
VGG_RELU5_3 = 29

DREAM_LAYER_WEIGHTS = {
  VGG_RELU3_3: 0.10,
  VGG_RELU4_3: 0.45,
  VGG_RELU5_3: 0.45,
}

DEEPEST_DREAM_LAYER = max(DREAM_LAYER_WEIGHTS)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRADIENT_EPSILON = 1e-8

# High-resolution DeepDream strategy:
# 1. Dream a coarse copy to generate large coherent hallucinations.
# 2. Upscale the dreamed residual into larger octaves.
# 3. Finish at the full requested frame resolution.
DEFAULT_OCTAVES = 3
DEFAULT_OCTAVE_SCALE = 1.8

# Full-resolution VGG activations can consume a large amount of memory.
# Only the final high-resolution octave is tiled when necessary. Coarser octaves
# remain whole-frame so large hallucinated structures can form coherently.
DEFAULT_MAX_WHOLE_FRAME_WIDTH = 768
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 96


class DeepDream:
  """Reusable multi-octave DeepDream engine."""

  def __init__(
    self,
    device,
    steps=2,
    step_size=0.018,
    octaves=DEFAULT_OCTAVES,
    octave_scale=DEFAULT_OCTAVE_SCALE,
    max_whole_frame_width=DEFAULT_MAX_WHOLE_FRAME_WIDTH,
    tile_size=DEFAULT_TILE_SIZE,
    tile_overlap=DEFAULT_TILE_OVERLAP,
    model_name="vgg16",
  ):
    self.device = device
    self.steps = steps
    self.step_size = step_size
    self.octaves = max(1, octaves)
    self.octave_scale = max(1.01, octave_scale)
    self.max_whole_frame_width = max_whole_frame_width
    self.tile_size = tile_size
    self.tile_overlap = tile_overlap
    self.model_name = model_name

    if self.model_name == "vgg16":
      self.model = vgg16(
        weights=VGG16_Weights.DEFAULT
      ).features[:DEEPEST_DREAM_LAYER + 1]

      self.dream_layer_weights = {
        15: 0.10,
        22: 0.45,
        29: 0.45,
      }

    elif self.model_name == "googlenet":
      self.model = googlenet(
        weights=GoogLeNet_Weights.DEFAULT,
        aux_logits=True,
      )

      self.dream_layer_weights = GOOGLENET_DREAM_LAYER_WEIGHTS

    else:
      raise ValueError(f"Unknown model: {self.model_name}")

    self.model.eval()
    self.model.to(device)

    for parameter in self.model.parameters():
      parameter.requires_grad_(False)

    self.use_channels_last = device.type == "cpu"

    if self.use_channels_last:
      self.model.to(memory_format=torch.channels_last)

    self.mean = torch.tensor(
      IMAGENET_MEAN,
      device=device,
    ).view(1, 3, 1, 1)

    self.std = torch.tensor(
      IMAGENET_STD,
      device=device,
    ).view(1, 3, 1, 1)

  def normalize(self, image):
    """Normalize RGB image values for VGG16."""
    return (image - self.mean) / self.std

  def dream_loss(self, image):
    """Weighted activation-maximization objective."""
    if self.model_name == "vgg16":
      return self._vgg_dream_loss(image)

    return self._googlenet_dream_loss(image)

  def _vgg_dream_loss(self, image):
    features = self.normalize(image)
    loss = image.new_zeros(())

    for index, layer in enumerate(self.model):
      features = layer(features)

      weight = self.dream_layer_weights.get(index)

      if weight is not None:
        loss = loss + features.square().mean() * weight

    return loss

  def _googlenet_dream_loss(self, image):
    x = self.normalize(image)
    loss = image.new_zeros(())

    x = self.model.conv1(x)
    x = self.model.maxpool1(x)
    x = self.model.conv2(x)
    x = self.model.conv3(x)
    x = self.model.maxpool2(x)

    x = self.model.inception3a(x)

    loss += (
      x.square().mean()
      * self.dream_layer_weights["mixed3a"]
    )

    x = self.model.inception3b(x)
    x = self.model.maxpool3(x)

    x = self.model.inception4a(x)

    loss += (
      x.square().mean()
      * self.dream_layer_weights["mixed4a"]
    )

    x = self.model.inception4b(x)
    x = self.model.inception4c(x)
    x = self.model.inception4d(x)

    loss += (
      x.square().mean()
      * self.dream_layer_weights["mixed4d"]
    )

    x = self.model.inception4e(x)

    loss += (
      x.square().mean()
      * self.dream_layer_weights["mixed4e"]
    )

    return loss

  def dream(self, image):
    """
    Dream at progressively larger scales and finish at full resolution.

    The coarse octave creates large patterns. We carry only the dreamed residual
    into the next octave so the live/source image remains structurally present
    while hallucinated detail accumulates across scales.
    """
    original = image.detach().to(self.device)
    pyramid = self._build_octave_pyramid(original)

    detail = None
    dreamed = None

    for octave_index, octave_base in enumerate(pyramid):
      if detail is None:
        octave_input = octave_base
      else:
        detail = F.interpolate(
          detail,
          size=octave_base.shape[-2:],
          mode="bilinear",
          align_corners=False,
        )

        octave_input = (
          octave_base + detail
        ).clamp(0.0, 1.0)

      is_final_octave = octave_index == len(pyramid) - 1

      dreamed = self._dream_octave(
        octave_input,
        allow_tiling=is_final_octave,
      )

      # Carry hallucinated structure forward without replacing the next octave's
      # full-resolution source image.
      detail = dreamed - octave_base

    return dreamed

  def _build_octave_pyramid(self, image):
    """
    Build octave images from coarse to full resolution.

    The last octave is always exactly the input frame's resolution.
    """
    _, _, full_height, full_width = image.shape
    sizes = []

    for level in reversed(range(self.octaves)):
      scale = self.octave_scale ** level

      width = max(32, round(full_width / scale))
      height = max(32, round(full_height / scale))

      sizes.append((height, width))

    sizes[-1] = (full_height, full_width)

    pyramid = []

    for height, width in sizes:
      if height == full_height and width == full_width:
        octave = image
      else:
        octave = F.interpolate(
          image,
          size=(height, width),
          mode="bilinear",
          align_corners=False,
        )

      pyramid.append(octave)

    return pyramid

  def _dream_octave(self, image, allow_tiling):
    """
    Dream one octave.

    Whole-frame processing is preferred because it preserves spatial coherence.
    Tiling is used only for the final large octave when required for memory.
    """
    width = image.shape[-1]

    if (
      allow_tiling
      and self.max_whole_frame_width > 0
      and width > self.max_whole_frame_width
    ):
      return self._dream_tiled(image)

    return self._dream_tensor(image)

  def _dream_tensor(self, image):
    """Run repeated gradient-ascent steps on one tensor."""
    dreamed = image.detach().clone()

    if self.use_channels_last:
      dreamed = dreamed.contiguous(memory_format=torch.channels_last)

    for _ in range(self.steps):
      dreamed.requires_grad_(True)

      loss = self.dream_loss(dreamed)

      gradient = torch.autograd.grad(
        loss,
        dreamed,
        retain_graph=False,
        create_graph=False,
      )[0]

      gradient = gradient / gradient.std().clamp_min(GRADIENT_EPSILON)

      dreamed = (
        dreamed
        + gradient * self.step_size
      ).clamp(0.0, 1.0).detach()

    return dreamed

  def _dream_tiled(self, image):
    """
    Refine a large final octave with overlapping tiles.

    Large-scale hallucinations already exist from the coarse octaves, so tiling
    here adds full-resolution detail without reducing the dream to tiny noise.
    """
    _, _, height, width = image.shape

    tile_size = min(self.tile_size, height, width)
    overlap = min(self.tile_overlap, tile_size // 4)
    stride = max(1, tile_size - overlap)

    y_starts = self._tile_starts(height, tile_size, stride)
    x_starts = self._tile_starts(width, tile_size, stride)

    output = torch.zeros_like(image)
    weights = torch.zeros(
      (1, 1, height, width),
      device=image.device,
      dtype=image.dtype,
    )

    weight_mask = self._tile_weight(
      tile_size,
      tile_size,
      overlap,
      image.device,
      image.dtype,
    )

    for y in y_starts:
      for x in x_starts:
        tile = image[
          :,
          :,
          y:y + tile_size,
          x:x + tile_size,
        ]

        tile_height = tile.shape[-2]
        tile_width = tile.shape[-1]

        dreamed_tile = self._dream_tensor(tile)

        tile_weight = weight_mask[
          :,
          :,
          :tile_height,
          :tile_width,
        ]

        output[
          :,
          :,
          y:y + tile_height,
          x:x + tile_width,
        ] += dreamed_tile * tile_weight

        weights[
          :,
          :,
          y:y + tile_height,
          x:x + tile_width,
        ] += tile_weight

    return output / weights.clamp_min(GRADIENT_EPSILON)

  @staticmethod
  def _tile_starts(length, tile_size, stride):
    """Tile offsets that always cover the far edge."""
    if length <= tile_size:
      return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size

    if starts[-1] != final_start:
      starts.append(final_start)

    return starts

  @staticmethod
  def _tile_weight(height, width, overlap, device, dtype):
    """Soft mask for feathering overlapping tile boundaries."""
    if overlap <= 0:
      return torch.ones(
        (1, 1, height, width),
        device=device,
        dtype=dtype,
      )

    edge_y = min(overlap, height // 2)
    edge_x = min(overlap, width // 2)

    y_weight = torch.ones(height, device=device, dtype=dtype)
    x_weight = torch.ones(width, device=device, dtype=dtype)

    y_ramp = torch.linspace(
      0.05,
      1.0,
      edge_y,
      device=device,
      dtype=dtype,
    )

    x_ramp = torch.linspace(
      0.05,
      1.0,
      edge_x,
      device=device,
      dtype=dtype,
    )

    y_weight[:edge_y] = y_ramp
    y_weight[-edge_y:] = torch.flip(y_ramp, dims=[0])

    x_weight[:edge_x] = x_ramp
    x_weight[-edge_x:] = torch.flip(x_ramp, dims=[0])

    return (
      y_weight[:, None] * x_weight[None, :]
    ).view(1, 1, height, width)