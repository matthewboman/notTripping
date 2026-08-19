import torch
from torchvision.models import VGG16_Weights, vgg16

# VGG16 feature indices. These are ReLU activations immediately before pooling.
VGG_RELU3_3 = 15
VGG_RELU4_3 = 22
VGG_RELU5_3 = 29

# Mid/deep layers dominate so the result forms recognizable hallucinated patterns
# instead of only enhancing edges.
DREAM_LAYER_WEIGHTS = {
  VGG_RELU3_3: 0.25, # shape fragments
  VGG_RELU4_3: 0.45, # semantic-ish structure
  VGG_RELU5_3: 0.30, # strongly semantic features
}

DEEPEST_DREAM_LAYER = max(DREAM_LAYER_WEIGHTS)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRADIENT_EPSILON = 1e-8


class DeepDream:
  """Runs classic activation-maximizing DeepDream on an image tensor."""

  def __init__(self, device, steps=2, step_size=0.02):
    self.device = device
    self.steps = steps
    self.step_size = step_size

    self.model = vgg16(
      weights=VGG16_Weights.DEFAULT
    ).features[:DEEPEST_DREAM_LAYER + 1]

    self.model.eval()
    self.model.to(device)

    for parameter in self.model.parameters():
      parameter.requires_grad_(False)

    # Channels-last is optimized for many CPU convolution workloads. Keep the
    # default memory format on MPS/XPU/CUDA.
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
    """Normalize RGB image data to the distribution expected by VGG16."""
    return (image - self.mean) / self.std

  def dream_loss(self, image):
    """
    Compute the classic DeepDream objective.

    Each selected VGG activation is encouraged to become stronger. We accumulate
    the scalar loss during the forward pass instead of retaining a dictionary of
    feature tensors that is only used once.
    """
    features = self.normalize(image)
    loss = image.new_zeros(())

    for index, layer in enumerate(self.model):
      features = layer(features)

      weight = DREAM_LAYER_WEIGHTS.get(index)

      if weight is not None:
        loss = loss + features.square().mean() * weight

    return loss

  def dream(self, image):
    """
    Repeatedly modify the image in the direction that excites VGG features.

    Multiple smaller steps create more recursive structure than one large step
    because each new pattern is reinterpreted by VGG on the following step.
    """
    dreamed = image.detach().clone().to(self.device)

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

      # Normalize each update so step_size has predictable meaning even when
      # activation magnitude changes substantially as the dream evolves.
      gradient = gradient / gradient.std().clamp_min(GRADIENT_EPSILON)

      dreamed = (
        dreamed
        + gradient * self.step_size
      ).clamp(0.0, 1.0).detach()

    return dreamed