import torch
from torchvision.models import VGG16_Weights, vgg16

class DeepDream:
  def __init__(
      self,
      device,
      steps=2,
      step_size=0.02
  ):
    self.device = device
    self.steps = steps
    self.step_size = step_size

    weights = VGG16_Weights.DEFAULT

    model = vgg16(weights=weights).features
    model.eval()
    model.to(device)

    for parameter in model.parameters():
      parameter.requires_grad_(False)

    self.model = model
    self.layers = {
      15: 0.25, # shape fragments
      22: 0.45, # semantic-ish structure
      29: 0.30, # strongly semantic features
    }

    self.mean = torch.tensor(
      [0.485, 0.456, 0.406],
      device=device
    ).view(1, 3, 1, 1)

    self.std = torch.tensor(
      [0.229, 0.224, 0.225],
      device=device,
    ).view(1, 3, 1, 1)

  def normalize(self, image):
    return (image - self.mean) / self.std

  def get_features(self, image):
    features = {}
    x = self.normalize(image)

    for index, layer in enumerate(self.model):
        x = layer(x)

        if index in self.layers:
            features[index] = x

        if index >= max(self.layers):
            break

    return features


  def dream(self, image, guide):
    guide = guide.detach().to(self.device)

    with torch.no_grad():
        guide_features = self.get_features(guide)

        guide_directions = {}
        guide_strengths = {}

        for layer, features in guide_features.items():
            guide_directions[layer] = torch.nn.functional.normalize(
                features,
                dim=1,
            )

            strength = features.square().mean(
                dim=1,
                keepdim=True,
            )

            strength = strength / (
                strength.mean() + 1e-8
            )

            guide_strengths[layer] = strength.clamp(
                0.0,
                3.0,
            )

    dreamed = image.detach().clone().to(self.device)

    for _ in range(self.steps):
        dreamed.requires_grad_(True)

        dreamed_features = self.get_features(dreamed)

        dream_loss = 0.0

        for layer, weight in self.layers.items():
            projection = (
                dreamed_features[layer]
                * guide_directions[layer]
            ).sum(
                dim=1,
                keepdim=True,
            )

            layer_loss = (
                projection
                * guide_strengths[layer]
            ).mean()

            dream_loss = dream_loss + layer_loss * weight

        anchor_loss = (
            dreamed - guide
        ).square().mean()

        horizontal = (
            dreamed[:, :, :, 1:]
            - dreamed[:, :, :, :-1]
        ).abs().mean()

        vertical = (
            dreamed[:, :, 1:, :]
            - dreamed[:, :, :-1, :]
        ).abs().mean()

        texture_loss = horizontal + vertical

        # loss = (
        #     dream_loss
        #     - anchor_loss * 0.20
        #     - texture_loss * 0.002 # .1
        # )
        loss = dream_loss

        gradient = torch.autograd.grad(
            loss,
            dreamed,
            retain_graph=False,
            create_graph=False,
        )[0]

        gradient_std = gradient.std()

        if gradient_std > 0:
            gradient = gradient / (
                gradient_std + 1e-8
            )

        dreamed = (
            dreamed
            + gradient * self.step_size
        ).clamp(0.0, 1.0).detach()

    return dreamed