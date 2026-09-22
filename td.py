import sys

import numpy as np
import torch
import torch.nn.functional as F

# Put deepdream.py beside your .toe file.
if project.folder not in sys.path:
    sys.path.insert(0, project.folder)

from deepdream import DeepDream


CONTROL_CHOP = 'controls'

DEFAULT_STEPS = 1
DEFAULT_STEP_SIZE = 0.018
DEFAULT_FEEDBACK = 0.42
DEFAULT_PROCESS_WIDTH = 256
DEFAULT_UPDATE_EVERY = 1
DEFAULT_LAYER_1 = 0.10
DEFAULT_LAYER_2 = 0.45
DEFAULT_LAYER_3 = 0.45

_dreamer = None
_device = None
_previous_dream = None
_last_output = None
_cook_count = 0


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')

    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


def _control(name, default):
    controls = op(CONTROL_CHOP)

    if controls is None:
        return default

    channel = controls[name]

    if channel is None:
        return default

    return channel.eval()


def _get_dreamer():
    global _dreamer
    global _device

    if _dreamer is None:
        _device = _get_device()

        _dreamer = DeepDream(
            device=_device,
            steps=DEFAULT_STEPS,
            step_size=DEFAULT_STEP_SIZE,
            octaves=1,
            max_whole_frame_width=0, # set to 0 for performance, 768 for quality
            model_name='vgg16',
        )

        print('DeepDream device:', _device)

    return _dreamer


def _reset():
    global _previous_dream
    global _last_output

    _previous_dream = None
    _last_output = None


def onSetupParameters(scriptOp):
    return


def onPulse(par):
    if par.name == 'Reset':
        _reset()

    return


def onCook(scriptOp):
    global _previous_dream
    global _last_output
    global _cook_count

    if not scriptOp.inputs:
        return

    source = scriptOp.inputs[0].numpyArray(delayed=True)

    # delayed=True avoids forcing a synchronous GPU readback.
    # The first cook can therefore have no image yet.
    if source is None:
        if _last_output is not None:
            scriptOp.copyNumpyArray(_last_output)
        return

    if source.ndim != 3 or source.shape[2] < 3:
        raise RuntimeError('DeepDream Script TOP requires an RGB or RGBA input TOP')

    source = np.asarray(source, dtype=np.float32)

    height, width = source.shape[:2]

    steps = max(1, int(round(_control('steps', DEFAULT_STEPS))))
    step_size = max(0.0, float(_control('stepsize', DEFAULT_STEP_SIZE)))
    feedback = min(0.99, max(0.0, float(_control('feedback', DEFAULT_FEEDBACK))))
    process_width = max(32, int(round(_control('processWidth', DEFAULT_PROCESS_WIDTH))))
    update_every = max(1, int(round(_control('updateEvery', DEFAULT_UPDATE_EVERY))))
    layer_1 = float(_control('layer1', DEFAULT_LAYER_1))
    layer_2 = float(_control('layer2', DEFAULT_LAYER_2))
    layer_3 = float(_control('layer3', DEFAULT_LAYER_3))

    _cook_count += 1

    # Hold the most recently completed dream between neural updates.
    if (
        _last_output is not None
        and update_every > 1
        and (_cook_count - 1) % update_every != 0
    ):
        scriptOp.copyNumpyArray(_last_output)
        return

    dreamer = _get_dreamer()
    dreamer.steps = steps
    dreamer.step_size = step_size
    dreamer.dream_layer_weights = {
    	15: layer_1,
    	22: layer_2,
    	29: layer_3,
    }

    rgb = source[:, :, :3]
    black_mask = np.max(rgb, axis=2, keepdims=True) <= 0.08

    if source.shape[2] >= 4:
    	alpha_mask = source[:, :, :3] <= 0.08
    else:
    	alpha_mask = np.zeros((height, width), dtype=bool)

    protected_mask = black_mask | alpha_mask

    current = torch.from_numpy(rgb).to(
        device=_device,
        dtype=torch.float32,
    )

    current = current.permute(2, 0, 1).unsqueeze(0)

    process_height = max(
        32,
        round(height * process_width / width),
    )

    current = F.interpolate(
        current,
        size=(process_height, process_width),
        mode='bilinear',
        align_corners=False,
    )

    if (
        _previous_dream is None
        or _previous_dream.shape[-2:] != current.shape[-2:]
    ):
        _previous_dream = None

    if _previous_dream is not None and feedback > 0:
        dream_input = (
            current * (1.0 - feedback)
            + _previous_dream * feedback
        ).clamp(0.0, 1.0)
    else:
        dream_input = current

    dreamed = dreamer.dream(dream_input)

    # Keep only the small processing-resolution tensor as feedback state.
    _previous_dream = dreamed.detach()

    output = F.interpolate(
        dreamed,
        size=(height, width),
        mode='bicubic',
        align_corners=False,
    ).clamp(0.0, 1.0)

    output = (
        output[0]
        .permute(1, 2, 0)
        .detach()
        .to('cpu')
        .numpy()
        .astype(np.float32)
    )

    # Don't halucinate on black or alpha
    output[protected_mask] = 0.0

    # Preserve source alpha if the input TOP has one.
    if source.shape[2] >= 4:
        output = np.concatenate(
            [output, source[:, :, 3:4]],
            axis=2,
        )

    _last_output = np.ascontiguousarray(output)
    scriptOp.copyNumpyArray(_last_output)

    return
