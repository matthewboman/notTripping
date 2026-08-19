# DreamCam

Real-time DeepDream webcam effects using PyTorch and OpenCV.

## Getting Started

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision opencv-python numpy
```

### 3. Run DreamCam

```bash
python dreamcam.py --mirror
```

A good starting point for stronger hallucinations is:

```bash
python dreamcam.py \
  --width 160 \
  --steps 3 \
  --step-size 0.018 \
  --feedback 0.42 \
  --mirror
```

## Main Options

- `--width` — neural-processing resolution. Lower values are faster.
- `--steps` — number of DeepDream iterations per video update. More steps create more developed hallucinations but cost more performance.
- `--step-size` — strength of each DeepDream update.
- `--feedback` — how much of the previous hallucinated frame is carried into the next frame.
- `--camera` — OpenCV camera index. Defaults to `0`.
- `--mirror` — horizontally mirrors the webcam image.

## Controls

- `q` or `Esc` — quit
- `Space` — pause/resume
- `r` — reset the feedback state

## Hardware

DreamCam automatically selects the best available PyTorch device:

- NVIDIA GPU: CUDA
- Apple Silicon: MPS
- Supported Intel GPU: XPU
- Otherwise: CPU

The first run may download the pretrained VGG16 model weights.