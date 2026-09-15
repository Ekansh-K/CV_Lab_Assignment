#!/usr/bin/env bash
# HybridSORT env on Thunder instance. Do not downgrade the preinstalled CUDA torch.
set -euo pipefail
cd "$HOME/CV_Lab_Assignment/third_party/HybridSORT"

python3 -m pip install -U pip setuptools wheel

# Core deps from requirements.txt except torch/torchvision/onnx (already have torch 2.12+cu130)
python3 -m pip install \
  numpy opencv-python loguru scikit-image tqdm Pillow thop ninja tabulate \
  tensorboard filterpy h5py pandas xmltodict cython

# lap (assignment) — may need cmake/build
python3 -m pip install lap || python3 -m pip install lapx

python3 -m pip install cython_bbox
python3 -m pip install pycocotools || python3 -m pip install "pycocotools>=2.0.7"

# Install HybridSORT / YOLOX package in place
python3 setup.py develop

echo "==== SMOKE IMPORTS (no GPU inference) ===="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import yolox
print("yolox", getattr(yolox, "__file__", yolox))
from yolox.tracker.byte_tracker import BYTETracker
print("BYTETracker", BYTETracker)
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
print("Hybrid_Sort", Hybrid_Sort)
print("SMOKE_OK")
PY
