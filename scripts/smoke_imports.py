import torch
import cv2

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("cv2", cv2.__version__)

from trackers.byte_tracker.byte_tracker import BYTETracker
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort

print("BYTETracker", BYTETracker)
print("Hybrid_Sort", Hybrid_Sort)
print("SMOKE_OK")
