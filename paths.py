"""Machine-specific locations, set by environment variables (defaults are relative to the repository)."""
import os

DATA = os.environ.get("SAMFAIL_DATA", "data")                # datasets: DAVIS2017/, MOSE/, SA-V/, LVOS2/
CKPT = os.environ.get("SAMFAIL_CKPT", "checkpoints")         # sam2.1_hiera_base_plus.pt, sam3.pt
EXT = os.environ.get("SAMFAIL_EXT", "third_party")           # EdgeTAM/, sam3/, models/Qwen2.5-VL-3B-Instruct
CACHE = os.environ.get("SAMFAIL_CACHE", "cache")             # rendered overlays of the VLM gate
