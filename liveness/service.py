# liveness/service.py
"""
Wrapper around the original Silent-Face-Anti-Spoofing inference code.

Key improvements over the bare repo:
  • Models are loaded once at startup and cached in a dict — no disk I/O
    on every request (original _load_model() reloaded on every predict()).
  • Thread-safe singleton via a threading.Lock.
  • Accepts numpy BGR arrays (OpenCV) or file-like objects / bytes.
  • Returns a structured dict instead of a bare numpy array.
  • Soft-max is applied with explicit dim=1 (avoids deprecation warning).
  • Detection and cropping stay on the same object, no duplication.
"""

import io
import os
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings

# ── Original repo imports (unchanged) ────────────────────────────────────────
from src.model_lib.MiniFASNet import (
    MiniFASNetV1, MiniFASNetV2,
    MiniFASNetV1SE, MiniFASNetV2SE,
)
from src.data_io import transform as trans
from src.generate_patches import CropImage
from src.utility import get_kernel, parse_model_name

logger = logging.getLogger('liveness')

MODEL_MAPPING = {
    'MiniFASNetV1':   MiniFASNetV1,
    'MiniFASNetV2':   MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}

# Label index → human-readable class (matches the training setup)
LABEL_NAMES = {0: 'fake', 1: 'real', 2: 'fake'}


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LivenessResult:
    is_real: bool
    score: float          # confidence that this is a *real* face (0-1)
    label: int            # raw argmax label
    label_name: str       # 'real' | 'fake'
    raw_scores: list      # softmax scores for all classes
    face_bbox: Optional[list] = None   # [x, y, w, h] or None if detection failed
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'is_real': self.is_real,
            'score': round(self.score, 4),
            'label': self.label,
            'label_name': self.label_name,
            'raw_scores': [round(float(s), 4) for s in self.raw_scores],
            'face_bbox': self.face_bbox,
            'error': self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
class _RetinaFaceDetector:
    """Thin wrapper around the Caffe RetinaFace model."""

    def __init__(self, caffemodel: str, prototxt: str, confidence: float = 0.6):
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.confidence_threshold = confidence

    def get_bbox(self, img: np.ndarray) -> Optional[list]:
        """Return [x, y, w, h] of the most-confident face, or None."""
        import math
        height, width = img.shape[:2]
        aspect_ratio = width / height

        blob_img = img
        if width * height >= 192 * 192:
            blob_img = cv2.resize(
                img,
                (int(192 * math.sqrt(aspect_ratio)),
                 int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(blob_img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()

        if out.ndim == 1:
            out = out[np.newaxis, :]

        # Filter by confidence
        valid = out[out[:, 2] >= self.confidence_threshold]
        if len(valid) == 0:
            return None

        best = valid[np.argmax(valid[:, 2])]
        left   = best[3] * width
        top    = best[4] * height
        right  = best[5] * width
        bottom = best[6] * height
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]


# ─────────────────────────────────────────────────────────────────────────────
class LivenessService:
    """
    Singleton service for passive liveness detection.

    Usage:
        service = LivenessService.get_instance()
        result  = service.check(image_bytes_or_np_array)
    """

    _instance: Optional['LivenessService'] = None
    _lock = threading.Lock()

    # ── Construction ─────────────────────────────────────────────────────────
    def __init__(self):
        cfg = settings.LIVENESS_CONFIG
        device_id = cfg['DEVICE_ID']
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model_dir: str = cfg['MODEL_DIR']
        self.real_threshold: float = cfg['REAL_SCORE_THRESHOLD']

        self.detector = _RetinaFaceDetector(
            caffemodel=cfg['DETECTOR_CAFFEMODEL'],
            prototxt=cfg['DETECTOR_PROTOTXT'],
        )
        self.image_cropper = CropImage()

        # model_path → (model, kernel_size)
        self.loaded_models: Dict[str, Tuple[torch.nn.Module, tuple]] = {}
        self._preload_models()

    # ── Singleton ────────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> 'LivenessService':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Model loading ─────────────────────────────────────────────────────────
    def _preload_models(self):
        if not os.path.isdir(self.model_dir):
            logger.warning("Model dir not found: %s", self.model_dir)
            return

        for fname in os.listdir(self.model_dir):
            if fname.endswith('.pth'):
                path = os.path.join(self.model_dir, fname)
                try:
                    self._load_model(path)
                    logger.info("Loaded model: %s → %s", fname, self.device)
                except Exception as exc:
                    logger.error("Could not load model %s: %s", fname, exc)

    def _load_model(self, model_path: str):
        """Load a .pth file and cache it.  Skips if already cached."""
        if model_path in self.loaded_models:
            return

        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)

        if model_type not in MODEL_MAPPING:
            raise ValueError(f"Unknown model type: {model_type}")

        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        first_key = next(iter(state_dict))
        if 'module.' in first_key:
            state_dict = OrderedDict(
                (k[7:], v) for k, v in state_dict.items()
            )
        model.load_state_dict(state_dict)
        model.eval()

        self.loaded_models[model_path] = (model, kernel_size)

    # ── Inference ─────────────────────────────────────────────────────────────
    def _infer_one_model(
        self,
        model: torch.nn.Module,
        img_patch: np.ndarray,
    ) -> np.ndarray:
        """Run one model on a cropped patch; return softmax probability array."""
        transform = trans.Compose([trans.ToTensor()])
        tensor = transform(img_patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs   # shape (1, num_classes)

    def _decode_image(self, image_input) -> Optional[np.ndarray]:
        """
        Accept:
          • np.ndarray  (already decoded BGR)
          • bytes / bytearray
          • file-like object (Django InMemoryUploadedFile, etc.)
        Returns BGR numpy array or None on failure.
        """
        if isinstance(image_input, np.ndarray):
            return image_input

        if hasattr(image_input, 'read'):
            image_input = image_input.read()

        if isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        return None

    # ── Public API ────────────────────────────────────────────────────────────
    def check(self, image_input) -> LivenessResult:
        """
        Run passive liveness detection on an image.

        Args:
            image_input: BGR np.ndarray, bytes, or file-like object.

        Returns:
            LivenessResult dataclass.
        """
        if not self.loaded_models:
            return LivenessResult(
                is_real=False, score=0.0, label=-1,
                label_name='unknown', raw_scores=[],
                error="No models loaded. Check MODEL_DIR configuration.",
            )

        # 1. Decode
        img = self._decode_image(image_input)
        if img is None:
            return LivenessResult(
                is_real=False, score=0.0, label=-1,
                label_name='unknown', raw_scores=[],
                error="Could not decode image.",
            )

        # 2. Face detection
        bbox = self.detector.get_bbox(img)
        if bbox is None:
            return LivenessResult(
                is_real=False, score=0.0, label=-1,
                label_name='unknown', raw_scores=[],
                face_bbox=None,
                error="No face detected in image.",
            )

        # 3. Run all models, accumulate softmax scores (model fusion)
        prediction = None
        src_h, src_w = img.shape[:2]

        for model_path, (model, _) in self.loaded_models.items():
            model_name = os.path.basename(model_path)
            h_input, w_input, _, scale = parse_model_name(model_name)

            crop_param = {
                "org_img": img,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": scale is not None,
            }
            patch = self.image_cropper.crop(**crop_param)
            probs = self._infer_one_model(model, patch)

            if prediction is None:
                prediction = probs
            else:
                # Average-ensemble over models (more stable than sum)
                prediction = prediction + probs

        prediction /= len(self.loaded_models)   # normalise to probabilities

        # 4. Interpret result
        # Class 1 = real face (matches original training setup)
        label = int(np.argmax(prediction[0]))
        real_score = float(prediction[0][1]) if prediction.shape[1] > 1 else 0.0
        is_real = (label == 1) and (real_score >= self.real_threshold)

        return LivenessResult(
            is_real=is_real,
            score=real_score,
            label=label,
            label_name=LABEL_NAMES.get(label, 'unknown'),
            raw_scores=prediction[0].tolist(),
            face_bbox=bbox,
        )
