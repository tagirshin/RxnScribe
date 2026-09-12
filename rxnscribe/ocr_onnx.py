"""The deployed English EasyOCR greedy path, using CPU ONNX sessions."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .onnx import session
from ._vendor.easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from ._vendor.easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from ._vendor.easyocr.utils import group_text_box, get_image_list, diff
from ._vendor.easyocr.recognition import adjust_contrast_grey


class EasyOCRONNX:
    def __init__(self, model_path, threads=2):
        path = Path(model_path)
        self.config = json.loads((path / "manifest.json").read_text())["ocr"]
        self.detector = session(path / "ocr_detector.onnx", threads)
        self.recognizer = session(path / "ocr_recognizer.onnx", threads)

    def recognize(self, crop, width, contrast=False):
        if contrast:
            crop = adjust_contrast_grey(crop, target=0.5)
        image = Image.fromarray(crop, "L")
        resized_width = min(width, math.ceil(64 * image.width / image.height))
        values = np.asarray(
            image.resize((resized_width, 64), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        values = (values / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
        values = np.pad(values, ((0, 0), (0, width - resized_width)), mode="edge")
        (logits,) = self.recognizer.run(None, {"image": values[None, None]})
        probs = np.exp(logits - logits.max(axis=2, keepdims=True))
        probs /= probs.sum(axis=2, keepdims=True)
        probs[:, :, self.config["ignore_indices"]] = 0
        probs /= probs.sum(axis=2, keepdims=True)
        ids = probs[0].argmax(axis=1)
        keep = (ids != 0) & np.concatenate(([True], ids[1:] != ids[:-1]))
        text = "".join(self.config["characters"][i] for i in ids[keep])
        scores = probs[0].max(axis=1)[ids != 0]
        confidence = (
            float(scores.prod() ** (2 / np.sqrt(len(scores)))) if len(scores) else 0.0
        )
        return text, confidence

    def readtext(self, image, detail=1):
        if detail not in (0, 1):
            raise ValueError("ONNX OCR supports detail=0 or detail=1")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an H x W x 3 uint8 crop")
        # Preserve EasyOCR's ndarray BGR interpretation, including colored crops.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized, ratio, _ = resize_aspect_ratio(
            image, 2560, cv2.INTER_LINEAR, mag_ratio=1.0
        )
        values = normalizeMeanVariance(resized).transpose(2, 0, 1)[None]
        scores = self.detector.run(["scores"], {"image": values})[0][0]
        boxes, _, _ = getDetBoxes(
            scores[:, :, 0], scores[:, :, 1], 0.7, 0.4, 0.4, False
        )
        boxes = adjustResultCoordinates(boxes, 1 / ratio, 1 / ratio)
        polys = [np.asarray(box).astype(np.int32).reshape(-1) for box in boxes]
        horizontal, free = group_text_box(polys, 0.1, 0.5, 0.5, 0.5, 0.1, True)
        horizontal = [b for b in horizontal if max(b[1] - b[0], b[3] - b[2]) > 20]
        free = [
            b
            for b in free
            if max(diff([p[0] for p in b]), diff([p[1] for p in b])) > 20
        ]
        results = []
        for horizontal_box, free_box in [([b], []) for b in horizontal] + [
            ([], [b]) for b in free
        ]:
            crops, width = get_image_list(
                horizontal_box, free_box, gray, model_height=64
            )
            for box, crop in crops:
                text, confidence = self.recognize(crop, int(width))
                if confidence < 0.1:
                    retry = self.recognize(crop, int(width), contrast=True)
                    if retry[1] >= confidence:
                        text, confidence = retry
                results.append((box, text, confidence))
        return [r[1] for r in results] if detail == 0 else results
