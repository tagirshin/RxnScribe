"""Compare all condition crops with floating-point and deployed quantized EasyOCR."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import easyocr

from rxnscribe.onnx import RxnScribeONNX
from rxnscribe.ocr_onnx import EasyOCRONNX
from rxnscribe.data import ReactionImageData

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--bundle", required=True)
p.add_argument("--checkpoints", required=True)
p.add_argument("--images", nargs="+", required=True)
a = p.parse_args()
torch.set_num_threads(2)
reader = easyocr.Reader(
    ["en"],
    gpu=False,
    quantize=False,
    model_storage_directory=a.checkpoints,
    user_network_directory=a.checkpoints + "/network",
    download_enabled=False,
    verbose=False,
)
quantized = easyocr.Reader(
    ["en"],
    gpu=False,
    quantize=True,
    model_storage_directory=a.checkpoints,
    user_network_directory=a.checkpoints + "/network",
    download_enabled=False,
    verbose=False,
)
model = RxnScribeONNX(a.bundle)
ocr = EasyOCRONNX(a.bundle)
report = []
for filename in a.images:
    image = Image.open(filename).convert("RGB")
    data = ReactionImageData(predictions=model.predict_image(image), image=image)
    crops = [
        box.image()
        for reaction in data.pred_reactions
        for box in reaction.bboxes
        if not box.is_mol
    ]
    for i, crop in enumerate(crops):
        expected = reader.readtext(crop, detail=0)
        actual = ocr.readtext(crop, detail=0)
        deployed = quantized.readtext(crop, detail=0)
        item = dict(
            image=filename,
            crop=i,
            onnx=actual,
            native_float=expected,
            native_quantized=deployed,
        )
        report.append(item)
        print(json.dumps(item), flush=True)
        assert actual == expected, item
Path(a.bundle).parent.joinpath("ocr-parity.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
