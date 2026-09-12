"""Measure output changes from the old batched MolScribe path."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from rxnscribe.onnx import RxnScribeONNX, MolScribeONNX
from rxnscribe.export_onnx import load_molecule
from rxnscribe.data import ReactionImageData

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--bundle", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--image", required=True)
a = p.parse_args()
torch.set_num_threads(2)
image = Image.open(a.image).convert("RGB")
reaction = RxnScribeONNX(a.bundle)
data = ReactionImageData(predictions=reaction.predict_image(image), image=image)
crops = [b.image() for r in data.pred_reactions for b in r.bboxes if b.is_mol]
native, _ = load_molecule(a.checkpoint)
expected = native.predict_images(crops, batch_size=32)
actual = MolScribeONNX(a.bundle).predict_images(crops)
report = {
    "image": a.image,
    "crops": len(crops),
    "differences": [
        {
            "crop": i,
            "batched_smiles": x["smiles"],
            "single_smiles": y["smiles"],
            "molfile_equal": x["molfile"] == y["molfile"],
        }
        for i, (x, y) in enumerate(zip(expected, actual))
        if x != y
    ],
}
Path(a.bundle).parent.joinpath("batching-differences.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
