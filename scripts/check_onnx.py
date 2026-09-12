"""Real-checkpoint parity check, run in the export environment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rxnscribe.onnx import RxnScribeONNX, MolScribeONNX, reaction_image, molecule_image
from rxnscribe.export_onnx import load_molecule
from rxnscribe.interface import RxnScribe
from rxnscribe.dataset import make_transforms
from rxnscribe.tokenizer import get_tokenizer
from rxnscribe.data import ReactionImageData

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--reaction-checkpoint", required=True)
p.add_argument("--molecule-checkpoint", required=True)
p.add_argument("--bundle", required=True)
p.add_argument("--images", nargs="+", required=True)
p.add_argument("--report", default="parity.json")
a = p.parse_args()
torch.set_num_threads(2)
native = RxnScribe.__new__(RxnScribe)
args = native._get_args()
tokenizer = get_tokenizer(args)
native.model = native.get_model(
    args,
    tokenizer,
    torch.device("cpu"),
    torch.load(a.reaction_checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ],
)
transform = make_transforms("test", augment=False, debug=False)
molecule, _ = load_molecule(a.molecule_checkpoint)
onnx = RxnScribeONNX(a.bundle)
onnx_mol = MolScribeONNX(a.bundle)
report = []
for filename in a.images:
    image = Image.open(filename).convert("RGB")
    values, refs = transform(image)
    converted, scale = reaction_image(image)
    np.testing.assert_array_equal(converted[0], values.numpy())
    np.testing.assert_allclose(scale, refs["scale"], atol=1e-7)
    start = time.perf_counter()
    with torch.no_grad():
        seqs, scores = native.model(values[None], max_len=256)
    native_seconds = time.perf_counter() - start
    start = time.perf_counter()
    tokens, actual_scores, _ = onnx.predict_sequence(image)
    onnx_seconds = time.perf_counter() - start
    assert tokens == seqs[0].tolist(), (filename, tokens, seqs[0].tolist())
    np.testing.assert_allclose(actual_scores, scores[0].tolist(), atol=2e-4, rtol=2e-4)
    predictions = onnx.predict_image(image)
    data = ReactionImageData(predictions=predictions, image=image)
    crops = [
        box.image()
        for reaction in data.pred_reactions
        for box in reaction.bboxes
        if box.is_mol
    ]
    for crop in crops:
        tensor = molecule.transform(image=crop, keypoints=[])["image"]
        np.testing.assert_array_equal(molecule_image(crop)[0], tensor.numpy())
        with torch.no_grad():
            features, hiddens = molecule.encoder(tensor[None])
            graph = molecule.decoder.decode(features, hiddens)[0]
        converted, _ = onnx_mol.predict_graph(crop)
        for key in ["symbols", "coords", "indices", "smiles"]:
            assert converted[key] == graph["chartok_coords"][key], (
                filename,
                key,
                converted,
                graph,
            )
        assert converted["edges"] == graph["edges"], (
            filename,
            "bonds",
            converted,
            graph,
        )
    if crops:
        expected = molecule.predict_images(crops, batch_size=1)
        actual = onnx_mol.predict_images(crops)
        assert actual == expected, (
            filename,
            [
                (i, x["smiles"], y["smiles"], x["molfile"] == y["molfile"])
                for i, (x, y) in enumerate(zip(actual, expected))
                if x != y
            ],
        )
    item = dict(
        image=filename,
        tokens=len(tokens),
        molecules=len(crops),
        native_reaction_seconds=native_seconds,
        onnx_reaction_seconds=onnx_seconds,
    )
    report.append(item)
    print(json.dumps(item), flush=True)
Path(a.bundle).parent.joinpath(a.report).write_text(json.dumps(report, indent=2) + "\n")
