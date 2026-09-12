"""Export existing checkpoints; PyTorch is needed here, never at ONNX inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class ReactionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        from .pix2seq.misc import nested_tensor_from_tensor_list

        features, positions = self.model.backbone(nested_tensor_from_tensor_list(image))
        src = self.model.input_proj(features[-1].tensors).flatten(2).permute(2, 0, 1)
        pos = positions[-1].flatten(2).permute(2, 0, 1)
        mask = torch.zeros_like(features[-1].mask).flatten(1)
        memory = self.model.transformer.encoder(src, src_key_padding_mask=mask, pos=pos)
        return memory, mask, pos


class ReactionDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, embedding, memory, mask, positions, cache):
        hidden, updated = self.transformer.decoder(
            embedding,
            memory,
            memory_key_padding_mask=mask,
            pos=positions,
            pre_kv_list=list(cache.unbind(0)),
        )
        logits = self.transformer.vocal_classifier(hidden.transpose(0, 1))
        return logits[:, 0], torch.stack(updated)


class MoleculeEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder.decoder["chartok_coords"]

    def forward(self, image):
        features, _ = self.encoder(image)
        memory = self.decoder.enc_transform(features)
        caches = []
        for layer in self.decoder.decoder.transformer_layers:
            attn = layer.context_attn
            key = (
                attn.linear_keys(memory)
                .reshape(1, -1, attn.head_count, attn.dim_per_head)
                .transpose(1, 2)
            )
            value = (
                attn.linear_values(memory)
                .reshape(1, -1, attn.head_count, attn.dim_per_head)
                .transpose(1, 2)
            )
            caches.append(torch.stack([key, value]))
        return memory, torch.stack(caches)


class MoleculeDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.model = decoder

    def forward(self, token, memory, memory_cache, cache):
        decoder = self.model.decoder
        decoder.state["cache"] = {
            f"layer_{i}": {
                "memory_keys": memory_cache[i, 0],
                "memory_values": memory_cache[i, 1],
                "self_keys": cache[i, 0],
                "self_values": cache[i, 1],
            }
            for i in range(len(decoder.transformer_layers))
        }
        # Preserve native inference: each single-token embedding uses position zero.
        # step=1 only prevents resetting the supplied attention cache.
        embedding, padding = self.model.dec_embedding(token.reshape(1, 1, 1))
        hidden, *_ = decoder(embedding, memory, tgt_pad_mask=padding, step=1)
        updated = torch.stack(
            [
                torch.stack([item["self_keys"], item["self_values"]])
                for item in decoder.state["cache"].values()
            ]
        )
        return self.model.output_layer(hidden)[:, 0], hidden, updated


class MoleculeEdges(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden, indices):
        return self.model(hidden, indices)["edges"]


def export(module, inputs, destination, input_names, output_names, dynamic_axes=None):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(destination),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            do_constant_folding=True,
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(destination), options, providers=["CPUExecutionProvider"]
    )
    with torch.no_grad():
        expected = module(*inputs)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    feed = {
        name: tensor.detach().numpy()
        for name, tensor in zip(input_names, inputs)
        if name in {item.name for item in session.get_inputs()}
    }
    actual = session.run(None, feed)
    for baseline, converted in zip(expected, actual):
        np.testing.assert_allclose(
            converted, baseline.detach().numpy(), atol=2e-4, rtol=2e-4
        )
    print(f"Exported and checked {destination.name}", flush=True)


def export_reaction(checkpoint, output):
    from .interface import RxnScribe
    from .tokenizer import get_tokenizer

    # Construct only the diagram model: no eager MolScribe/OCR downloads.
    interface = RxnScribe.__new__(RxnScribe)
    args = interface._get_args()
    tokenizer = get_tokenizer(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = interface.get_model(
        args, tokenizer, torch.device("cpu"), state["state_dict"]
    )
    image = torch.zeros(1, 3, 1333, 1333)
    encoder = ReactionEncoder(model).eval()
    export(
        encoder,
        (image,),
        output / "reaction_encoder.onnx",
        ["image"],
        ["memory", "mask", "positions"],
    )
    with torch.no_grad():
        memory, mask, positions = encoder(image)
    tr = model.transformer
    embedding = tr.det_embed.weight.detach().reshape(1, 1, -1)
    cache_shape = (tr.num_decoder_layers, 2, 1, tr.nhead, 0, tr.d_model // tr.nhead)
    cache = torch.zeros(*cache_shape[:4], 2, cache_shape[5])
    export(
        ReactionDecoder(tr),
        (embedding, memory, mask, positions, cache),
        output / "reaction_decoder.onnx",
        ["embedding", "memory", "mask", "positions", "cache"],
        ["logits", "next_cache"],
        {"cache": {4: "past"}, "next_cache": {4: "present"}},
    )
    np.savez(
        output / "reaction_embeddings.npz",
        start=embedding.numpy(),
        tokens=tr.vocal_embed.weight.detach().numpy(),
    )
    return {
        "image_size": 1333,
        "cache_shape": cache_shape,
        "max_len": tokenizer["reaction"].max_len,
    }


def load_molecule(checkpoint):
    from molscribe import MolScribe
    from molscribe.tokenizer import get_tokenizer
    from molscribe.dataset import get_transforms

    model = MolScribe.__new__(MolScribe)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = model._get_args(state["args"])
    args.use_checkpoint = False
    model.device = torch.device("cpu")
    model.tokenizer = get_tokenizer(args)
    model.encoder, model.decoder = model._get_model(
        args, model.tokenizer, model.device, state
    )
    model.transform = get_transforms(args.input_size, augment=False)
    model.num_workers = 1
    return model, args


def export_molecule(checkpoint, output):
    model, args = load_molecule(checkpoint)
    if (
        args.formats != ["chartok_coords", "edges"]
        or args.continuous_coords
        or args.enc_pos_emb
        or args.max_relative_positions
    ):
        raise ValueError(
            "This exporter supports the deployed chartok_coords/edges checkpoint"
        )
    encoder = MoleculeEncoder(model).eval()
    image = torch.zeros(1, 3, args.input_size, args.input_size)
    export(
        encoder,
        (image,),
        output / "molecule_encoder.onnx",
        ["image"],
        ["memory", "memory_cache"],
    )
    with torch.no_grad():
        memory, memory_cache = encoder(image)
    shape = (
        args.dec_num_layers,
        2,
        1,
        args.dec_attn_heads,
        0,
        args.dec_hidden_size // args.dec_attn_heads,
    )
    cache = torch.zeros(*shape[:4], 2, shape[5])
    export(
        MoleculeDecoder(model.decoder.decoder["chartok_coords"]),
        (torch.tensor([1]), memory, memory_cache, cache),
        output / "molecule_decoder.onnx",
        ["token", "memory", "memory_cache", "cache"],
        ["logits", "hidden", "next_cache"],
        {"cache": {4: "past"}, "next_cache": {4: "present"}},
    )
    export(
        MoleculeEdges(model.decoder.decoder["edges"]),
        (torch.zeros(1, 5, args.dec_hidden_size), torch.tensor([[1, 3]])),
        output / "molecule_edges.onnx",
        ["hidden", "indices"],
        ["edges"],
        {
            "hidden": {1: "tokens"},
            "indices": {1: "atoms"},
            "edges": {2: "atoms", 3: "atoms"},
        },
    )
    vocabulary = model.tokenizer["chartok_coords"].stoi
    (output / "molecule_vocab.json").write_text(json.dumps(vocabulary))
    return {
        "image_size": args.input_size,
        "cache_shape": shape,
        "coord_bins": args.coord_bins,
        "sep_xy": args.sep_xy,
        "max_len": 480,
    }


class OCRRecognizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        # Native adaptive pool leaves sequence width unchanged and averages height.
        features = self.model.FeatureExtraction(image).permute(0, 3, 1, 2).mean(3)
        return self.model.Prediction(self.model.SequenceModeling(features).contiguous())


def export_ocr(checkpoints, output):
    import easyocr

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        quantize=False,
        model_storage_directory=str(checkpoints),
        user_network_directory=str(checkpoints / "network"),
        verbose=False,
    )
    export(
        reader.detector,
        (torch.zeros(1, 3, 128, 256),),
        output / "ocr_detector.onnx",
        ["image"],
        ["scores", "features"],
        {
            "image": {2: "height", 3: "width"},
            "scores": {1: "half_height", 2: "half_width"},
            "features": {2: "half_height", 3: "half_width"},
        },
    )
    export(
        OCRRecognizer(reader.recognizer),
        (torch.zeros(1, 1, 64, 128),),
        output / "ocr_recognizer.onnx",
        ["image"],
        ["logits"],
        {"image": {3: "width"}, "logits": {1: "sequence"}},
    )
    return {
        "language": "en",
        "characters": reader.converter.character,
        "ignore_indices": [
            i + 1
            for i, char in enumerate(reader.character)
            if char not in reader.lang_char
        ],
        "quantized": False,
        "checkpoint_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in checkpoints.glob("*.pth")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-checkpoint", type=Path)
    parser.add_argument("--molecule-checkpoint", type=Path)
    parser.add_argument("--ocr-checkpoints", type=Path)
    parser.add_argument(
        "--component", choices=["reaction", "molecule", "ocr", "all"], default="all"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for component, value in [
        ("reaction", args.reaction_checkpoint),
        ("molecule", args.molecule_checkpoint),
        ("ocr", args.ocr_checkpoints),
    ]:
        if args.component in {component, "all"} and value is None:
            parser.error(f"{component} export requires its checkpoint argument")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    config = (
        json.loads(manifest.read_text()) if manifest.exists() else {"format_version": 1}
    )
    if args.component in {"reaction", "all"}:
        config["reaction"] = export_reaction(args.reaction_checkpoint, args.output)
        config["reaction_checkpoint_sha256"] = hashlib.sha256(
            args.reaction_checkpoint.read_bytes()
        ).hexdigest()
    if args.component in {"molecule", "all"} and args.molecule_checkpoint:
        config["molecule"] = export_molecule(args.molecule_checkpoint, args.output)
        config["molecule_checkpoint_sha256"] = hashlib.sha256(
            args.molecule_checkpoint.read_bytes()
        ).hexdigest()
    if args.component in {"ocr", "all"} and args.ocr_checkpoints:
        config["ocr"] = export_ocr(args.ocr_checkpoints, args.output)
    config["files"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in args.output.iterdir()
        if path.suffix in {".onnx", ".npz"} or path.name == "molecule_vocab.json"
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(manifest)


if __name__ == "__main__":
    main()
