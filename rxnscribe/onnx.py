"""CPU ONNX inference with the original reaction tokenizer and post-processing."""

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .data import postprocess_reactions
from .tokenizer import ReactionTokenizer


def session(path, threads=2):
    if threads < 1:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def reaction_image(image, size=1333):
    image = image.convert("RGB")
    # Match torchvision's float32 aspect scaling, rounding and PIL bilinear resize.
    dimensions = np.array(image.size[::-1], dtype=np.float32)
    scaled = np.maximum(
        np.round(dimensions * (np.float32(size) / dimensions.max())), 1
    ).astype(int)
    height, width = scaled
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (0, 0))
    values = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    values = (
        values - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return values[None], [width / size, height / size]


class RxnScribeONNX:
    def __init__(self, model_path, threads=2):
        self.path = Path(model_path)
        self.config = json.loads((self.path / "manifest.json").read_text())
        if self.config["format_version"] != 1:
            raise ValueError("Unsupported ONNX bundle version")
        self.encoder = session(self.path / "reaction_encoder.onnx", threads)
        self.decoder = session(self.path / "reaction_decoder.onnx", threads)
        with np.load(
            self.path / "reaction_embeddings.npz", allow_pickle=False
        ) as arrays:
            self.start = arrays["start"]
            self.embeddings = arrays["tokens"]
        self.threads = threads
        self.molscribe = None
        self.ocr = None
        self.tokenizer = ReactionTokenizer(input_size=2000, sep_xy=False, pix2seq=True)

    def predict_sequence(self, image):
        values, scale = reaction_image(image, self.config["reaction"]["image_size"])
        memory, mask, positions = self.encoder.run(None, {"image": values})
        embedding = self.start
        cache = np.zeros(self.config["reaction"]["cache_shape"], dtype=np.float32)
        states, previous, tokens, scores = [None], [None], [], []
        end = None
        for step in range(self.config["reaction"]["max_len"]):
            logits, cache = self.decoder.run(
                None,
                {
                    "embedding": embedding,
                    "memory": memory,
                    "mask": mask,
                    "positions": positions,
                    "cache": cache,
                },
            )
            log_probs = logits - logits.max(axis=-1, keepdims=True)
            log_probs -= np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
            if self.tokenizer.output_constraint:
                states, masks = self.tokenizer.update_states_and_masks(states, previous)
                log_probs[np.asarray(masks, dtype=bool)] = -10000
            token = int(log_probs[0].argmax())
            if token == self.tokenizer.EOS_ID and end is None:
                end = step
            if end is not None and step > 4:
                break
            tokens.append(token)
            scores.append(float(log_probs[0, token]))
            previous = [token]
            embedded_token = int(log_probs[0, : len(self.embeddings)].argmax())
            embedding = self.embeddings[embedded_token].reshape(1, 1, -1)
        # Preserve native EOS slicing, including empty output if EOS never arrives.
        return tokens[: end or 0], scores[: end or 0], scale

    def predict_image(self, image, molscribe=False, ocr=False):
        image = image.convert("RGB")
        if molscribe and self.molscribe is None:
            self.molscribe = MolScribeONNX(self.path, self.threads)
        if ocr and self.ocr is None:
            from .ocr_onnx import EasyOCRONNX

            self.ocr = EasyOCRONNX(self.path, self.threads)
        tokens, scores, scale = self.predict_sequence(image)
        reactions = self.tokenizer.sequence_to_data(tokens, scores, scale=scale)
        return postprocess_reactions(
            reactions,
            image=image,
            molscribe=self.molscribe if molscribe else None,
            ocr=self.ocr if ocr else None,
        )

    def predict_images(self, input_images, batch_size=16, **kwargs):
        # ponytail: one image per encoder call bounds RAM; batch after profiling real workloads.
        return [self.predict_image(image, **kwargs) for image in input_images]

    def predict_image_files(self, image_files, **kwargs):
        return [self.predict_image_file(path, **kwargs) for path in image_files]

    def predict_image_file(self, image_file, **kwargs):
        with Image.open(image_file) as image:
            return self.predict_image(image.convert("RGB"), **kwargs)


def molecule_image(image, size=384):
    image = np.asarray(image, dtype=np.uint8)
    rows, cols = np.where(np.any(image != 255, axis=2))
    if len(rows):
        image = image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    image = cv2.copyMakeBorder(
        image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    image = cv2.resize(image, (size, size))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    values = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    # Albumentations normalizes in pixel space, then multiplies by inverse std.
    values -= np.array([0.485, 0.456, 0.406], dtype=np.float32) * np.float32(255)
    values *= 1 / (np.array([0.229, 0.224, 0.225], dtype=np.float32) * np.float32(255))
    return values.transpose(2, 0, 1)[None]


class MolScribeONNX:
    def __init__(self, model_path, threads=2):
        from ._vendor.molscribe.tokenizer import CharTokenizer

        path = Path(model_path)
        self.config = json.loads((path / "manifest.json").read_text())["molecule"]
        self.encoder = session(path / "molecule_encoder.onnx", threads)
        self.decoder = session(path / "molecule_decoder.onnx", threads)
        self.edges = session(path / "molecule_edges.onnx", threads)
        self.tokenizer = CharTokenizer(
            input_size=self.config["coord_bins"],
            path=path / "molecule_vocab.json",
            sep_xy=self.config["sep_xy"],
        )
        self.decoder_inputs = {item.name for item in self.decoder.get_inputs()}

    def predict_graph(self, image):
        from ._vendor.molscribe.edges import get_edge_prediction

        memory, memory_cache = self.encoder.run(
            None, {"image": molecule_image(image, self.config["image_size"])}
        )
        cache = np.zeros(self.config["cache_shape"], dtype=np.float32)
        token, tokens, hiddens = 1, [], []
        for step in range(self.config["max_len"]):
            feed = {
                "token": np.array([token], dtype=np.int64),
                "memory": memory,
                "memory_cache": memory_cache,
                "cache": cache,
            }
            logits, hidden, cache = self.decoder.run(
                None, {k: v for k, v in feed.items() if k in self.decoder_inputs}
            )
            logits[0, self.tokenizer.get_output_mask(token)] = -10000
            if step == 0:
                logits[0, 2] = -1e20
            token = int(logits[0].argmax())
            tokens.append(token)
            hiddens.append(hidden)
            if token == 2:
                break
        graph = self.tokenizer.sequence_to_smiles(tokens)
        if graph["indices"]:
            (edges,) = self.edges.run(
                None,
                {
                    "hidden": np.concatenate(hiddens, axis=1),
                    "indices": np.array([graph["indices"]], dtype=np.int64),
                },
            )
            probs = np.exp(edges - edges.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            graph["edges"], _ = get_edge_prediction(
                probs[0].transpose(1, 2, 0).tolist()
            )
        else:
            graph["edges"] = []
        return graph, tokens

    def predict_images(self, input_images, batch_size=16):
        from ._vendor.molscribe.chemistry import convert_graph_to_smiles

        graphs = [self.predict_graph(image)[0] for image in input_images]
        if not graphs:
            return []
        smiles, molfiles, _ = convert_graph_to_smiles(
            [g["coords"] for g in graphs],
            [g["symbols"] for g in graphs],
            [g["edges"] for g in graphs],
            images=input_images,
            num_workers=1,
        )
        return [{"smiles": smi, "molfile": mol} for smi, mol in zip(smiles, molfiles)]
