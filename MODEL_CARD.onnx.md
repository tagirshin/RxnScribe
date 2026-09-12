---
language: en
license: other
license_name: upstream-component-licenses
license_link: https://huggingface.co/tagirshin/RxnScribe-ONNX/blob/main/LICENSES/README.md
base_model:
- yujieq/RxnScribe
- yujieq/MolScribe
tags:
- onnx
- chemistry
- optical-chemical-structure-recognition
- image-to-text
---

# RxnScribe ONNX CPU bundle

ONNX conversions of the existing RxnScribe, MolScribe and English EasyOCR
checkpoints. This is a community conversion, not a newly trained model or an
upstream release. Use the matching
[RxnScribe feature branch](https://github.com/tagirshin/RxnScribe/tree/feat/onnx-inference)
and its `RxnScribeONNX` interface. PyTorch is needed for export, not inference.

## Download

Download the complete repository, including `manifest.json`, all seven `.onnx`
graphs, `reaction_embeddings.npz` and `molecule_vocab.json`:

```sh
hf download tagirshin/RxnScribe-ONNX --revision onnx-poc-v1 --local-dir local/onnx-models
```

Then use the matching source checkout with `requirements-onnx.txt` installed:

```python
from rxnscribe import RxnScribeONNX
model = RxnScribeONNX("local/onnx-models", threads=2)
reactions = model.predict_image_file("scheme.png", molscribe=True, ocr=True)
```

The runtime reads local files and makes no Hub requests. Pin the Hub commit SHA
from the source branch's `README.onnx.md` for reproducible deployments.

## Components and provenance

| Component | Source | License notice |
|---|---|---|
| Reaction encoder, decoder, embeddings | [yujieq/RxnScribe](https://huggingface.co/yujieq/RxnScribe), `pix2seq_reaction_full.ckpt`, revision `034bcfeaa6780624b2897f7955853271de1d1f65` | [MIT](LICENSES/RxnScribe-MIT.txt) |
| Molecule encoder, decoder, bond predictor and vocabulary | [yujieq/MolScribe](https://huggingface.co/yujieq/MolScribe), `swin_base_char_aux_1m.pth`, revision `a0189776b7415b82795c7ee81eed311bf5c8724b` | [MIT](LICENSES/MolScribe-MIT.txt) |
| Text detector | [EasyOCR](https://github.com/JaidedAI/EasyOCR) 1.7.2, `craft_mlt_25k.pth`, based on [CRAFT](https://github.com/clovaai/CRAFT-pytorch) | [Apache-2.0](LICENSES/EasyOCR-Apache-2.0.txt), [CRAFT MIT](LICENSES/CRAFT-MIT.txt) |
| English text recognizer | EasyOCR 1.7.2, `english_g2.pth` | [Apache-2.0](LICENSES/EasyOCR-Apache-2.0.txt) |

Original checkpoint and exported file SHA-256 hashes are recorded in
`manifest.json`. Conversion uses ONNX opset 17. It was validated with Python
3.11, Torch 2.6.0 CPU for export, and ONNX Runtime 1.20.1 for inference on
Linux ARM64 via Docker Desktop. Model artifacts occupy approximately 659 MB.

## Validation and limitations

- Exact reaction tokens on five bundled schemes, the app Ugi example and a blank
  portrait; token log scores within `2e-4` absolute/relative tolerance.
- Exact native single-crop molecular graphs, SMILES and molfiles on 46 crops.
- Exact native floating-point OCR text on 32 condition crops.
- A Torch-free, network-disabled container completed full reaction, molecule and
  OCR inference; the companion app's Dagster extraction job also passed.
- CPU, one image/crop at a time. Only English greedy OCR is supported. GPU,
  x86 execution and production multi-user workloads were not validated.
- Batch-32 Torch inference on one 17-crop scheme gave identical SMILES but four
  different molfiles compared with single-crop inference.
- OCR uses floating-point weights. The quantized Torch CPU path produced
  corrupted text in the tested ARM environment; its text matched only 8 of 32
  crops. The companion native fork also disables this quantization.
- These checks establish conversion parity, not chemical correctness. Review
  structures and conditions before use. Atom mapping and rule extraction are
  separate stages and are not supplied by this model bundle.

The original training data, methods and intended model behavior are documented
by the linked upstream projects. No fine-tuning was performed for this conversion.
