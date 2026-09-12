# ONNX inference PoC — feature branch

This branch runs the reaction diagram model, MolScribe, and English EasyOCR on
CPU with ONNX Runtime. Model bundles are mounted separately. No PyTorch,
TorchVision, Lightning, OpenNMT, timm, Hugging Face client, or model downloads
are needed at inference. The existing PyTorch interface is retained.

## Build and run

Clone the public feature branch and download the pre-exported model bundle:

```sh
git clone --branch feat/onnx-inference https://github.com/tagirshin/RxnScribe.git RxnScribe-onnx
cd RxnScribe-onnx
hf download tagirshin/RxnScribe-ONNX --revision 4f552d917cd47af38de89a3a4835a429e568e900 --local-dir local/onnx-models
```

`hf` comes from the `huggingface_hub` package and is needed only for model
transfer, outside the inference image. The bundle contains all three components;
keep `manifest.json`, the graphs, embeddings and vocabulary together. See
[the model card](https://huggingface.co/tagirshin/RxnScribe-ONNX) for provenance.

Then build and run locally:

```sh
docker build -f Dockerfile.onnx -t rxnscribe-onnx:local .
docker run --rm --network none \
  -v "$PWD/local/onnx-models:/models:ro" \
  -v "$PWD/assets:/images:ro" \
  rxnscribe-onnx:local python -c \
  'from rxnscribe import RxnScribeONNX; model = RxnScribeONNX("/models"); print(model.predict_image_file("/images/acs.joc.5b01703-Scheme-c1.png", molscribe=True, ocr=True))'
```

The image uses source directly; `pip install .` still installs the legacy
PyTorch package. For a separate Python 3.11 environment, install
`requirements-onnx.txt` and put this checkout on `PYTHONPATH`.

## Publishing a new model version

Upload the complete exported bundle to Hugging Face, as with the original Torch
checkpoints. Include the model card and upstream license notices:

```sh
hf upload tagirshin/RxnScribe-ONNX local/onnx-models .
```

Use a write-enabled Hugging Face login for upload. Public downloads do not need
a token. After validating the uploaded files, update the pinned model commit SHA
in the consuming release. Keep old release pins unchanged so existing users can
reproduce or roll back their deployment. The `onnx-poc-v1` model tag names this
first validated bundle; the GitHub code stays on `feat/onnx-inference`.

## Export once, outside the inference environment

Use the existing RxnScribe/MolScribe export environment (tested with Python 3.11,
PyTorch 2.6.0 CPU, MolScribe 1.1.1 at commit
`7296a30413eb55436702011efdff78131f66d162`, EasyOCR 1.7.2), adding
`onnx==1.17.0` and `onnxruntime==1.20.1`:

```sh
python -m rxnscribe.export_onnx \
  --reaction-checkpoint /checkpoints/pix2seq_reaction_full.ckpt \
  --molecule-checkpoint /checkpoints/swin_base_char_aux_1m.pth \
  --ocr-checkpoints /checkpoints/easyocr \
  --output local/onnx-models
```

The OCR directory contains `craft_mlt_25k.pth` and `english_g2.pth`; the exporter
can download them if absent. Export loads trusted PyTorch checkpoints. Export
into a new directory and switch the mounted bundle after validation; do not
rewrite a bundle used by a running worker. `--component reaction|molecule|ocr`
can export one component. Each graph is compared numerically with its Torch
wrapper. The manifest records source-checkpoint hashes and exported file hashes.

## Verification

```sh
python scripts/check_onnx.py \
  --reaction-checkpoint /checkpoints/pix2seq_reaction_full.ckpt \
  --molecule-checkpoint /checkpoints/swin_base_char_aux_1m.pth \
  --bundle local/onnx-models --images assets/acs.*.png assets/jacs.*.png
python scripts/check_ocr_onnx.py --bundle local/onnx-models \
  --checkpoints /checkpoints/easyocr --images assets/acs.*.png assets/jacs.*.png
```

`check_native_batching.py` measures differences from the old batch-32 molecule
path. `smoke_onnx.py` runs all components and asserts that Torch is absent; run
it in the ONNX image with a writable output path. Validation results and known
differences are summarized in [MODEL_CARD.onnx.md](MODEL_CARD.onnx.md).

## Deliberate limits

- CPU, one image/crop at a time; `batch_size` is retained for the existing call
  site but does not batch neural execution. No throughput or GPU claim.
- Fixed reaction and molecule input sizes; dynamic decoder caches and OCR
  image dimensions. The exporter supports the deployed discrete-coordinate
  molecule checkpoint, not every MolScribe architecture.
- OCR supports the app's English greedy `readtext(crop, detail=0|1)` path.
  It uses float inference. Quantized Torch CPU OCR produced garbled text in
  the ARM validation environment, so the native reader now also disables it.
- Existing tokenization, grammar constraints, chemistry/stereo handling, and
  OCR post-processing are retained. Pure upstream code and licenses live under
  `rxnscribe/_vendor`; no new chemistry parser was written.
- This does not restore atom mapping or make image-to-rule extraction available.
  OCR/model errors still require human review of extracted chemistry.
- Experimental feature branch, separate from `main`. Code and model files are
  public; Docker images are built locally. No production deployment is implied.
