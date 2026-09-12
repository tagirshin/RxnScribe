"""Full inference in an environment where PyTorch is not installed."""

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

from rxnscribe import RxnScribeONNX

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--bundle", required=True)
p.add_argument("--image", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
assert importlib.util.find_spec("torch") is None, "Run in the Torch-free image"
start = time.perf_counter()
model = RxnScribeONNX(a.bundle)
results = model.predict_image_file(a.image, molscribe=True, ocr=True)
assert results, "Expected reactions in this fixture"
assert any(
    b.get("smiles") for r in results for k in ["reactants", "products"] for b in r[k]
)
assert any(b.get("text") for r in results for b in r["conditions"])
assert not any(m == "torch" or m.startswith("torch.") for m in sys.modules)
output = dict(
    reactions=results,
    seconds=time.perf_counter() - start,
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
)
Path(a.output).write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        dict(
            reactions=len(results),
            seconds=output["seconds"],
            max_rss_kib=output["max_rss_kib"],
        )
    ),
    flush=True,
)
