"""Dynamic INT8 quantisation, with the before/after measured rather than assumed.

Quantisation stores weights as 8-bit integers instead of 32-bit floats. It is
worth shipping only if it makes the file meaningfully smaller without moving
the predictions, so this module measures both and reports them; the caller
decides which file to publish.

For the models this project ships today the answer is "not applicable": each
is one linear layer of a few dozen weights, exported as the ONNX-ML
`LinearClassifier` operator, which dynamic quantisation has no integer form
for. The check stays in the pipeline so that a larger future model (the
roadmap's distilled transformer) is quantised and measured automatically.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

# A quantised model is preferred only if it saves at least this share of the
# file and keeps every sampled probability within this distance of the original.
MIN_SIZE_SAVING = 0.10
MAX_PROBABILITY_SHIFT = 0.01

# Operators whose weights dynamic quantisation can store as integers.
QUANTISABLE_OPS = frozenset({"MatMul", "Gemm", "Conv", "LSTM", "GRU", "Attention", "Gather"})


def _probabilities(onnx_bytes: bytes, features: np.ndarray, input_name: str) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    return session.run(None, {input_name: features.astype(np.float32)})[1][:, 1]


def quantise_and_measure(
    onnx_bytes: bytes, features: np.ndarray, input_name: str
) -> tuple[bytes | None, dict[str, Any]]:
    """Return (quantised model or None, report). None means there was nothing to quantise."""
    report: dict[str, Any] = {"original_kb": len(onnx_bytes) / 1024}
    operators = sorted({node.op_type for node in onnx.load_from_string(onnx_bytes).graph.node})
    if not QUANTISABLE_OPS & set(operators):
        report.update(
            quantised_kb=None, shipped="float32",
            reason=f"not applicable: the graph is {' + '.join(operators)}, "
                   "which has no layer dynamic quantisation can convert",
        )
        return None, report

    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "model.onnx", Path(tmp) / "model.int8.onnx"
        source.write_bytes(onnx_bytes)
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
        quantised = target.read_bytes()
    shift = float(np.max(np.abs(
        _probabilities(onnx_bytes, features, input_name)
        - _probabilities(quantised, features, input_name)
    )))
    saving = 1 - len(quantised) / len(onnx_bytes)
    report.update(quantised_kb=len(quantised) / 1024, size_saving=saving,
                  max_probability_shift=shift)
    if saving < MIN_SIZE_SAVING:
        report.update(shipped="float32", reason="quantisation did not make the file smaller")
    elif shift > MAX_PROBABILITY_SHIFT:
        report.update(shipped="float32", reason="quantisation moved the predictions too far")
    else:
        report.update(shipped="int8", reason="smaller, with predictions unchanged within tolerance")
    return quantised, report
