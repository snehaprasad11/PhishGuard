"""The whole training pipeline as one command (spec FR9).

Usage (from the repo root):
    python -m training.pipeline

Reads the pinned data, trains and calibrates the model, benchmarks the
alternatives, exports to ONNX (refusing if the export is not faithful),
trains and exports the message-text model the same way, evaluates on the
held-out split and writes the model card. Each
stage logs its own MLflow run. Fetching data is deliberately not part of this:
the pipeline only ever reads what the manifest pins.
"""

from training import benchmark, evaluate, export_onnx, train
from training.text import pipeline as text_pipeline


def main() -> None:
    train.main()
    benchmark.main()
    export_onnx.main()
    text_pipeline.main()
    evaluate.main()


if __name__ == "__main__":
    main()
