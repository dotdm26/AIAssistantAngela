"""Export a Hugging Face feature model and create an INT8 ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def export_model(model_name: str, output_directory: Path, *, quantize: bool = True) -> Path:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        trust_remote_code=True,
    )
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)

    fp32_model = output_directory / "model.onnx"
    if not fp32_model.exists():
        raise FileNotFoundError(f"ONNX exporter did not create {fp32_model}")

    selected_model = fp32_model
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_model = output_directory / "model.int8.onnx"
        quantize_dynamic(
            str(fp32_model),
            str(int8_model),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        selected_model = int8_model

    digest = hashlib.sha256(selected_model.read_bytes()).hexdigest()
    (output_directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "backend": "onnx",
                "artifact": selected_model.name,
                "sha256": digest,
                "quantized": quantize,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    artifact = export_model(args.model, args.output, quantize=not args.no_quantize)
    print(f"Exported {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
