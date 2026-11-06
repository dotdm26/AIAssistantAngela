"""Export a Hugging Face feature model and create an INT8 ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def export_model(
    model_name: str,
    output_directory: Path,
    *,
    quantize: bool = True,
    query_prefix: str = "",
    document_prefix: str = "",
    max_length: int = 256,
) -> Path:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            trust_remote_code=True,
        )
    except ValueError as exc:
        if "custom or unsupported architecture" in str(exc):
            raise RuntimeError(
                f"{model_name!r} uses an architecture that Optimum cannot export automatically. "
                f"Use the supported default model ({DEFAULT_MODEL}) or supply a pre-converted ONNX model."
            ) from exc
        raise
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
                "query_prefix": query_prefix,
                "document_prefix": document_prefix,
                "max_length": max_length,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    artifact = export_model(
        args.model,
        args.output,
        quantize=not args.no_quantize,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        max_length=args.max_length,
    )
    print(f"Exported {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
