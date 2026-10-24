"""Collect responses without judging refusal or compliance.

Run from the repository root: python -m experiment_1.run_inference --help
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import uuid

from utils.dataset_utils import load_dataset_config, load_prompt_records
from utils.generation_utils import generate_responses
from utils.model_config import load_model_config
from utils.model_utils import load_model_and_tokenizer

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Key in model config YAML")
    parser.add_argument("--dataset", required=True, help="Key in dataset config YAML")
    parser.add_argument("--quantization", choices=["bf16", "int8", "int4"], default="bf16",
                        help="INT8 uses LLM.int8; INT4 uses bitsandbytes NF4 weights")
    parser.add_argument("--model-config", type=Path, default=ROOT / "model_configs.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "datasets.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New output JSONL file")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--max-input-tokens", type=positive_int,
                        help="Opt in to tokenizer truncation at this length")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--offset", type=nonnegative_int, default=0)
    parser.add_argument("--split", help="Override Hugging Face dataset split")
    parser.add_argument("--seed", type=nonnegative_int, default=123)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None,
                        help="Override YAML decoding policy; --no-do-sample selects greedy decoding")
    return parser


def package_versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "accelerate", "bitsandbytes", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def write_metadata(path, metadata):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run(args):
    cfg = load_model_config(args.model_config, args.model)
    dataset_cfg = load_dataset_config(args.dataset_config, args.dataset)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_new_tokens is not None:
        cfg.generation["max_new_tokens"] = args.max_new_tokens
    if args.do_sample is not None:
        cfg.generation["do_sample"] = args.do_sample
    if args.quantization == "bf16":
        cfg.dtype = "bfloat16"
    if args.split is not None:
        dataset_cfg.split = args.split
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(f"Output or metadata already exists for {args.output}; use a new path")
    records = load_prompt_records(dataset_cfg, offset=args.offset, limit=args.limit)
    # Includes source labels and original row indices, so precision runs can be paired.
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False,
                                            default=str).encode()).hexdigest()
    metadata = {
        "run_id": uuid.uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "completed_rows": 0, "selected_rows": len(records),
        "model": args.model, "dataset": args.dataset, "quantization": args.quantization,
        "model_config": asdict(cfg), "dataset_config": asdict(dataset_cfg),
        "dataset_sha256": fingerprint, "arguments": vars(args), "versions": package_versions(),
        "quantization_method": {"bf16": "unquantized bfloat16", "int8": "bitsandbytes LLM.int8",
                                "int4": "bitsandbytes NF4 (no double quantization)"}[args.quantization],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        write_metadata(metadata_path, metadata)
        try:
            from transformers import set_seed
            set_seed(args.seed)
            LOGGER.info("Loading %s in %s; %d prompts", cfg.model_id, args.quantization, len(records))
            model, tokenizer = load_model_and_tokenizer(cfg, quantization=args.quantization,
                                                        device=args.device)
            metadata["model_commit"] = getattr(model.config, "_commit_hash", None)
            metadata["device"] = str(model.get_input_embeddings().weight.device)
            metadata["hardware"] = platform.platform()
            quant_cfg = getattr(model.config, "quantization_config", None)
            metadata["resolved_quantization_config"] = quant_cfg.to_dict() if quant_cfg else None
            metadata["resolved_generation_config"] = model.generation_config.to_dict()
            metadata["chat_template"] = tokenizer.chat_template
            write_metadata(metadata_path, metadata)
            for start in range(0, len(records), cfg.batch_size):
                batch = records[start:start + cfg.batch_size]
                responses = generate_responses(model, tokenizer, cfg, [r["prompt"] for r in batch],
                                               max_input_tokens=args.max_input_tokens)
                for record, response in zip(batch, responses):
                    result = {"run_id": metadata["run_id"], "model": args.model,
                              "dataset": args.dataset, "quantization": args.quantization,
                              **record, **response}
                    output.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                output.flush()
                metadata["completed_rows"] += len(batch)
                write_metadata(metadata_path, metadata)
                LOGGER.info("Saved %d/%d responses", metadata["completed_rows"], len(records))
            metadata["status"] = "completed"
        except BaseException as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_metadata(metadata_path, metadata)
    return args.output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
