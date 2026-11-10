"""Collect responses without judging refusal or compliance.

Run from the repository root: python -m experiment_1.run_inference --help
"""

import hashlib
import json
import logging
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import click
from transformers import set_seed

from utils.cli_utils import CLI_SETTINGS, QUANTIZATION, configured_keys_help, validate_device
from utils.dataset_utils import load_dataset_config, load_prompt_records
from utils.generation_utils import generate_responses
from utils.model_config import load_model_config
from utils.model_utils import load_model_and_tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "output"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceOptions:
    """Typed settings passed from the CLI to the inference runner."""

    model: str
    dataset: str
    quantization: str = "bf16"
    model_config: Path = ROOT / "model_configs.yaml"
    dataset_config: Path = ROOT / "datasets.yaml"
    output: Path | None = None
    batch_size: int | None = None
    max_new_tokens: int | None = None
    max_input_tokens: int | None = None
    limit: int | None = None
    offset: int = 0
    split: str | None = None
    seed: int = 123
    device: str = "auto"
    do_sample: bool | None = None


def resolve_output_path(args: InferenceOptions) -> Path:
    """Keep each run and its metadata in the selected model and quantization directory."""
    for label, value in (
        ("model", args.model),
        ("dataset", args.dataset),
        ("quantization", args.quantization),
    ):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be a single directory-safe config key")
    filename = args.output or Path(f"{args.dataset}_seed{args.seed}.jsonl")
    if (
        filename.is_absolute()
        or len(filename.parts) != 1
        or "\\" in str(filename)
        or filename.suffix.lower() != ".jsonl"
    ):
        raise ValueError(
            "--output must be a .jsonl filename, without directories; "
            "files are saved under data/output/<model>/<quantization>/"
        )

    model_directory = (OUTPUT_ROOT / args.model).resolve()
    quantization_directory = (model_directory / args.quantization).resolve()
    output_path = quantization_directory / filename
    if not model_directory.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Model output directory must remain inside data/output/")
    if not quantization_directory.is_relative_to(model_directory):
        raise ValueError("Quantization output directory must remain inside the model directory")
    if output_path.is_symlink():
        raise ValueError("Output must not be a symbolic link")

    return output_path


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "accelerate", "bitsandbytes", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    return versions


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: InferenceOptions) -> Path:
    """Collect completions and save response rows and run metadata."""
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

    output_path = resolve_output_path(args)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Output or metadata already exists for {output_path}; "
            "use a new --output filename"
        )

    records = load_prompt_records(dataset_cfg, offset=args.offset, limit=args.limit)
    # Includes source labels and original row indices, so precision runs can be paired.
    serialized_records = json.dumps(
        records, sort_keys=True, ensure_ascii=False, default=str
    )
    fingerprint = hashlib.sha256(serialized_records.encode()).hexdigest()
    metadata = {
        "run_id": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "completed_rows": 0,
        "selected_rows": len(records),
        "model": args.model,
        "dataset": args.dataset,
        "quantization": args.quantization,
        "model_config": asdict(cfg),
        "dataset_config": asdict(dataset_cfg),
        "dataset_sha256": fingerprint,
        "arguments": asdict(args),
        "versions": package_versions(),
        "output_path": str(output_path),
        "quantization_method": {
            "bf16": "unquantized bfloat16",
            "int8": "bitsandbytes LLM.int8",
            "int4": "bitsandbytes NF4 (no double quantization)",
        }[args.quantization],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving completions to %s", output_path)

    with output_path.open("x", encoding="utf-8") as output:
        write_metadata(metadata_path, metadata)

        try:
            set_seed(args.seed)
            LOGGER.info(
                "Loading %s in %s; %d prompts", cfg.model_id, args.quantization, len(records)
            )
            model, tokenizer = load_model_and_tokenizer(
                cfg, quantization=args.quantization, device=args.device
            )

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
                responses = generate_responses(
                    model,
                    tokenizer,
                    cfg,
                    [record["prompt"] for record in batch],
                    max_input_tokens=args.max_input_tokens,
                )

                for record, response in zip(batch, responses):
                    result = {
                        "run_id": metadata["run_id"],
                        "model": args.model,
                        "dataset": args.dataset,
                        "quantization": args.quantization,
                        **record,
                        **response,
                    }
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

    return output_path


@click.command(context_settings=CLI_SETTINGS)
@click.option(
    '--model',
    required=True,
    metavar='MODEL_KEY',
    help=(
        'Model key in --model-config. Bundled keys: '
        + configured_keys_help(ROOT / 'model_configs.yaml', 'models')
        + '. Custom YAML keys are also accepted.'
    ),
)
@click.option(
    '--dataset',
    required=True,
    metavar='DATASET_KEY',
    help=(
        'Dataset key in --dataset-config. Bundled keys: '
        + configured_keys_help(ROOT / 'datasets.yaml', 'datasets')
        + '. Custom YAML keys are also accepted.'
    ),
)
@click.option(
    '--quantization',
    type=QUANTIZATION,
    default='bf16',
    help=(
        "bf16: unquantized BF16; int8: bitsandbytes LLM.int8; int4: NF4 weights."
    ),
)
@click.option(
    '--model-config',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / 'model_configs.yaml',
    help=(
        "Readable YAML file defining model keys/settings."
    ),
)
@click.option(
    '--dataset-config',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / 'datasets.yaml',
    help=(
        "Readable YAML file defining dataset keys/sources."
    ),
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    metavar='FILENAME.jsonl',
    help=(
        "Filename only, without directories. Saved under "
        "data/output/<model>/<quantization>/. When omitted: <dataset>_seed<seed>.jsonl. "
        "Existing files are never overwritten."
    ),
)
@click.option(
    '--batch-size',
    type=click.IntRange(min=1),
    help=(
        "Prompts per batch. When omitted: model YAML batch_size (bundled default: 8)."
    ),
)
@click.option(
    '--max-new-tokens',
    type=click.IntRange(min=1),
    help=(
        "Maximum generated tokens per response. When omitted: model YAML (bundled "
        "default: 200)."
    ),
)
@click.option(
    '--max-input-tokens',
    type=click.IntRange(min=1),
    help=(
        "Opt in to prompt truncation at this token count. When omitted: no truncation."
    ),
)
@click.option(
    '--limit',
    type=click.IntRange(min=1),
    help=(
        "Maximum number of dataset rows to process. When omitted: all remaining rows."
    ),
)
@click.option(
    '--offset',
    type=click.IntRange(min=0),
    default=0,
    help=(
        "Number of original dataset rows to skip before selection (zero-based offset)."
    ),
)
@click.option(
    '--split',
    metavar='HF_SPLIT',
    help=(
        "Hugging Face split name or expression, e.g. train, test, validation, "
        "train[:100]. Must exist in the selected dataset. When omitted: dataset YAML "
        "split (default: train). Unused for CSV input."
    ),
)
@click.option(
    '--seed',
    type=click.IntRange(min=0),
    default=123,
    help=(
        "Random seed for generation."
    ),
)
@click.option(
    '--device',
    default='auto',
    callback=validate_device,
    metavar='DEVICE',
    help=(
        "auto: CUDA if available, otherwise CPU; cpu; cuda; or cuda:N (e.g. cuda:0). "
        "INT8/INT4 require CUDA."
    ),
)
@click.option(
    '--do-sample/--no-do-sample',
    default=None,
    help=(
        "Enable sampling or select greedy decoding. If neither flag is given, inherit the "
        "model YAML do_sample setting (bundled default: enabled)."
    ),
)
def main(
    model: str,
    dataset: str,
    quantization: str,
    model_config: Path,
    dataset_config: Path,
    output: Path | None,
    batch_size: int | None,
    max_new_tokens: int | None,
    max_input_tokens: int | None,
    limit: int | None,
    offset: int,
    split: str | None,
    seed: int,
    device: str,
    do_sample: bool | None,
) -> Path:
    """Generate completions for one model, dataset, and precision without evaluating them."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    options = InferenceOptions(
        model=model,
        dataset=dataset,
        quantization=quantization,
        model_config=model_config,
        dataset_config=dataset_config,
        output=output,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        limit=limit,
        offset=offset,
        split=split,
        seed=seed,
        device=device,
        do_sample=do_sample,
    )

    try:
        return run(options)
    except (ValueError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
