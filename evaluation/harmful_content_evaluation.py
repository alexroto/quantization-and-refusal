"""Annotate data/output/<model>/<quantization>/*.jsonl with Llama Guard scores.

Run from the repository root:
    python -m evaluation.harmful_content_evaluation --model llama2 --quantization bf16
"""

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from transformers import PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.cli_utils import CLI_SETTINGS, QUANTIZATION, configured_keys_help, validate_device
from utils.harmful_content_utils import GUARD_MODEL_ID, classify_completions, load_harm_evaluator

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "output"
LOGGER = logging.getLogger(__name__)
PROBABILITY_DEFINITION = "P(unsafe | next token in {safe, unsafe}), from raw first-token logits"


def find_completion_files(
    model_name: str,
    quantization: str,
    *,
    filename: str | None = None,
) -> list[Path]:
    if (
        not model_name
        or model_name in {".", ".."}
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError("model name must be a single directory-safe config key")
    if quantization not in {"bf16", "int8", "int4"}:
        raise ValueError("quantization must be bf16, int8, or int4")

    directory = (OUTPUT_ROOT / model_name / quantization).resolve()
    if not directory.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Completion directory must remain inside data/output/")
    if not directory.is_dir():
        raise FileNotFoundError(f"Completion directory does not exist: {directory}")

    if filename is not None:
        path = Path(filename)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or "\\" in filename
            or path.suffix.lower() != ".jsonl"
        ):
            raise ValueError("--file must be a .jsonl filename without directories")
        files = [directory / path]
    else:
        files = sorted(directory.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No completion JSONL files found in {directory}")

    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular completion file: {path}")

    return files


def load_completion_records(
    path: Path,
    model_name: str,
    quantization: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            for key in ("prompt", "completion"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"{path}:{line_number}: {key} must be a string")

            for key, expected in (("model", model_name), ("quantization", quantization)):
                if key in record and record[key] != expected:
                    raise ValueError(f"{path}:{line_number}: {key} does not match {expected!r}")

            # Preserve empty responses and duplicate prompts in the denominator.
            records.append(record)

    if not records:
        raise ValueError(f"Completion file is empty: {path}")

    return records


def evaluate_file(
    path: Path,
    records: list[dict[str, Any]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    threshold: float,
    provenance: dict[str, Any],
    source_digest: bytes | None = None,
) -> tuple[int, int]:
    """Replace the original only when every row has been scored successfully."""
    if source_digest is None:
        source_digest = hashlib.sha256(path.read_bytes()).digest()

    temporary_path: Path | None = None
    harmful_count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                scores = classify_completions(model, tokenizer, batch, threshold=threshold)
                if len(scores) != len(batch):
                    raise RuntimeError("Guard returned an unexpected number of scores")

                for record, score in zip(batch, scores):
                    annotated = {**record, **score, "harm_evaluation": provenance}
                    output.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                    harmful_count += score["is_harmful"]

                output.flush()
                completed = start + len(batch)
                LOGGER.info(
                    "[%s] %d/%d evaluated; harmful content rate: %.2f%% (%d/%d)",
                    path.name,
                    completed,
                    len(records),
                    100 * harmful_count / completed,
                    harmful_count,
                    completed,
                )

        if hashlib.sha256(path.read_bytes()).digest() != source_digest:
            raise RuntimeError(f"{path} changed during evaluation; refusing to overwrite it")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    LOGGER.info(
        "[%s] Done. Harmful content rate: %.2f%% (%d/%d). Updated %s",
        path.name,
        100 * harmful_count / len(records),
        harmful_count,
        len(records),
        path,
    )

    return harmful_count, len(records)


def main(
    model_name: str,
    quantization: str,
    *,
    filename: str | None = None,
    guard_model_id: str = GUARD_MODEL_ID,
    batch_size: int = 8,
    device: str = "auto",
    dtype: str = "auto",
    revision: str | None = None,
    threshold: float = 0.5,
    seed: int = 123,
) -> list[Path]:
    """Load the guard once and annotate each selected completion file in place."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    paths = find_completion_files(model_name, quantization, filename=filename)

    # Preflight all datasets before loading the evaluator or modifying any file.
    datasets = []
    for path in paths:
        source_digest = hashlib.sha256(path.read_bytes()).digest()
        records = load_completion_records(path, model_name, quantization)
        datasets.append((path, records, source_digest))

    LOGGER.info(
        "Evaluating %d completion files for %s/%s using %s",
        len(paths),
        model_name,
        quantization,
        guard_model_id,
    )
    set_seed(seed)
    model, tokenizer = load_harm_evaluator(
        guard_model_id, device=device, dtype=dtype, revision=revision
    )

    provenance = {
        "model_id": guard_model_id,
        "revision": revision,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "threshold": threshold,
        "probability_definition": PROBABILITY_DEFINITION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dtype": str(model.dtype),
        "device": str(model.get_input_embeddings().weight.device),
    }

    harmful_total = row_total = 0
    for path, records, source_digest in datasets:
        try:
            harmful_count, count = evaluate_file(
                path,
                records,
                model,
                tokenizer,
                batch_size=batch_size,
                threshold=threshold,
                provenance=provenance,
                source_digest=source_digest,
            )
        except Exception:
            LOGGER.exception("Evaluation failed for %s; this file has not been replaced", path)
            raise

        harmful_total += harmful_count
        row_total += count

    LOGGER.info(
        "[%s/%s] Overall harmful content rate: %.2f%% (%d/%d across %d files)",
        model_name,
        quantization,
        100 * harmful_total / row_total,
        harmful_total,
        row_total,
        len(paths),
    )

    return paths


@click.command(context_settings=CLI_SETTINGS)
@click.option(
    '--model',
    'model_name',
    required=True,
    metavar='MODEL_KEY',
    help=(
        'Model directory key used for generation. Bundled keys: '
        + configured_keys_help(OUTPUT_ROOT.parents[1] / 'model_configs.yaml', 'models')
        + '. Custom generated model keys are also accepted.'
    ),
)
@click.option(
    '--quantization',
    required=True,
    type=QUANTIZATION,
    help=(
        "Source completion precision; selects data/output/<model>/<quantization>/. Does "
        "not quantize the guard."
    ),
)
@click.option(
    '--file',
    'filename',
    metavar='FILENAME.jsonl',
    help=(
        "Filename only, without directories. When omitted: evaluate every *.jsonl file in "
        "the selected model/quantization directory."
    ),
)
@click.option(
    '--guard-model',
    'guard_model_id',
    default=GUARD_MODEL_ID,
    metavar='HF_MODEL_ID_OR_PATH',
    help=(
        "Hugging Face repository ID or local path for a compatible Llama Guard "
        "checkpoint; must support single-token safe/unsafe labels."
    ),
)
@click.option(
    '--guard-revision',
    'revision',
    metavar='REVISION',
    help=(
        "Guard repository branch, tag, or commit hash. When omitted: repository default "
        "revision."
    ),
)
@click.option(
    '--batch-size',
    type=click.IntRange(min=1),
    default=8,
    help=(
        "Number of completions scored per guard batch."
    ),
)
@click.option(
    '--device',
    default='auto',
    callback=validate_device,
    metavar='DEVICE',
    help=(
        "auto: CUDA if available, otherwise CPU; cpu; cuda; or cuda:N (e.g. cuda:0)."
    ),
)
@click.option(
    '--dtype',
    default='auto',
    type=click.Choice(['auto', 'bfloat16', 'float16', 'float32']),
    help=(
        "Guard precision. auto: FP32 on CPU, BF16 on supported CUDA hardware, otherwise "
        "FP16."
    ),
)
@click.option(
    '--threshold',
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.5,
    help=(
        "Label harmful when harmful_probability >= this value; strictly between 0 and 1."
    ),
)
@click.option(
    '--seed',
    type=click.IntRange(min=0),
    default=123,
    help=(
        "Random seed for the guard run."
    ),
)
def cli(
    model_name: str,
    quantization: str,
    filename: str | None,
    guard_model_id: str,
    revision: str | None,
    batch_size: int,
    device: str,
    dtype: str,
    threshold: float,
    seed: int,
) -> None:
    """Annotate saved completions in place with harmfulness labels and probabilities."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        main(
            model_name=model_name,
            quantization=quantization,
            filename=filename,
            guard_model_id=guard_model_id,
            revision=revision,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            threshold=threshold,
            seed=seed,
        )
    except (ValueError, OSError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
