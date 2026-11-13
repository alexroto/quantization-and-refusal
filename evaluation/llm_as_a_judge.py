"""Classify saved completions using the OpenAI Responses API."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal

import click
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from utils.cli_utils import CLI_SETTINGS, QUANTIZATION, configured_keys_help

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "output"
RUBRIC_PATH = Path(__file__).with_name("evaluation_prompt.txt")


class Verdict(BaseModel):
    classification: Literal[
        "refusal", "safe-compliance", "unsafe-compliance", "partial-compliance", "over-refusal"
    ]


def judge_row(
    row: dict[str, Any],
    *,
    client: OpenAI,
    judge_model: str,
    rubric: str,
) -> dict[str, Any]:
    """Return a verdict or an explicit evaluation error for one row."""
    try:
        response = client.responses.parse(
            model=judge_model,
            instructions=rubric,
            input=json.dumps({"prompt": row["prompt"], "completion": row["completion"]}),
            text_format=Verdict,
            store=False,
        )
        verdict = response.output_parsed
        if response.status != "completed" or verdict is None:
            return {"status": "error", "error": "no_verdict", "response_id": response.id}
        return {
            "status": "ok",
            "classification": verdict.classification,
            "judge_model": response.model,
            "response_id": response.id,
        }
    except (OpenAIError, ValueError) as exc:
        # Avoid writing provider error bodies, which may include input content.
        return {"status": "error", "error": type(exc).__name__}


def evaluate_parallel(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    client: OpenAI,
    judge_model: str,
    rubric: str,
    workers: int,
) -> Counter[str]:
    """Score bounded batches concurrently and write results in input order."""
    if workers < 1:
        raise ValueError("workers must be positive")
    score = partial(judge_row, client=client, judge_model=judge_model, rubric=rubric)
    rubric_digest = hashlib.sha256(rubric.encode()).hexdigest()
    counts: Counter[str] = Counter()

    with output.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(rows), workers):
            batch = rows[start : start + workers]
            for row, result in zip(batch, pool.map(score, batch)):
                counts[result.get("classification", "error")] += 1
                annotated = {
                    **row,
                    "llm_judge": {
                        **result,
                        "requested_judge_model": judge_model,
                        "rubric_sha256": rubric_digest,
                    },
                }
                handle.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            handle.flush()

    return counts


def load_rows(source: Path, model: str, dataset: str, quantization: str) -> list[dict[str, Any]]:
    """Validate that the file belongs to the requested experiment condition."""
    rows = []
    expected = {"model": model, "dataset": dataset, "quantization": quantization}
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{source.name}:{number}: invalid JSON") from exc
            if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{source.name}:{number}: row does not match the requested condition")
            if any(not isinstance(row.get(key), str) for key in ("prompt", "completion")):
                raise ValueError(f"{source.name}:{number}: expected a string prompt and completion")
            rows.append(row)
    if not rows:
        raise ValueError(f"No completions in {source}")
    return rows


@click.command(context_settings=CLI_SETTINGS)
@click.option("--model", required=True, help="Source model key. Bundled: " + configured_keys_help(ROOT / "model_configs.yaml", "models"))
@click.option("--dataset", required=True, help="Dataset key. Bundled: " + configured_keys_help(ROOT / "datasets.yaml", "datasets"))
@click.option("--quantization", required=True, type=QUANTIZATION, help="Source inference precision.")
@click.option("--seed", default=123, type=click.IntRange(min=0), help="Source filename seed; ignored with --file.")
@click.option("--file", "filename", help="Custom source JSONL filename, without directories.")
@click.option("--judge-model", default="gpt-4o-mini", help="OpenAI model supporting Structured Outputs.")
@click.option("--workers", default=4, type=click.IntRange(min=1), help="Maximum concurrent API requests.")
def main(
    model: str,
    dataset: str,
    quantization: str,
    seed: int,
    filename: str | None,
    judge_model: str,
    workers: int,
) -> None:
    """Send saved prompt/completion pairs to OpenAI for classification."""
    filename = filename or f"{dataset}_seed{seed}.jsonl"
    for value in (model, dataset, filename):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise click.ClickException("Model, dataset, and filename must not contain directories.")
    if Path(filename).suffix.lower() != ".jsonl":
        raise click.ClickException("--file must name a JSONL file.")

    directory = (OUTPUT_ROOT / model / quantization).resolve()
    source = directory / filename
    output = directory / "judge" / filename
    if not source.resolve().is_relative_to(OUTPUT_ROOT.resolve()) or not output.resolve().is_relative_to(directory):
        raise click.ClickException("Input and output must stay within the experiment output directory.")

    try:
        rows = load_rows(source, model, dataset, quantization)
        rubric = RUBRIC_PATH.read_text(encoding="utf-8")
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Judge results already exist: {output}")
        with OpenAI(timeout=60, max_retries=2) as client:
            counts = evaluate_parallel(
                rows, output, client=client, judge_model=judge_model, rubric=rubric, workers=workers
            )
    except (OSError, ValueError, OpenAIError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Saved {sum(counts.values())} rows to {output}")
    click.echo(json.dumps(dict(counts), sort_keys=True))
    if counts["error"]:
        raise click.ClickException("Some rows could not be judged; see their error statuses in the output.")


if __name__ == "__main__":
    main()
