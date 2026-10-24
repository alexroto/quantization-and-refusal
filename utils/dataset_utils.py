"""Load prompt records without dropping source labels or changing row order."""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.config_utils import load_named_config


@dataclass
class DatasetConfig:
    column: str
    dataset_id: str | None = None
    file_path: str | None = None
    split: str = "train"
    subset: str | None = None
    revision: str | None = None
    input_column: str | None = None

    def __post_init__(self):
        if not self.file_path and not self.dataset_id:
            raise ValueError("Dataset requires file_path or dataset_id")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("Dataset column must be a nonempty string")


def load_dataset_config(path: str | Path, name: str) -> DatasetConfig:
    config = DatasetConfig(**load_named_config(path, "datasets", name))
    if config.file_path:
        # Relative data paths are anchored to the YAML file, not the working directory.
        config.file_path = str((Path(path).resolve().parent / config.file_path).resolve())
    return config


def load_prompt_records(config: DatasetConfig, *, offset: int = 0,
                        limit: int | None = None) -> list[dict[str, Any]]:
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("offset must be nonnegative and limit must be positive")
    if config.file_path:
        path = Path(config.file_path)
        if path.suffix.lower() != ".csv" or not path.is_file():
            raise ValueError(f"Expected a CSV file at {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    else:
        from datasets import load_dataset
        dataset = load_dataset(config.dataset_id, name=config.subset,
                               split=config.split, revision=config.revision)
        columns = dataset.column_names
        rows = dataset
    required = [config.column] + ([config.input_column] if config.input_column else [])
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"Missing dataset columns {sorted(missing)}; available: {columns}")
    stop = len(rows) if limit is None else min(len(rows), offset + limit)
    records = []
    for index in range(offset, stop):
        source = dict(rows[index])
        prompt = source[config.column]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {index}: {config.column!r} must contain a nonempty string")
        if config.input_column:
            context = source[config.input_column]
            if context is not None and not isinstance(context, str):
                raise ValueError(f"Row {index}: input context must be a string or null")
            if context and context.strip():
                prompt = f"{prompt}\n\n{context}"
        records.append({"row_index": index, "prompt": prompt, "source": source})
    if not records:
        raise ValueError("Dataset selection contains no prompts")
    return records
