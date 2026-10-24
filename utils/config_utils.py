"""Shared YAML configuration loading."""
from pathlib import Path
from typing import Any

import yaml


def load_named_config(path: str | Path, section: str, name: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get(section), dict):
        raise ValueError(f"{path}: expected a '{section}' mapping")
    if name not in document[section]:
        raise ValueError(f"Unknown {section} key {name!r}; choose from {list(document[section])}")
    defaults = document.get("defaults", {})
    entry = document[section][name]
    if not isinstance(defaults, dict) or not isinstance(entry, dict):
        raise ValueError(f"{path}: defaults and {section}.{name} must be mappings")
    merged = {**defaults, **entry}
    for key in ("generation", "chat_template_kwargs"):
        if key in defaults or key in entry:
            if not isinstance(defaults.get(key, {}), dict) or not isinstance(entry.get(key, {}), dict):
                raise ValueError(f"{key} must be a mapping")
            merged[key] = {**defaults.get(key, {}), **entry.get(key, {})}
    return merged
