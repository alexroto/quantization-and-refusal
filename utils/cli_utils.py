"""Shared Click parameter validation for experiment entry points."""
import re

import click
import yaml

CLI_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}
QUANTIZATION = click.Choice(["bf16", "int8", "int4"])


def configured_keys_help(path, section: str) -> str:
    """List bundled keys in help without restricting custom YAML configurations."""
    try:
        with path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        return ", ".join(document[section])
    except (OSError, TypeError, KeyError, yaml.YAMLError):
        return f"see {path.name}"


def validate_device(ctx, param, value):
    if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", value):
        raise click.BadParameter("choose auto, cpu, cuda, or cuda:N (N is a nonnegative GPU index)")
    return value
