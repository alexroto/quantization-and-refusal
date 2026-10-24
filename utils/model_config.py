"""Model settings shared by inference and later activation experiments."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config_utils import load_named_config


@dataclass
class ModelConfig:
    model_id: str
    dtype: str = "bfloat16"
    batch_size: int = 8
    system_prompt: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False
    refusal_tokens: list[str] = field(default_factory=lambda: ["I"])
    generation: dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 200})
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.generation, dict) or not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("generation and chat_template_kwargs must be mappings")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string or null")


def load_model_config(path: str | Path, name: str) -> ModelConfig:
    return ModelConfig(**load_named_config(path, "models", name))
