"""Batched generation; returns only newly generated response text."""
from utils.model_config import ModelConfig
from utils.model_utils import format_chat_prompt


def generate_responses(model, tokenizer, cfg: ModelConfig, prompts: list[str], *,
                       max_input_tokens: int | None = None) -> list[dict]:
    import torch

    formatted = [format_chat_prompt(tokenizer, cfg, prompt) for prompt in prompts]
    tokenization = {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
    if max_input_tokens is not None:
        # Explicit opt-in: truncation can remove safety-relevant prompt content.
        tokenization.update(truncation=True, max_length=max_input_tokens)
    inputs = tokenizer(formatted, **tokenization)
    width = inputs["input_ids"].shape[1]
    generation = dict(cfg.generation)
    if generation.get("num_return_sequences", 1) != 1:
        raise ValueError("Only num_return_sequences=1 is supported")
    if generation.get("return_dict_in_generate", False):
        raise ValueError("return_dict_in_generate must be false")
    max_new = generation.get("max_new_tokens", 200)
    if type(max_new) is not int or max_new < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    generation["max_new_tokens"] = max_new
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and width + max_new > context_limit:
        raise ValueError(f"Prompt batch ({width} tokens) + max_new_tokens ({max_new}) exceeds "
                         f"context window ({context_limit}); select --max-input-tokens explicitly")
    # Use the embedding device; quantized models must not be moved with .to().
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation.setdefault("pad_token_id", tokenizer.pad_token_id)
    if not generation.get("do_sample", False):
        for key in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            generation.pop(key, None)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation)
    completions = tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
    lengths = inputs["attention_mask"].sum(dim=1).tolist()
    if len(completions) != len(prompts):
        raise RuntimeError("Model returned an unexpected number of completions")
    return [{"formatted_prompt": text, "completion": completion, "input_tokens": length}
            for text, completion, length in zip(formatted, completions, lengths)]
