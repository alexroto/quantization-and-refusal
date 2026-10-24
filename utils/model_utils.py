"""Model loading and chat formatting shared across experiments."""
import os

from utils.model_config import ModelConfig


def load_model_and_tokenizer(cfg: ModelConfig, *, quantization: str = "none",
                             device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if quantization not in {"none", "bf16", "int8", "int4"}:
        raise ValueError(f"Unsupported quantization: {quantization}")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    dtype = torch.bfloat16 if quantization == "bf16" else getattr(torch, cfg.dtype)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if quantization in {"int8", "int4"} and target.type != "cuda":
        raise ValueError("This runner's bitsandbytes INT8/INT4 modes require a CUDA device")
    if target.type == "cuda" and dtype == torch.bfloat16:
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support BF16; use a compatible GPU")
    common = {"token": os.getenv("HF_TOKEN"), "trust_remote_code": cfg.trust_remote_code,
              "revision": cfg.revision}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a padding nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if cfg.chat_template is not None:
        tokenizer.chat_template = cfg.chat_template
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; configure chat_template in model YAML")
    kwargs = {**common, "dtype": dtype}
    if quantization in {"int8", "int4"}:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "int8", load_in_4bit=quantization == "int4",
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False,
        )
        kwargs["device_map"] = {"": str(target)}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    if quantization not in {"int8", "int4"}:
        model = model.to(target)
    model.eval()
    return model, tokenizer


def format_chat_prompt(tokenizer, cfg: ModelConfig, prompt: str) -> str:
    messages = []
    if cfg.system_prompt is not None:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **cfg.chat_template_kwargs,
    )
