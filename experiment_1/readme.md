# Experiment 1: response collection

Collect model responses under BF16, INT8, or 4-bit quantization. This runner does
not classify or score responses. Shared loading and generation code lives in
`utils/` for reuse in later experiments.

## Run

Use Python 3.10+ and run from the repository root in an environment with PyTorch,
Transformers, Accelerate, Datasets, and PyYAML installed. Quantized runs also need
bitsandbytes. `requirements.txt` contains the project's environment dependencies;
it includes Linux/CUDA packages and is not a portable macOS environment file.
Set `HF_TOKEN` in the environment for gated checkpoints, after obtaining access.

```bash
python -m experiment_1.run_inference \
  --model llama2 \
  --dataset harmful \
  --quantization bf16 \
  --output results/llama2_harmful_bf16.jsonl \
  --no-do-sample

python -m experiment_1.run_inference \
  --model llama2 \
  --dataset harmful \
  --quantization int4 \
  --output results/llama2_harmful_int4.jsonl \
  --no-do-sample
```

`--help` lists all arguments. Useful overrides include `--batch-size`,
`--max-new-tokens`, `--limit`, `--offset`, `--seed`, `--split`,
`--model-config`, `--dataset-config`, and `--device cuda:0`.
`--do-sample` / `--no-do-sample` override the model YAML's decoding policy;
otherwise the YAML sampling defaults are preserved. Keep prompt selection,
batching, generation settings, seed, and checkpoint revision fixed across
precision comparisons. A fixed seed does not guarantee identical results across
hardware or library versions. Sampled behavioral comparisons need repeated runs
to distinguish precision effects from sampling variation.

## Configuration

Select model and dataset names from `model_configs.yaml` and `datasets.yaml`.
Model entries inherit `defaults`; `generation` and `chat_template_kwargs` merge
key by key. Optional model `revision` can pin a Hugging Face checkpoint commit.
The runner preserves each model's system prompt and chat template settings.

Dataset entries support:

```yaml
datasets:
  example:
    dataset_id: organization/dataset
    file_path: null
    column: prompt
    split: train
    subset: null
    revision: null
    input_column: null
```

If `file_path` is set, it takes precedence and must identify a CSV file. Relative
paths resolve against the dataset YAML's directory. Otherwise, `dataset_id` is
loaded from Hugging Face using the specified subset, split, and revision.
`column` selects the prompt. Optional `input_column` appends nonempty context
with a blank line separator, as configured for Alpaca's instruction/input pairs.
Source columns are preserved, including labels and IDs. Invalid/missing prompt
values fail explicitly; rows are never silently skipped. Selection uses the
original ordered rows, with a zero-based `--offset` and optional `--limit`.

## Precision and context handling

- `bf16`: unquantized weights loaded in BF16, overriding the YAML dtype.
- `int8`: bitsandbytes LLM.int8 with its default outlier threshold.
- `int4`: bitsandbytes NF4 weights, without double quantization. NF4 is a
  4-bit codebook format; the CLI label does not imply uniform integer INT4.

For quantized modes, YAML `dtype` controls nonquantized components and the NF4
compute dtype. These are mixed-precision inference methods, not all-integer
execution. The runner supports quantization on a single CUDA device and requires
a compatible GPU for BF16 CUDA runs. It does not silently fall back to another
precision. Quantized weights are placed with a device map rather than moving the
loaded model. See the [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Prompts are not truncated by default. If a batch plus its generation budget
exceeds the model's declared context window, the runner fails. Explicitly set
`--max-input-tokens` to allow tokenizer truncation; this uses the tokenizer's
truncation side and can remove meaningful prompt content. Chat-formatted prompts
are tokenized without adding a second set of special tokens, following the
[Transformers chat-template guidance](https://huggingface.co/docs/transformers/chat_templating).

## Outputs

Each JSONL line contains `run_id`, model/dataset keys, precision, `row_index`,
`prompt`, `formatted_prompt`, `completion`, `input_tokens`, and the original
`source` record. `completion` contains only generated text with special tokens
removed. A companion `<output>.metadata.json` records the resolved user configs,
model generation defaults, actual quantization config, checkpoint commit when
available, selected-data fingerprint, seed, package versions, device, timestamps,
and run status. These files contain no refusal or compliance labels added by the
runner.

Responses flush after each batch. Failures propagate and mark metadata as failed;
already completed batches remain available. Output paths must be new, so existing
runs cannot be silently overwritten. Automatic resume is intentionally not
implemented: changing sampled batch boundaries can alter results. For a new
partial run, use a new output path and an explicit offset.

## Verification

```bash
python -m unittest discover -s tests -v
```

The offline tests cover config merging, local and Hub dataset contracts, chat
formatting, generated-token slicing, context checks, quantized device placement,
and output/failure handling. They use test doubles for model inference. Before a
full experiment, run a small real-GPU collection with `--limit 2 --batch-size 1`
for each precision to validate the installed inference stack.
