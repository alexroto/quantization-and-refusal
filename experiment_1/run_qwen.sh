python -m experiment_1.run_inference --model qwen --dataset harmful --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmless --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset adverserial --quantization bf16 --no-do-sample

python -m experiment_1.run_inference --model qwen --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmful --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmless --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset adverserial --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample

python -m experiment_1.run_inference --model qwen --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmful --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmless --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset adverserial --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model qwen --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample