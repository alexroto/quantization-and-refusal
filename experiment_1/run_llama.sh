python -m experiment_1.run_inference --model llama2 --dataset harmful --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmless --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset adverserial --quantization bf16 --no-do-sample

python -m experiment_1.run_inference --model llama2 --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmful --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmless --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset adverserial --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample

python -m experiment_1.run_inference --model llama2 --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmful --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmless --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset adverserial --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model llama2 --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample