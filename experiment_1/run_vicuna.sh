python -m experiment_1.run_inference --model vicuna --dataset harmful --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmless --quantization bf16 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset adverserial --quantization bf16 --no-do-sample

python -m experiment_1.run_inference --model vicuna --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmful --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmless --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset adverserial --quantization int8 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmless_with_sensitive_words --quantization int8 --no-do-sample

python -m experiment_1.run_inference --model vicuna --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmful --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmless --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset adverserial --quantization int4 --no-do-sample
python -m experiment_1.run_inference --model vicuna --dataset harmless_with_sensitive_words --quantization int4 --no-do-sample