"""Offline contract tests: no model downloads or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from experiment_1.run_inference import build_parser, run
from utils.dataset_utils import DatasetConfig, load_dataset_config, load_prompt_records
from utils.generation_utils import generate_responses
from utils.model_config import ModelConfig, load_model_config
from utils.model_utils import format_chat_prompt, load_model_and_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class ConfigAndDatasetTests(unittest.TestCase):
    def test_defaults_and_nested_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'models.yaml'
            path.write_text('defaults:\n  generation:\n    max_new_tokens: 10\n    do_sample: false\n'
                            'models:\n  tiny:\n    model_id: test/tiny\n'
                            '    generation:\n      max_new_tokens: 4\n')
            cfg = load_model_config(path, 'tiny')
            self.assertEqual(cfg.generation, {'max_new_tokens': 4, 'do_sample': False})
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                load_model_config(path, 'absent')

    def test_local_configs_use_actual_prompts(self):
        cfg = load_dataset_config(ROOT / 'datasets.yaml', 'harmful')
        record = load_prompt_records(cfg, offset=1, limit=1)[0]
        self.assertEqual(record['row_index'], 1)
        self.assertEqual(record['prompt'], record['source']['goal'])
        self.assertNotEqual(record['prompt'], record['source']['target'])
        self.assertTrue(Path(cfg.file_path).is_absolute())

    def test_csv_selection_validation_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('prompt,label\nhello,benign\n,unknown\n')
            cfg = DatasetConfig(column='prompt', file_path=str(path))
            self.assertEqual(load_prompt_records(cfg, limit=1)[0]['source']['label'], 'benign')
            with self.assertRaisesRegex(ValueError, 'Row 1'):
                load_prompt_records(cfg)
            with self.assertRaisesRegex(ValueError, 'Missing dataset columns'):
                load_prompt_records(DatasetConfig(column='missing', file_path=str(path)))
            with self.assertRaisesRegex(ValueError, 'no prompts'):
                load_prompt_records(cfg, offset=10)
            with self.assertRaisesRegex(ValueError, 'CSV file'):
                load_prompt_records(DatasetConfig(column='prompt', file_path=directory))

    def test_hub_loading_split_revision_and_input(self):
        class Rows(list):
            column_names = ['instruction', 'input', 'label']
        loader = MagicMock(return_value=Rows([
            {'instruction': 'Summarize', 'input': 'some text', 'label': 1},
            {'instruction': 'Hello', 'input': '', 'label': 0},
        ]))
        with patch.dict(sys.modules, {'datasets': types.SimpleNamespace(load_dataset=loader)}):
            cfg = DatasetConfig(column='instruction', dataset_id='test/data', subset='subset',
                                split='test', revision='abc', input_column='input')
            rows = load_prompt_records(cfg)
        loader.assert_called_once_with('test/data', name='subset', split='test', revision='abc')
        self.assertEqual(rows[0]['prompt'], 'Summarize\n\nsome text')
        self.assertEqual(rows[1]['prompt'], 'Hello')

    def test_system_prompt_none_and_empty_are_distinct(self):
        tokenizer = MagicMock()
        for system, expected in [(None, [{'role': 'user', 'content': 'hello'}]),
                                 ('', [{'role': 'system', 'content': ''},
                                       {'role': 'user', 'content': 'hello'}])]:
            format_chat_prompt(tokenizer, ModelConfig('test/model', system_prompt=system), 'hello')
            self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], expected)


class GenerationTests(unittest.TestCase):
    def test_prompt_tokens_are_not_decoded_and_greedy_sampling_args_removed(self):
        torch = types.SimpleNamespace(inference_mode=MagicMock())
        tokenizer = MagicMock()
        tokenizer.chat_template = 'template'
        tokenizer.pad_token_id = 0
        tokenizer.apply_chat_template.side_effect = ['formatted a', 'formatted b']
        ids, mask = MagicMock(), MagicMock()
        ids.shape = (2, 8)
        ids.to.return_value = ids
        mask.to.return_value = mask
        mask.sum.return_value.tolist.return_value = [8, 5]
        tokenizer.return_value = {'input_ids': ids, 'attention_mask': mask}
        tokenizer.batch_decode.return_value = ['answer a', 'answer b']
        model = MagicMock()
        model.config.max_position_embeddings = 32
        outputs = model.generate.return_value
        cfg = ModelConfig('test/model', generation={'max_new_tokens': 4, 'do_sample': False,
                                                    'temperature': 0.7, 'top_p': 0.9})
        with patch.dict(sys.modules, {'torch': torch}):
            rows = generate_responses(model, tokenizer, cfg, ['a', 'b'])
        self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
        self.assertNotIn('truncation', tokenizer.call_args.kwargs)
        self.assertNotIn('temperature', model.generate.call_args.kwargs)
        self.assertEqual(model.generate.call_args.kwargs['pad_token_id'], 0)
        outputs.__getitem__.assert_called_once_with((slice(None), slice(8, None)))
        self.assertEqual(rows[1], {'formatted_prompt': 'formatted b', 'completion': 'answer b',
                                   'input_tokens': 5})
        tokenizer.apply_chat_template.side_effect = ['formatted a', 'formatted b']
        model.config.max_position_embeddings = 10
        with patch.dict(sys.modules, {'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'context window'):
                generate_responses(model, tokenizer, cfg, ['a', 'b'])

    def test_quantized_load_uses_device_map_without_moving_model(self):
        torch = MagicMock()
        class Device:
            type = 'cuda'
            def __str__(self):
                return 'cuda:0'
        torch.device.return_value = Device()
        transformers = MagicMock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.pad_token_id = 0
        model = transformers.AutoModelForCausalLM.from_pretrained.return_value
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            loaded, _ = load_model_and_tokenizer(ModelConfig('test/model'), quantization='int4')
        self.assertIs(loaded, model)
        model.to.assert_not_called()
        kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['device_map'], {'': 'cuda:0'})
        quant = transformers.BitsAndBytesConfig.call_args.kwargs
        self.assertTrue(quant['load_in_4bit'])
        self.assertFalse(quant['load_in_8bit'])
        self.assertEqual(quant['bnb_4bit_quant_type'], 'nf4')
        torch.device.return_value = types.SimpleNamespace(type='cpu')
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            with self.assertRaisesRegex(ValueError, 'require a CUDA'):
                load_model_and_tokenizer(ModelConfig('test/model'), quantization='int8', device='cpu')


class RunnerTests(unittest.TestCase):
    def execute(self, directory, *, fail=False):
        output = Path(directory) / 'responses.jsonl'
        args = build_parser().parse_args(['--model', 'qwen', '--dataset', 'harmful',
                                         '--output', str(output), '--limit', '3', '--batch-size', '2',
                                         '--no-do-sample'])
        model, tokenizer = MagicMock(), MagicMock()
        model.config._commit_hash = 'commit'
        model.config.quantization_config = None
        model.generation_config.to_dict.return_value = {'max_new_tokens': 20}
        tokenizer.chat_template = 'template'
        def respond(model, tokenizer, cfg, prompts, **kwargs):
            if fail and len(prompts) == 1:
                raise RuntimeError('generation failed')
            return [{'completion': f'response {p}', 'formatted_prompt': p, 'input_tokens': 5}
                    for p in prompts]
        with patch.dict(sys.modules, {'transformers': types.SimpleNamespace(set_seed=MagicMock())}), \
             patch('experiment_1.run_inference.load_model_and_tokenizer', return_value=(model, tokenizer)), \
             patch('experiment_1.run_inference.generate_responses', side_effect=respond):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    run(args)
            else:
                run(args)
        return args, output, json.loads(Path(str(output) + '.metadata.json').read_text())

    def test_batched_runner_records_provenance_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, metadata = self.execute(directory)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual([r['row_index'] for r in rows], [0, 1, 2])
            self.assertIn('goal', rows[0]['source'])
            self.assertNotIn('is_harmful', rows[0])
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['completed_rows'], 3)
            self.assertFalse(metadata['model_config']['generation']['do_sample'])
            self.assertEqual(len(metadata['dataset_sha256']), 64)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_failure_preserves_completed_batches_and_records_status(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, metadata = self.execute(directory, fail=True)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['completed_rows'], 2)
            self.assertIn('generation failed', metadata['error'])


if __name__ == '__main__':
    unittest.main()
