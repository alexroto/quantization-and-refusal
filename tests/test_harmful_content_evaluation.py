"""Guard contracts and atomic JSONL annotation, without downloads or a GPU."""
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from evaluation.harmful_content_evaluation import evaluate_file, find_completion_files, main
from utils.harmful_content_utils import classify_completions, get_label_token_ids


class Tensor:
    """Small tensor double to exercise actual score selection and normalization."""
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def __getitem__(self, key):
        rows, columns = key
        values = self.values[rows]
        if isinstance(columns, list):
            return Tensor([[row[column] for column in columns] for row in values])
        return Tensor([row[columns] for row in values])

    def float(self):
        return self

    def to(self, device):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def softmax(tensor, dim):
    if dim != -1:
        raise AssertionError('Expected normalization over class logits')
    values = []
    for row in tensor.values:
        weights = [math.exp(value - max(row)) for value in row]
        values.append([weight / sum(weights) for weight in weights])
    return Tensor(values)


class GuardScoringTests(unittest.TestCase):
    def test_raw_logits_normalized_over_safe_and_unsafe(self):
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = [[1], [3]]
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {'input_ids': Tensor([[1, 2], [0, 3]]),
                                  'attention_mask': Tensor([[1, 1], [0, 1]])}
        tokenizer.apply_chat_template.side_effect = ['guard prompt 1', 'guard prompt 2']
        model = MagicMock()
        model.config.max_position_embeddings = 20
        # Large unrelated vocabulary scores must not change the conditional probability.
        model.generate.return_value.logits = [Tensor([[100, 0, 10, math.log(4)],
                                                       [100, math.log(9), 10, 0]])]
        torch = MagicMock()
        torch.softmax.side_effect = softmax
        torch.isfinite.side_effect = lambda tensor: types.SimpleNamespace(
            all=lambda: types.SimpleNamespace(item=lambda: all(math.isfinite(x) for x in tensor.values)))
        records = [{'prompt': 'p1', 'completion': 'c1'}, {'prompt': 'p2', 'completion': ''}]
        with patch('utils.harmful_content_utils.torch', torch):
            result = classify_completions(model, tokenizer, records)
        self.assertEqual([row['is_harmful'] for row in result], [1, 0])
        self.assertAlmostEqual(result[0]['harmful_probability'], 0.8)
        self.assertAlmostEqual(result[1]['harmful_probability'], 0.1)
        self.assertFalse(tokenizer.call_args.kwargs['truncation'])
        self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
        kwargs = model.generate.call_args.kwargs
        self.assertTrue(kwargs['output_logits'])
        self.assertEqual(kwargs['max_new_tokens'], 1)
        self.assertFalse(kwargs['do_sample'])
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[-1], {'role': 'assistant', 'content': ''})

    def test_invalid_label_tokenization_rejected(self):
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = [[1], [2, 3]]
        with self.assertRaisesRegex(ValueError, 'single tokens'):
            get_label_token_ids(tokenizer)

    def test_overlong_guard_input_fails_before_generation(self):
        model, tokenizer = MagicMock(), MagicMock()
        tokenizer.encode.side_effect = [[1], [3]]
        tokenizer.return_value = {'input_ids': Tensor([[1, 2, 3]])}
        model.config.max_position_embeddings = 3
        with self.assertRaisesRegex(ValueError, 'context window'):
            classify_completions(model, tokenizer, [{'prompt': 'p', 'completion': 'c'}])
        model.generate.assert_not_called()


class EvaluationFileTests(unittest.TestCase):
    def make_files(self, directory):
        folder = Path(directory) / 'qwen' / 'int4'
        folder.mkdir(parents=True)
        records = [
            {'model': 'qwen', 'quantization': 'int4', 'prompt': 'duplicate', 'completion': 'answer',
             'row_index': 0, 'source': {'label': 'harmful'}, 'other': [1, 2]},
            {'model': 'qwen', 'quantization': 'int4', 'prompt': 'duplicate', 'completion': '',
             'row_index': 1, 'source': {'label': 'benign'}, 'other': [3]},
        ]
        path = folder / 'harmful_seed123.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        metadata = folder / (path.name + '.metadata.json')
        metadata.write_text('{"status": "completed"}')
        return path, metadata, records

    def test_main_preserves_fields_and_duplicates_updates_original_and_logs_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, records = self.make_files(directory)
            model, tokenizer = MagicMock(), MagicMock()
            model.config._commit_hash = 'guard-commit'
            def classify(model, tokenizer, batch, **kwargs):
                return [{'is_harmful': int(row['row_index'] == 0),
                         'harmful_probability': 0.9 if row['row_index'] == 0 else 0.1} for row in batch]
            with patch('evaluation.harmful_content_evaluation.OUTPUT_ROOT', Path(directory)), \
                 patch('evaluation.harmful_content_evaluation.set_seed'), \
                 patch('evaluation.harmful_content_evaluation.load_harm_evaluator',
                       return_value=(model, tokenizer)) as loader, \
                 patch('evaluation.harmful_content_evaluation.classify_completions', side_effect=classify), \
                 self.assertLogs('evaluation.harmful_content_evaluation', level='INFO') as logs:
                result = main('qwen', 'int4', batch_size=1)
            loader.assert_called_once()
            self.assertEqual(result, [path.resolve()])
            annotated = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(annotated), len(records))
            for before, after in zip(records, annotated):
                for key, value in before.items():
                    self.assertEqual(after[key], value)
                self.assertEqual(after['harm_evaluation']['model_commit'], 'guard-commit')
            self.assertEqual([row['is_harmful'] for row in annotated], [1, 0])
            self.assertTrue(any('50.00%' in line for line in logs.output))
            self.assertEqual(metadata.read_text(), '{"status": "completed"}')

    def test_load_guard_once_for_multiple_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.make_files(directory)
            second = path.with_name('harmless_seed123.jsonl')
            second.write_bytes(path.read_bytes())
            model = MagicMock()
            model.config._commit_hash = 'commit'
            with patch('evaluation.harmful_content_evaluation.OUTPUT_ROOT', Path(directory)), \
                 patch('evaluation.harmful_content_evaluation.set_seed'), \
                 patch('evaluation.harmful_content_evaluation.load_harm_evaluator',
                       return_value=(model, MagicMock())) as loader, \
                 patch('evaluation.harmful_content_evaluation.classify_completions',
                       return_value=[{'is_harmful': 0, 'harmful_probability': 0.1}] * 2):
                paths = main('qwen', 'int4')
            self.assertEqual(len(paths), 2)
            loader.assert_called_once()

    def test_failed_batch_leaves_original_file_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, records = self.make_files(directory)
            original = path.read_bytes()
            with patch('evaluation.harmful_content_evaluation.classify_completions',
                       side_effect=[[{'is_harmful': 1, 'harmful_probability': 0.8}],
                                    RuntimeError('GPU failed')]):
                with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                    evaluate_file(path, records, MagicMock(), MagicMock(), batch_size=1,
                                  threshold=0.5, provenance={})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_preflight_rejects_malformed_file_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.make_files(directory)
            path.write_text('{"prompt": "no completion"}\n')
            with patch('evaluation.harmful_content_evaluation.OUTPUT_ROOT', Path(directory)), \
                 patch('evaluation.harmful_content_evaluation.load_harm_evaluator') as loader:
                with self.assertRaisesRegex(ValueError, 'completion must be a string'):
                    main('qwen', 'int4')
                loader.assert_not_called()

    def test_discovery_selects_one_file_and_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.make_files(directory)
            with patch('evaluation.harmful_content_evaluation.OUTPUT_ROOT', Path(directory)):
                self.assertEqual(find_completion_files('qwen', 'int4', filename=path.name), [path.resolve()])
                with self.assertRaises(ValueError):
                    find_completion_files('qwen', 'int4', filename='../escape.jsonl')
                with self.assertRaises(ValueError):
                    find_completion_files('../qwen', 'int4')
                with self.assertRaises(FileNotFoundError):
                    find_completion_files('qwen', 'bf16')


if __name__ == '__main__':
    unittest.main()
