"""Click help, validation, and dispatch tests; no inference is performed."""
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from evaluation.harmful_content_evaluation import cli as evaluation_cli
from experiment_1.run_inference import main as inference_cli


class CommandLineTests(unittest.TestCase):
    def test_help_shows_choices_defaults_and_formats(self):
        runner = CliRunner()
        for command in (inference_cli, evaluation_cli):
            with self.subTest(command=command.name):
                result = runner.invoke(command, ['--help'], terminal_width=120)
                self.assertEqual(result.exit_code, 0, result.output)
                for value in ('bf16', 'int8', 'int4', 'llama2', 'qwen', 'vicuna', 'cuda:N', 'default:'):
                    self.assertIn(value, result.output)
                self.assertEqual(runner.invoke(command, ['-h']).exit_code, 0)
        help_text = runner.invoke(inference_cli, ['--help']).output
        for value in ('harmless', 'adversarial', 'harmful', 'harmless_with_sensitive_words',
                      '--do-sample', '--no-do-sample', 'Filename only'):
            self.assertIn(value, help_text)
        for value in ('auto', 'bfloat16', 'float16', 'float32'):
            self.assertIn(value, runner.invoke(evaluation_cli, ['--help']).output)

    def test_inference_sampling_flags_preserve_three_states(self):
        runner = CliRunner()
        for flags, expected in (([], None), (['--do-sample'], True), (['--no-do-sample'], False)):
            with patch('experiment_1.run_inference.run') as run:
                result = runner.invoke(inference_cli, ['--model', 'llama2', '--dataset', 'harmful', *flags])
                self.assertEqual(result.exit_code, 0, result.output)
                args = run.call_args.args[0]
                self.assertIs(args.do_sample, expected)
                self.assertEqual(args.quantization, 'bf16')
                self.assertIsNone(args.output)

    def test_invalid_inference_values_rejected_before_loading(self):
        invalid = [('--quantization', 'fp8'), ('--batch-size', '0'), ('--max-new-tokens', '-1'),
                   ('--max-input-tokens', '0'), ('--limit', '0'), ('--offset', '-1'),
                   ('--seed', '-1'), ('--device', 'cuda:-1')]
        for flag, value in invalid:
            with self.subTest(flag=flag), patch('experiment_1.run_inference.run') as run:
                result = CliRunner().invoke(inference_cli,
                                            ['--model', 'llama2', '--dataset', 'harmful', flag, value])
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn(flag, result.output)
                run.assert_not_called()

    def test_evaluation_dispatch_preserves_python_main_api(self):
        with patch('evaluation.harmful_content_evaluation.main') as main:
            result = CliRunner().invoke(evaluation_cli, [
                '--model', 'llama2', '--quantization', 'int4', '--file', 'harmful_seed123.jsonl',
                '--guard-model', 'custom/guard', '--guard-revision', 'commit', '--dtype', 'float32',
                '--threshold', '0.7', '--batch-size', '2', '--device', 'cuda:1',
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        main.assert_called_once_with(model_name='llama2', quantization='int4',
                                     filename='harmful_seed123.jsonl', guard_model_id='custom/guard',
                                     revision='commit', dtype='float32', threshold=0.7,
                                     batch_size=2, device='cuda:1', seed=123)

    def test_invalid_guard_options_rejected_before_loading(self):
        for flag, value in (('--threshold', '0'), ('--threshold', '1'), ('--dtype', 'int8'),
                            ('--batch-size', '0'), ('--seed', '-1'), ('--device', 'gpu')):
            with self.subTest(flag=flag), patch('evaluation.harmful_content_evaluation.main') as main:
                result = CliRunner().invoke(evaluation_cli,
                                            ['--model', 'llama2', '--quantization', 'bf16', flag, value])
                self.assertEqual(result.exit_code, 2, result.output)
                main.assert_not_called()

    def test_command_errors_have_readable_messages(self):
        for command, target, args in (
            (inference_cli, 'experiment_1.run_inference.run', ['--model', 'llama2', '--dataset', 'harmful']),
            (evaluation_cli, 'evaluation.harmful_content_evaluation.main',
             ['--model', 'llama2', '--quantization', 'bf16']),
        ):
            with patch(target, side_effect=ValueError('Example validation error')):
                result = CliRunner().invoke(command, args)
                self.assertEqual(result.exit_code, 1)
                self.assertIn('Error: Example validation error', result.output)


if __name__ == '__main__':
    unittest.main()
