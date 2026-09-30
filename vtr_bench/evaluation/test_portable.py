"""Offline integration checks. The fake backend does not perform model inference."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate as app
from summarize import summarize_run


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prompts = self.root / 'prompts.json'
        self.checks = self.root / 'checklists.json'
        self.case = {'id': 'AD-0001', 'prompt_en': 'A card displays "SECRET REFERENCE".',
                     'required_text': [{'id': 'T01', 'carrier': 'card', 'language': 'English',
                                        'verbatim': 'SECRET REFERENCE'}]}
        self.check = {'id': 'AD-0001', 'checklist': [
            {'id': f'C{i:02}', 'dimension': 'Scene Attributes', 'question_en': f'Condition {i}?'}
            for i in range(1, 21)]}
        app.write_json(self.prompts, {'cases': [self.case]})
        app.write_json(self.checks, {'cases': [self.check]})
        self.video = self.root / 'videos'
        self.video.mkdir()
        (self.video / 'AD-0001.mp4').write_bytes(b'fake video for backend contract tests only')
        self.model = self.root / 'model'
        self.model.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_data(self):
        cases, checks = app.load_inputs(app.DATA / 'prompts.json', app.DATA / 'checklists.json')
        self.assertEqual(len(cases), 300)
        self.assertEqual(sum(len(c['required_text']) for c in cases.values()), 1202)
        self.assertEqual(sum(len(c['checklist']) for c in checks.values()), 6000)

    def test_minimal_interface_and_no_reference_leak(self):
        cases, checks = app.load_inputs(self.prompts, self.checks)
        masked, public, private = app.common.mask_scene_prompt(cases['AD-0001'])
        self.assertEqual(private[0]['verbatim'], 'SECRET REFERENCE')
        request = app.wer.user_prompt(masked, public)
        self.assertNotIn('SECRET REFERENCE', request)
        self.assertIn('[TARGET T01]', request)
        self.assertNotIn('Scene Attributes', app.checklist.MODULE.user_prompt(checks['AD-0001']['checklist']))

    def test_duplicate_ids_rejected(self):
        app.write_json(self.prompts, {'cases': [self.case, self.case]})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            app.load_inputs(self.prompts, self.checks)

    def test_unknown_dimension_rejected(self):
        self.check['checklist'][0]['dimension'] = 'not-a-dimension'
        app.write_json(self.checks, {'cases': [self.check]})
        with self.assertRaisesRegex(ValueError, 'dimension'):
            app.load_inputs(self.prompts, self.checks)

    def test_id_alignment_rejected(self):
        self.check['id'] = 'AD-0002'
        app.write_json(self.checks, {'cases': [self.check]})
        with self.assertRaisesRegex(ValueError, 'match exactly'):
            app.load_inputs(self.prompts, self.checks)

    def test_video_mapping_not_directory_order(self):
        (self.video / 'AD-0002.mp4').write_bytes(b'two')
        videos, missing = app.select_videos(self.video, {'AD-0002': {}, 'AD-0001': {}})
        self.assertEqual(list(videos), ['AD-0001', 'AD-0002'])
        self.assertEqual(videos['AD-0002'].read_bytes(), b'two')
        self.assertEqual(missing, [])

    def test_missing_and_unknown_videos(self):
        cases = {'AD-0001': {}, 'AD-0002': {}}
        with self.assertRaisesRegex(ValueError, 'Missing'):
            app.select_videos(self.video, cases)
        _, missing = app.select_videos(self.video, cases, allow_missing=True)
        self.assertEqual(missing, ['AD-0002'])
        (self.video / 'wrong_name.mp4').write_bytes(b'unknown')
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            app.select_videos(self.video, cases, allow_missing=True)

    def test_single_video_path_ignores_unrelated_neighbor_files(self):
        (self.video / 'unrelated.mp4').write_bytes(b'unrelated')
        videos, missing = app.select_videos(
            self.video / 'AD-0001.mp4', {'AD-0001': {}, 'AD-0002': {}})
        self.assertEqual(list(videos), ['AD-0001'])
        self.assertEqual(missing, [])

    def run_mock(self, task, responses, output, resume=False):
        calls = []
        constructors = []

        class Sampling:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                assert 'seed' not in kwargs and 'stop' not in kwargs
                assert 'structured_outputs' not in kwargs

        class LLM:
            def __init__(self, **kwargs):
                constructors.append(kwargs)

            def chat(self, messages, **kwargs):
                config = kwargs['sampling_params'].__dict__.copy()
                calls.append({'messages': messages, 'config': config, 'kwargs': kwargs})
                raw, finish = responses[len(calls) - 1]
                result = SimpleNamespace(text=raw, finish_reason=finish,
                    token_ids=list(range(config['max_tokens'])) if finish == 'length' else [1, 2])
                return [SimpleNamespace(outputs=[result], prompt_token_ids=[1, 2, 3])]

        fake_vllm = SimpleNamespace(LLM=LLM, SamplingParams=Sampling, __version__='offline-fake')
        fake_torch = SimpleNamespace(__version__='offline-fake', version=SimpleNamespace(cuda=None))
        args = ['--task', task, '--prompts', str(self.prompts), '--checklists', str(self.checks),
                '--video-root', str(self.video), '--profile', 'test-model', '--model-path', str(self.model),
                '--output-root', str(output)] + (['--resume'] if resume else [])
        with patch.dict(sys.modules, {'vllm': fake_vllm, 'torch': fake_torch}), \
             patch('importlib.metadata.version', return_value='offline-fake'), \
             patch.object(app.common, 'video_metadata', return_value={'test': True}), \
             patch.object(app.checklist.MODULE, 'video_metadata', return_value={'test': True}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = app.main(args)
        self.assertEqual(code, 0)
        for call in calls:
            self.assertEqual(call['kwargs']['chat_template_kwargs'], {'enable_thinking': False})
            for key, value in {'temperature': .7, 'top_p': .8, 'top_k': 20, 'min_p': 0,
                               'presence_penalty': 1.5, 'repetition_penalty': 1., 'frequency_penalty': 0.}.items():
                self.assertEqual(call['config'][key], value)
        self.assertEqual(constructors[0]['tensor_parallel_size'], 1)
        self.assertEqual(constructors[0]['media_io_kwargs'], {'video': {'fps': 2}})
        return calls, app.read_json(output / 'reports/metrics.json')

    def test_checklist_retry_escalation_and_resume(self):
        answer = [{'id': f'C{i:02}' if i < 10 else f'C0{i}', 'answer': 'Yes' if i % 2 else 'NO'}
                  for i in range(1, 21)]
        raw = '<answer>' + json.dumps(answer) + '</answer><answer>[]</answer>'
        out = self.root / 'checklist-run'
        calls, report = self.run_mock('checklist', [('<answer>[', 'length'), ('<answer>[', 'length'), (raw, 'stop')], out)
        self.assertEqual([c['config']['max_tokens'] for c in calls], [1024, 2048, 4096])
        self.assertIn('PREVIOUS ATTEMPT FEEDBACK', calls[1]['messages'][-1]['content'][-1]['text'])
        self.assertEqual(report['overall']['yes_rate'], .5)
        dims = report['overall']['dimensions']
        self.assertEqual(dims['Scene Attributes']['item_count'], 20)
        self.assertIsNone(dims['Entity Presence']['yes_rate'])
        calls, _ = self.run_mock('checklist', [], out, resume=True)
        self.assertEqual(calls, [])

    def test_wer_retry_terminal_acceptance_and_resume(self):
        answer = {'T01': {'status': 'transcribed', 'text': 'x' * 641, 'readability_note': ''}}
        raw = '<answer>' + json.dumps(answer) + '</answer>trailing text'
        out = self.root / 'wer-run'
        calls, report = self.run_mock('wer', [(raw, 'stop')] * 3, out)
        self.assertEqual([c['config']['max_tokens'] for c in calls], [2048] * 3)
        self.assertEqual(report['status_counts'], {'pass_after_max_attempts': 1})
        self.assertTrue(report['all_scored'])
        for call in calls:
            self.assertNotIn('SECRET REFERENCE', json.dumps(call['messages']))
        self.assertEqual(set(report['overall']['case_balanced_wer']), {'R+1', 'R+5', 'R+10'})
        calls, _ = self.run_mock('wer', [], out, resume=True)
        self.assertEqual(calls, [])

    def test_checklist_partial_resume_keeps_truncation_budget(self):
        out = self.root / 'interrupted-run'
        # The second call fails at the fake backend after round 1 was persisted.
        with self.assertRaises(IndexError):
            self.run_mock('checklist', [('<answer>[', 'length')], out)
        raw = '<answer>' + json.dumps([{'id': f'C{i:02}', 'answer': 'yes'} for i in range(1, 21)]) + '</answer>'
        calls, report = self.run_mock('checklist', [(raw, 'stop')], out, resume=True)
        self.assertEqual([c['config']['max_tokens'] for c in calls], [2048])
        self.assertEqual(report['overall']['yes_rate'], 1.0)

    def test_unparseable_is_not_scored(self):
        calls, report = self.run_mock('wer', [('<answer>{', 'length')] * 3, self.root / 'bad-run')
        self.assertEqual(report['status_counts'], {'invalid': 1})
        self.assertFalse(report['all_scored'])
        self.assertEqual(report['overall']['scored_videos'], 0)
        self.assertIsNone(report['overall']['case_balanced_wer']['R+5'])

    def test_case_balanced_not_text_balanced(self):
        out = self.root / 'aggregation'
        app.write_json(out / 'run_manifest.json', {'job_identities': [
            {'profile': 'test', 'case_id': c} for c in ('AD-0001', 'AD-0002')]})
        for cid, items in [('AD-0001', [{'reference': 'a', 'hypothesis': ''}]),
                           ('AD-0002', [{'reference': 'a b c d e f g h', 'hypothesis': 'a b c d e f g h'},
                                        {'reference': 'one two', 'hypothesis': 'one two'}])]:
            app.write_json(out / 'test' / cid / 'case_status.json', {'status': 'pass'})
            app.write_json(out / 'test' / cid / 'score.json', {'profile': 'test', 'case_id': cid, 'items': items})
        report = summarize_run(out, 'wer')
        self.assertEqual(report['overall']['case_balanced_wer']['R+5'], .5)


if __name__ == '__main__':
    unittest.main()
