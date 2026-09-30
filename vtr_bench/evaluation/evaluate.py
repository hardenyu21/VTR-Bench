#!/usr/bin/env python3
"""Portable public-ID adapter for the frozen Checklist and VLM-WER engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parent / 'data'
sys.path.insert(0, str(PACKAGE / 'engine'))
import evaluate_vlm_wer as common
import evaluate_vlm_wer_retry as wer
import evaluate_checklist_retry_v2_compat as checklist
from vtextbench_tokenizer_v2 import wer_tokens

DIMENSIONS = ('Entity Presence', 'Spatial Relationship', 'Temporal Consistency',
              'Motion Adherence', 'Scene Attributes')
ORIGINAL_RESULT_BUILDER = checklist.MODULE.result_from_decisions


def checklist_result(job, decisions):
    result = ORIGINAL_RESULT_BUILDER(job, decisions)
    for item in result['items']:
        item['dimension'] = item.pop('category')
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_inputs(prompts_path, checklists_path):
    payload = read_json(prompts_path)
    cases = payload.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('prompts.json must contain a nonempty cases array')
    by_id = {}
    for case in cases:
        case_id = case.get('id')
        if not isinstance(case_id, str) or not re.fullmatch(r'(AD|SCI|UI|CULT|LIFE)-\d{4}', case_id):
            raise ValueError(f'Invalid public id: {case_id!r}')
        if case_id in by_id:
            raise ValueError(f'Duplicate public id: {case_id}')
        if not isinstance(case.get('prompt_en'), str) or not case['prompt_en'].strip():
            raise ValueError(f'{case_id}: missing prompt_en')
        targets = case.get('required_text')
        if not isinstance(targets, list) or not targets:
            raise ValueError(f'{case_id}: missing required_text')
        ids = set()
        for target in targets:
            for key in ('id', 'carrier', 'language', 'verbatim'):
                if not isinstance(target.get(key), str) or not target[key].strip():
                    raise ValueError(f'{case_id}: missing or empty target {key}')
            if not re.fullmatch(r'T\d{2,}', target['id']) or target['id'] in ids:
                raise ValueError(f'{case_id}: invalid or duplicate target ID')
            ids.add(target['id'])
            if not wer_tokens(target['verbatim']):
                raise ValueError(f'{case_id}:{target["id"]}: empty reference token sequence')
        masked, public, private = common.mask_scene_prompt(case)
        model_input = wer.SYSTEM_PROMPT + wer.user_prompt(masked, public)
        if any(t['verbatim'] in model_input for t in private):
            raise ValueError(f'{case_id}: reference text would leak into the request')
        by_id[case_id] = case
    raw_checks = read_json(checklists_path).get('cases')
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError('checklists.json must contain a nonempty cases array')
    checks = {}
    for case in raw_checks:
        case_id = case.get('id')
        if not isinstance(case_id, str) or case_id in checks:
            raise ValueError(f'Invalid or duplicate checklist case ID: {case_id!r}')
        items = case.get('checklist')
        if not isinstance(items, list) or [v.get('id') for v in items] != [f'C{i:02}' for i in range(1, 21)]:
            raise ValueError(f'{case_id}: expected exactly C01-C20 in order')
        for item in items:
            if item.get('dimension') not in DIMENSIONS:
                raise ValueError(f'{case_id}:{item["id"]}: unknown dimension')
            if not isinstance(item.get('question_en'), str) or not item['question_en'].strip():
                raise ValueError(f'{case_id}:{item["id"]}: missing question_en')
        # Compatibility fields exist only inside the frozen engine; no old
        # category taxonomy or source IDs are required by the public interface.
        checks[case_id] = {'case_id': case_id, 'family': case_id.split('-')[0],
                          'checklist': [{**item, 'category': item['dimension'], 'expected_answer': 'yes'}
                                        for item in items]}
    if set(checks) != set(by_id):
        raise ValueError('Prompt and checklist public IDs must match exactly')
    return by_id, checks


def select_videos(root, cases, selected_ids=None, allow_missing=False):
    root = Path(root).expanduser().resolve()
    single = root.is_file() and root.suffix.lower() == '.mp4'
    if not root.is_dir() and not single:
        raise FileNotFoundError(root)
    requested = set(selected_ids or ([root.stem] if single else cases))
    if not requested <= cases.keys():
        raise ValueError(f'Unknown requested IDs: {sorted(requested - cases.keys())}')
    videos = {}
    unknown = []
    for path in ([root] if single else sorted(root.iterdir())):
        if path.is_file() and path.suffix.lower() == '.mp4':
            if path.stem not in cases:
                unknown.append(path.name)
            elif path.stem in videos:
                raise ValueError(f'Duplicate video ID: {path.stem}')
            else:
                resolved = path.resolve()
                if not single and not resolved.is_relative_to(root):
                    raise ValueError(f'{path.name}: symlink target is outside video-root')
                videos[path.stem] = resolved
    if unknown:
        raise ValueError(f'Unmapped MP4 filenames (rename explicitly to <id>.mp4): {unknown}')
    missing = sorted(requested - videos.keys())
    if missing and not allow_missing:
        raise ValueError(f'Missing videos: {missing}; use --allow-missing only for intentional omissions')
    selected = {key: videos[key] for key in sorted(requested & videos.keys())}
    if not selected:
        raise ValueError('No selected videos')
    return selected, missing


def sampling_arguments(task, gpu_memory):
    return ['--tensor-parallel-size', '1', '--gpu-memory-utilization', str(gpu_memory),
            '--max-model-len', '32768', '--max-tokens', '1024' if task == 'checklist' else '2048',
            '--video-fps', '2', '--temperature', '0.7', '--top-p', '0.8', '--top-k', '20',
            '--min-p', '0.0', '--presence-penalty', '1.5', '--repetition-penalty', '1.0',
            '--frequency-penalty', '0.0']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['checklist', 'wer'])
    p.add_argument('--prompts', type=Path, default=DATA / 'prompts.json')
    p.add_argument('--checklists', type=Path, default=DATA / 'checklists.json')
    p.add_argument('--video-root', type=Path, help='Flat directory of public-ID.mp4 files')
    p.add_argument('--profile', default='video-model')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--case-id', action='append', help='Repeat to select a subset')
    p.add_argument('--allow-missing', action='store_true')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='Validate all data without loading a model')
    p.add_argument('--dry-run', action='store_true', help='Validate data and video mapping, without inference')
    p.add_argument('--show-prompt', metavar='CASE_ID', help='Print exact first-attempt system/user texts')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases, checks = load_inputs(args.prompts, args.checklists)
    if args.show_prompt:
        c = cases[args.show_prompt]
        masked, public, _ = common.mask_scene_prompt(c)
        print(json.dumps({'checklist': {'system': checklist.MODULE.SYSTEM_PROMPT,
              'user': checklist.MODULE.user_prompt(checks[c['id']]['checklist'])},
              'wer': {'system': wer.SYSTEM_PROMPT, 'user': wer.user_prompt(masked, public)}},
              ensure_ascii=False, indent=2))
        return 0
    if args.validate_only:
        print(json.dumps({'status': 'pass', 'cases': len(cases),
              'checklist_items': sum(len(c['checklist']) for c in checks.values()),
              'text_blocks': sum(len(c['required_text']) for c in cases.values()),
              'reference_masking_and_no_leak_check': 'pass'}, indent=2))
        return 0
    if not args.video_root:
        raise ValueError('--video-root is required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.profile):
        raise ValueError('Profile must be a simple filename-safe identifier')
    videos, missing = select_videos(args.video_root, cases, args.case_id, args.allow_missing)
    if args.dry_run:
        print(json.dumps({'selected_count': len(videos), 'missing_count': len(missing),
              'missing_ids': missing, 'ids': list(videos)}, indent=2))
        return 0
    if not args.task or not args.model_path or not args.output_root:
        raise ValueError('Inference requires --task, --model-path and --output-root')
    model = args.model_path.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError('GPU memory utilization must be between 0 and 1')
    output = args.output_root.expanduser().resolve()
    identity = {'adapter_version': 'public_id_v1', 'task': args.task, 'profile': args.profile,
                'prompts_sha256': common.sha256_file(args.prompts),
                'checklists_sha256': common.sha256_file(args.checklists),
                'missing_ids': missing,
                'videos': [{'id': k, 'path': str(v), 'size': v.stat().st_size,
                            'mtime_ns': v.stat().st_mtime_ns} for k, v in videos.items()]}
    digest = common.sha256_json(identity)
    prepared = output.parent / '.prepared_inputs' / digest
    # Kept outside the engine's output root, which must be empty for a fresh run.
    write_json(prepared / 'manifest.json', identity)
    for case_id in videos:
        write_json(prepared / f'{case_id}.json', cases[case_id])
    source = args.video_root.expanduser().resolve()
    common.PROJECT = source.parent if source.is_file() else source
    official = sampling_arguments(args.task, args.gpu_memory_utilization)
    base = ['--model-path', str(model), '--output-root', str(output),
            '--run-id', f'{args.task}_{args.profile}_{digest[:16]}', *official]
    if args.resume:
        base.append('--resume')
    previous_argv = sys.argv
    try:
        if args.task == 'checklist':
            jobs = [{'profile': args.profile, 'case_id': k, 'video': v, 'case': checks[k]}
                    for k, v in videos.items()]
            checklist.MODULE.build_jobs = lambda *a, **kw: jobs
            checklist.MODULE.load_checklists = lambda path: ({'suite': 'VTextBench',
                'case_count': len(checks), 'checklist_count_per_case': 20}, checks)
            checklist.MODULE.result_from_decisions = checklist_result
            roots = {v.as_uri(): output / args.profile / k for k, v in videos.items()}
            checklist.case_root_for_messages = lambda messages, root: roots.get(checklist.video_key(messages))
            sys.argv = [str(Path(__file__).resolve()), *base, '--project-root', str(common.PROJECT),
                        '--model-name', model.name, '--checklists-json', str(args.checklists.resolve()),
                        '--profile-video-root', f'{args.profile}={common.PROJECT}',
                        '--expected-jobs', str(len(jobs)), '--max-rounds', '3']
            checklist.install_adaptive_chat()
            code = checklist.MODULE.main()
        else:
            jobs = [{'profile': args.profile, 'case_id': k, 'layer': k.split('-')[0],
                     'video': v, 'prompt': prepared / f'{k}.json'} for k, v in videos.items()]
            common.build_jobs = lambda *a, **kw: jobs
            sys.argv = [str(Path(__file__).resolve()), *base, '--mode', 'full',
                        '--recipe-name', 'official_instruct_noseed',
                        '--sample-manifest', str(prepared / 'manifest.json'),
                        '--max-attempts-per-case', '3']
            code = wer.main()
    finally:
        sys.argv = previous_argv
    write_json(output / 'interface_manifest.json', identity)
    from summarize import summarize_run
    report = summarize_run(output, args.task, list(videos), missing)
    write_json(output / 'reports/metrics.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
