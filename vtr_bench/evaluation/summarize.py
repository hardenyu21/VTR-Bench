#!/usr/bin/env python3
"""Report current Checklist and contextual-tokenizer R+N metrics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engine'))
from vtextbench_wer_v2 import score_case, SCORING_VERSION
from vtextbench_tokenizer_v2 import TOKENIZER_VERSION

DIMENSIONS = ('Entity Presence', 'Spatial Relationship', 'Temporal Consistency',
              'Motion Adherence', 'Scene Attributes')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize_run(root, task, expected_ids=None, missing_ids=None, ns=(1, 5, 10)):
    root = Path(root)
    manifest = load(root / 'run_manifest.json')
    expected_keys = {(v['profile'], v['case_id']) for v in manifest['job_identities']}
    if expected_ids is not None and {key[1] for key in expected_keys} != set(expected_ids):
        raise ValueError('Requested IDs disagree with run manifest')
    rows = []
    statuses = Counter()
    unscored = []
    for profile, case_id in sorted(expected_keys):
        folder = root / profile / case_id
        state = load(folder / 'case_status.json') if (folder / 'case_status.json').exists() else {}
        status = state.get('status', 'pending')
        statuses[status] += 1
        file = folder / ('result.json' if task == 'checklist' else 'score.json')
        allowed = {'pass'} if task == 'checklist' else {'pass', 'pass_after_max_attempts'}
        if status not in allowed or not file.exists():
            unscored.append({'profile': profile, 'case_id': case_id, 'status': status})
            continue
        result = load(file)
        if result['profile'] != profile or result['case_id'] != case_id:
            raise ValueError(f'{file}: result identity mismatch')
        row = {'profile': profile, 'case_id': case_id, 'scene': case_id.split('-')[0]}
        if task == 'checklist':
            if [x['id'] for x in result['items']] != [f'C{i:02}' for i in range(1, 21)]:
                raise ValueError(f'{file}: expected exactly C01-C20')
            dims = defaultdict(Counter)
            for item in result['items']:
                if item['answer'] not in ('yes', 'no'):
                    raise ValueError(f'{file}: unexpected label')
                d = item.get('dimension') or item.get('category')
                if d not in DIMENSIONS:
                    raise ValueError(f'{file}: unknown dimension {d}')
                dims[d]['total'] += 1
                dims[d]['yes'] += item['answer'] == 'yes'
            row.update(yes=sum(v['yes'] for v in dims.values()), total=20, dimensions=dict(dims))
        else:
            # Use raw strings, NOT the old pre-truncated hypothesis_tokens field.
            row['wer'] = {str(n): score_case(result['items'], extra_tokens=n)['summary'] for n in ns}
        rows.append(row)

    def aggregate(group):
        out = {'scored_videos': len(group)}
        if task == 'checklist':
            yes = sum(r['yes'] for r in group)
            total = sum(r['total'] for r in group)
            out.update(yes_count=yes, item_count=total, yes_rate=yes / total if total else None)
            out['dimensions'] = {}
            for d in DIMENSIONS:
                values = [r['dimensions'][d] for r in group if d in r['dimensions']]
                dy = sum(v['yes'] for v in values)
                dt = sum(v['total'] for v in values)
                out['dimensions'][d] = {'yes_count': dy, 'item_count': dt,
                    'applicable_videos': len(values), 'yes_rate': dy / dt if dt else None}
        else:
            out['case_balanced_wer'] = {f'R+{n}': sum(r['wer'][str(n)]['wer'] for r in group) / len(group)
                                        if group else None for n in ns}
        return out

    grouped = {}
    for field in ('profile', 'scene'):
        grouped[f'by_{field}'] = {key: aggregate([r for r in rows if r[field] == key])
                                 for key in sorted({r[field] for r in rows})}
    return {'task': task, 'expected_videos': len(expected_keys), 'status_counts': dict(statuses),
            'all_scored': len(rows) == len(expected_keys), 'unscored_cases': unscored,
            'intentionally_missing_video_ids': missing_ids or [],
            'tokenizer_version': TOKENIZER_VERSION if task == 'wer' else None,
            'scoring_version': SCORING_VERSION if task == 'wer' else None,
            'overall': aggregate(rows), **grouped, 'cases': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--task', choices=['checklist', 'wer'], required=True)
    p.add_argument('--n', type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    interface = args.run_root / 'interface_manifest.json'
    missing = load(interface).get('missing_ids', []) if interface.exists() else []
    result = summarize_run(args.run_root, args.task, missing_ids=missing, ns=args.n)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
