"""Audit real short ns-3 tests without treating them as performance evidence."""
import json
from pathlib import Path
import statistics
from campaign import ROOT, atomic, now, verify


def main():
    result = {'checked_utc': now(), 'purpose': 'functional_integration_not_900s_performance', 'runs': []}
    for name in ('smoke_eval_none', 'smoke_train_both', 'smoke_eval_both', 'smoke_resume_both'):
        directory = ROOT / 'results' / name
        execution = json.loads((directory / 'execution.json').read_text())
        audit = verify(directory, 25)
        if execution['status'] != 'complete' or not audit['passed']:
            raise RuntimeError(f'Smoke verification failed: {name}: {audit}')
        summary = json.loads((directory / 'control/summary.json').read_text())
        events = [json.loads(s) for s in (directory / 'control/actions.jsonl').read_text().splitlines()]
        receipts = [json.loads(s) for s in (directory / 'control/receipts.jsonl').read_text().splitlines()]
        records = [r for r in receipts if r['status'] != 'terminal']
        changes = sum(r['any_parameter_changed'] for r in records)
        gradient_delta = {a: summary['final_agents'][a]['gradient_updates_total'] - initial['gradient_updates_total']
                          for a, initial in summary['initial_agents'].items()}
        if name == 'smoke_eval_both':
            assert all(v == 0 for v in gradient_delta.values())
            assert all(not app.get('explored', False) for e in events for app in e['apps'].values())
            assert all(not a['learning_enabled'] and a['epsilon'] == 0 for a in summary['final_agents'].values())
        elif name in ('smoke_train_both', 'smoke_resume_both'):
            assert all(v > 0 for v in gradient_delta.values())
            assert all(v == 24 for v in summary['episode_transitions'].values())
        if name == 'smoke_resume_both':
            assert all(v['transitions_total'] == 24 for v in summary['initial_agents'].values())
            assert all(v['transitions_total'] == 48 for v in summary['final_agents'].values())
        inference = [app['decision_ms'] for e in events for app in e['apps'].values() if 'decision_ms' in app]
        result['runs'].append(dict(name=name, passed=True, duration_s=25, cell_rows=audit['cell_rows'],
                                  wall_seconds=execution['wall_seconds'], application_receipts=len(records),
                                  changed_decisions=changes, gradient_update_delta=gradient_delta,
                                  decision_ms_median=statistics.median(inference) if inference else None,
                                  per_app=summary['final_agents'], source_directory=str(directory)))
    result['all_passed'] = True
    atomic(ROOT / 'smoke_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
