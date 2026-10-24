"""Audit frozen offline policies using only the original train partition.

No ns-3 runs, learning, checkpoint changes, test-data reads, or policy selection
occur here. All policy outcomes below are synthetic response-model diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import (PARAMETERS, STATE_DIM, action_catalog, apply_actions,
                     metrics, states, valid_mask)
from .models import QNetwork, ResponseEnsemble, ResponseNetwork
from .train import contexts, transition

ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0., .01, .05, .5, .95, .99, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite audit metric')
    return {'mean': float(values.mean()), 'quantiles': {
        str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))}}


def describe(m):
    attempts, successes = m['ho_attempts'], m['ho_successes']
    return {
        'ho_attempts_per_30s': distribution(attempts),
        'ho_successes_per_30s': distribution(successes),
        'contexts_with_attempts_below_1': int((attempts < 1).sum()),
        'contexts_with_attempts_below_10': int((attempts < 10).sum()),
        'contexts_with_successes_below_10': int((successes < 10).sum()),
        'mean_jain': float(m['jain'].mean()),
        'mean_rlf_per_30s': float(m['rlf'].mean()),
        'mean_delivery_ratio': float(m['delivery_ratio'].mean()),
        'pp_ratio_of_sums': float(m['pingpongs'].sum() / max(attempts.sum(), 1e-12)),
        'ho_success_ratio_of_sums': float(successes.sum() / max(attempts.sum(), 1e-12)),
    }


def compare(hold, policy):
    supported = hold['ho_successes'] >= 10
    retention = policy['ho_successes'][supported] / hold['ho_successes'][supported]
    return {
        'reference_contexts_with_at_least_10_successful_ho': int(supported.sum()),
        'success_retention_ratio_of_sums': float(policy['ho_successes'].sum() /
                                               max(hold['ho_successes'].sum(), 1e-12)),
        'supported_context_success_retention': distribution(retention) if len(retention) else None,
        'supported_contexts_below_50pct_success_retention': int((retention < .5).sum()),
        'supported_contexts_below_20pct_success_retention': int((retention < .2).sum()),
        'fraction_with_higher_predicted_rlf': float((policy['rlf'] > hold['rlf']).mean()),
        'fraction_with_lower_predicted_delivery': float((policy['delivery_ratio'] < hold['delivery_ratio']).mean()),
        'fraction_with_lower_predicted_jain': float((policy['jain'] < hold['jain']).mean()),
    }


@torch.inference_mode()
def choose(model, app, history, parameters, clock):
    q = model(torch.from_numpy(states(history, parameters, clock))).numpy()
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite policy values')
    return np.where(valid_mask(parameters, app), q, -np.inf).argmax(1)


def action_summary(app, ids):
    counts = np.bincount(ids, minlength=len(action_catalog(app)))
    return [dict(action_id=i, label=action_catalog(app)[i]['label'], count=int(n))
            for i, n in enumerate(counts) if n]


def audit_contexts(data, ensemble, model, app, n):
    rng = np.random.default_rng(20261018)
    runs, bins, history, parameters = contexts(data, data['inner_holdout_idx'], rng, n)
    reference = data['params'][data['inner_fit_idx']]
    endings, policy_actions = {}, []
    for mode in ('hold', 'policy'):
        h, p = history.copy(), parameters.copy()
        for step in range(3):
            ids = np.zeros(n, int) if mode == 'hold' else choose(model, app, h, p, (bins + step) * 30)
            if mode == 'policy':
                policy_actions.append(ids)
            newp = apply_actions(p, ids, app)
            h, _, _, _ = transition(ensemble, app, h, p, newp,
                                    data['exog'][runs, bins + step], reference)
            p = newp
        endings[mode] = metrics(h)['short']
    return {
        'context_samples': n, 'sampling_seed': 20261018,
        'independent_holdout_runs': int(len(np.unique(runs))),
        'rollout_steps': 3, 'bin_width_s': 30,
        'hold': describe(endings['hold']), 'policy': describe(endings['policy']),
        'comparison': compare(endings['hold'], endings['policy']),
        'policy_actions_over_all_rollout_steps': action_summary(app, np.concatenate(policy_actions)),
    }


def audit_fixed_background(data, ensemble, model, app):
    """Synthetic first activation; each app alone, not the joint scheduled run."""
    now = 180 if app == 'lb' else 540
    k = now // 30
    ids = data['inner_holdout_idx']
    p = np.tile(np.asarray([43,43,43,9,9,9,6,0,-1,512,1.], dtype=float), (len(ids), 1))
    # Histories are response-model outputs under the same fixed background;
    # never splice observed histories collected under different parameters.
    h = np.stack([ensemble.predict(p, data['exog'][ids, j]).mean(0)
                  for j in range(k-3, k)], axis=1)
    actions = choose(model, app, h, p, now)
    newp = apply_actions(p, actions, app)
    reference = data['params'][data['inner_fit_idx']]
    held, _, _, _ = transition(ensemble, app, h, p, p, data['exog'][ids, k], reference)
    changed, _, _, _ = transition(ensemble, app, h, p, newp, data['exog'][ids, k], reference)
    hm, pm = metrics(held)['short'], metrics(changed)['short']
    return {
        'decision_time_s': now, 'app_alone': True, 'profiles': len(ids),
        'baseline_parameters': dict(zip(PARAMETERS, p[0].tolist())),
        'history_source': 'Synthetic response under this same baseline; not actual ns3 observations',
        'driver_source': 'Original-train inner-holdout profiles; future drivers are supplied only inside this synthetic audit',
        'actions': action_summary(app, actions),
        'hold_next_30s': describe(hm), 'policy_next_30s': describe(pm),
        'comparison': compare(hm, pm),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', type=int, default=2048)
    args = parser.parse_args(argv)
    if args.contexts <= 0:
        raise ValueError('Context count must be positive')
    torch.set_num_threads(2)
    paths = {
        'data': ROOT/'data_audit/data_arrays.npz',
        'response': ROOT/'training/response_ensemble.pt',
        'training_report': ROOT/'training/training_report.json',
        'lb': ROOT/'selected/lb.pt', 'mo': ROOT/'selected/mo.pt',
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    report = json.loads(paths['training_report'].read_text(encoding='utf-8'))
    if report['status'] != 'complete':
        raise RuntimeError('Offline training is not complete')
    data = dict(np.load(paths['data'], allow_pickle=False))
    checkpoint = torch.load(paths['response'], map_location='cpu', weights_only=True)
    members = []
    for weights in checkpoint['models']:
        member = ResponseNetwork()
        member.load_state_dict(weights, strict=True)
        member.eval().requires_grad_(False)
        members.append(member)
    ensemble = ResponseEnsemble(members, *[np.asarray(checkpoint[key]) for key in
        ('x_mean','x_std','z_mean','z_std')], 'cpu')
    policies = {}
    for app in ('lb', 'mo'):
        cp = torch.load(paths[app], map_location='cpu', weights_only=True)
        selected = next(row for row in report['policies'] if row['app'] == app)
        if selected['selected_sha256'] != hashes[app]:
            raise RuntimeError('Selected policy differs from completed training report')
        if cp['metadata']['response_ensemble_sha256'] != hashes['response']:
            raise RuntimeError('Policy and response-model hashes differ')
        if cp['metadata']['data_arrays_sha256'] != hashes['data']:
            raise RuntimeError('Policy and training-array hashes differ')
        model = QNetwork(STATE_DIM, len(action_catalog(app)))
        model.load_state_dict(cp['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        policies[app] = {
            'selected_seed': cp['metadata']['seed'],
            'inner_holdout_context_audit': audit_contexts(data, ensemble, model, app, args.contexts),
            'fixed_evaluation_background_preview': audit_fixed_background(data, ensemble, model, app),
        }
    if {key: digest(path) for key, path in paths.items()} != hashes:
        raise RuntimeError('Training or checkpoint changed during this audit')
    result = {
        'schema': 'offline-policy-sanity-v1', 'status': 'complete',
        'audit_script_sha256': digest(__file__), 'input_sha256': hashes,
        'partition': 'Only original train720; 576 response-fit / 144 inner-selection runs',
        'original_validation_or_test_used': False,
        'online_training': False, 'ns3_simulations_launched': 0,
        'interpretation': 'Synthetic diagnostics, not measured ns3 gains or causal intervention validation',
        'limitations': [
            'Successful-HO support removes the positive zero-HO bonus but cannot prove necessary handovers are preserved.',
            'The PP-local objective permits tradeoffs with RLF and delivery; compare every reported outcome.',
            'Static response has no learned switch transient; ensemble disagreement is not calibrated safety assurance.',
            'Future driver profiles are model-environment inputs only, never frozen runtime policy observations.',
            'Profiles and sampled windows are dependent; the sample count is not an independent replication count.',
        ],
        'policies': policies,
    }
    target = ROOT/'training/policy_sanity.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'output': str(target), 'input_sha256': hashes,
                      'mo_final': policies['mo']['inner_holdout_context_audit']['comparison']}, indent=2))


if __name__ == '__main__':
    main()
