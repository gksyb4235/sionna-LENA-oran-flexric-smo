"""Contract tests for isolation, frozen-policy verification, and matched timing."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = importlib.util.spec_from_file_location('offline_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(campaign)


class PlanTests(unittest.TestCase):
    def load(self, plan):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            path.write_text(json.dumps(plan), encoding='utf-8')
            return campaign.read_plan(path)

    def test_complete_matched_plan(self):
        plan = self.load(campaign.make_plan())
        self.assertEqual(len(plan['jobs']), 16)
        self.assertEqual({job['trace_seed'] for job in plan['jobs']}, {1419, 1420, 1421, 1422})
        self.assertEqual(plan['baseline']['cio_db'], [6, 0, -1])
        self.assertEqual(plan['baseline']['ttt_ms'], 512)
        self.assertFalse(plan['online_learning'])

    def test_reject_online_training_and_unmatched_controls(self):
        base = campaign.make_plan()
        invalid = []
        for field, value in [('policy', 'train'), ('mode', 'lb'), ('mo_start_s', 180), ('trace_seed', 1414)]:
            plan = copy.deepcopy(base)
            plan['jobs'][0][field] = value
            invalid.append(plan)
        plan = copy.deepcopy(base)
        plan['online_learning'] = True
        invalid.append(plan)
        plan = copy.deepcopy(base)
        plan['jobs'].pop()
        invalid.append(plan)
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.load(plan)

    def test_paths_are_isolated(self):
        with self.assertRaises(ValueError):
            campaign.safe_child(Path.cwd(), '../old-experiment')
        plan = campaign.make_plan()
        plan['binary'] = '../drl_xapp_conflict/bin/ns3-drl-xapp-conflict'
        with self.assertRaises(ValueError):
            self.load(plan)

    def test_commands_are_frozen_and_e2_disabled(self):
        plan = campaign.make_plan()
        job = plan['jobs'][3]
        args = campaign.controller_argv(plan, job, Path('/tmp/control'))
        self.assertEqual(args[args.index('--policy') + 1], 'eval')
        self.assertEqual(args[args.index('--lb-start-s') + 1], '180')
        self.assertEqual(args[args.index('--mo-start-s') + 1], '540')
        self.assertIn('--ns3::NrHelper::E2ModeNr=false', campaign.sim_argv(plan, job, Path('/tmp/run')))
        self.assertIn('--ns3::NrHelper::UseIdealRrc=true', campaign.sim_argv(plan, job, Path('/tmp/run')))
        self.assertIn('--cellCioDb=gNB_5G:6,gNB_4G_1:0,gNB_4G_2:-1', campaign.sim_argv(plan, job, Path('/tmp/run')))

    def test_smoke_has_two_decisions_after_complete_history_and_no_test_contamination(self):
        plan = campaign.make_smoke_plan()
        self.assertEqual(plan['duration_s'], 120)
        self.assertEqual(plan['n_ues'], 3)
        self.assertEqual(plan['control_start_s'], 90)
        self.assertEqual(plan['control_interval_s'], 15)
        self.assertEqual({job['mode'] for job in plan['jobs']}, {'none', 'both'})
        self.assertTrue(all(job['phase'] == 'smoke' and job['lb_start_s'] == job['mo_start_s'] == 90 for job in plan['jobs']))
        self.assertEqual(len(campaign.make_plan()['jobs']), 16)
        args = campaign.sim_argv(plan, plan['jobs'][0], Path('/tmp/smoke'))
        self.assertIn('--simTime=120', args)
        self.assertIn('--N_Ues=3', args)

    def test_smoke_plan_outside_experiment_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            path.write_text(json.dumps(campaign.make_smoke_plan()))
            with self.assertRaises(ValueError):
                campaign.read_smoke_plan(path)

    def test_terminal_counts_clear_stale_running_jobs(self):
        before = dict(status='failed', running_ids=['a', 'b'], completed_ids=[], failed_jobs=['b'])
        records = {'a': {'status': 'complete'}, 'b': {'status': 'failed'}, 'c': {'status': 'failed'}}
        result = campaign.terminal_campaign_state(before, records, {'d': {}})
        self.assertEqual(result['running_ids'], [])
        self.assertEqual(result['running_count'], 0)
        self.assertEqual(result['completed_ids'], ['a'])
        self.assertEqual(result['failed_ids'], ['b', 'c'])
        self.assertEqual((result['completed_count'], result['failed_count'], result['pending_count']), (1, 2, 1))

    def test_main_plan_rejects_even_empty_trace_override(self):
        plan = campaign.make_plan()
        plan['trace_overrides'] = {}
        with self.assertRaisesRegex(ValueError, 'without overrides'):
            self.load(plan)


class SmokeTraceTests(unittest.TestCase):
    def test_filter_preserves_original_and_only_selected_earliest_active_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.csv'
            original = ('# comment\ntime_s,ue_id,x,y,z,active\n'
                        '0,z,1,2,3,false\n20,a,1,2,3,true\n'
                        '5,b,1,2,3,true\n2,z,1,2,3,true\n'
                        '5,c,1,2,3,true\n30,d,1,2,3,true\n'
                        '120,z,1,2,3,false\n110,b,1,2,3,true\n')
            source.write_text(original)
            destination = root / 'filtered.csv'
            result = campaign.filter_smoke_trace(source, destination, 3)
            self.assertEqual(result['selected_ue_ids'], ['z', 'b', 'c'])
            self.assertEqual(result['output_ue_count'], 3)
            self.assertEqual(result['output_rows'], 6)
            self.assertEqual(source.read_text(), original)
            with destination.open() as stream:
                rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
            self.assertEqual({row['ue_id'] for row in rows}, {'z', 'b', 'c'})
            self.assertEqual(rows[0]['active'], 'false')
            with self.assertRaises(ValueError):
                campaign.filter_smoke_trace(source, destination, 3)
            with self.assertRaises(ValueError):
                campaign.filter_smoke_trace(source, source, 3)

    def test_smoke_override_is_used_and_hashed_in_its_own_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.csv'
            source.write_text('time_s,ue_id,x,y,z,active\n2,a,1,2,3,true\n3,b,1,2,3,true\n4,c,1,2,3,true\n5,d,1,2,3,true\n')
            session = root / 'smoke' / 'retry01'
            subset = campaign.filter_smoke_trace(source, session / 'inputs' / 'filtered.csv', 3)
            plan = campaign.make_smoke_plan()
            plan['trace_overrides'] = {'1419': subset['output']}
            plan['trace_override_metadata'] = {'1419': subset}
            plan_path = session / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            with patch.object(campaign, 'ROOT', root):
                accepted = campaign.read_smoke_plan(plan_path)
                args = campaign.sim_argv(accepted, accepted['jobs'][0], session / 'results')
                self.assertIn('--sumoTrace=' + subset['output'], args)
                Path(subset['output']).write_text('changed input')
                with self.assertRaisesRegex(ValueError, 'count/hash mismatch'):
                    campaign.read_smoke_plan(plan_path)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'control').mkdir()
        self.spec = dict(mode='lb', lb_start_s=1, mo_start_s=540, control_start_s=1, control_interval_s=30)
        self.before = dict(cio_5g=6, cio_4g1=0, cio_4g2=-1, ttt_ms=512, hys_db=1)
        self.after = {**self.before, 'cio_4g1': 1}
        with (self.root / 'cell_kpi.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('time_s', 'cell_id', 'cio_bias_db', 'ttt_ms', 'hysteresis_db'))
            writer.writeheader()
            for t in (.5, 1.5):
                params = self.before if t < 1 else self.after
                for cell, name in enumerate(('5g', '4g1', '4g2'), 1):
                    writer.writerow(dict(time_s=t, cell_id=cell, cio_bias_db=params['cio_' + name], ttt_ms=512, hysteresis_db=1))
        self.write('observation.json', dict(terminal=True, time_s=2))
        self.summary = dict(status='complete', last_time_s=2, policy='eval', online_learning=False,
                            model_hashes_before={'lb': 'abc'}, model_hashes_after={'lb': 'abc'},
                            online_gradient_updates=0, online_replay_inserts=0, exploratory_actions=0)
        self.write('summary.json', self.summary)
        self.action = dict(seq=0, time_s=1, terminal=False, updates={'cio_4g1': 1}, controller_ms=.5,
                           apps={'lb': {'decision_ms': .2}})
        self.receipt = dict(seq=0, time_s=1, status='applied', updates={'cio_4g1': 1}, before=self.before, after=self.after)
        self.write_lines('actions.jsonl', [self.action, dict(seq=1, time_s=2, terminal=True, updates={})])
        self.write_lines('receipts.jsonl', [self.receipt, dict(seq=1, time_s=2, status='terminal')])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        (self.root / 'control' / name).write_text(json.dumps(value), encoding='utf-8')

    def write_lines(self, name, values):
        (self.root / 'control' / name).write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')

    def result(self):
        return campaign.verify(self.root, 2, self.spec)

    def test_real_readback_and_planned_timing_pass(self):
        result = self.result()
        self.assertTrue(result['passed'], result['issues'])
        self.assertEqual(result['control_timing'][0]['planned_simulation_s'], 1)
        self.assertEqual(result['control_timing'][0]['per_app_decision_ms'], {'lb': .2})

    def test_policy_updates_are_rejected(self):
        for field, changed in [('online_gradient_updates', 1), ('online_replay_inserts', 1),
                               ('exploratory_actions', 1), ('online_learning', True),
                               ('model_hashes_after', {'lb': 'different'})]:
            with self.subTest(field=field):
                self.write('summary.json', {**self.summary, field: changed})
                self.assertFalse(self.result()['passed'])

    def test_missing_frozen_attestation_rejected(self):
        value = dict(self.summary)
        del value['online_replay_inserts']
        self.write('summary.json', value)
        self.assertFalse(self.result()['passed'])

    def test_parameter_not_actually_applied_is_rejected(self):
        self.receipt['after'] = self.before
        self.write_lines('receipts.jsonl', [self.receipt, dict(status='terminal')])
        self.assertIn('action_receipt_readback_mismatch', self.result()['issues'])

    def test_out_of_schedule_apply_rejected(self):
        self.action['time_s'] = self.receipt['time_s'] = 1.25
        self.write_lines('actions.jsonl', [self.action])
        self.write_lines('receipts.jsonl', [self.receipt, dict(status='terminal')])
        self.assertIn('control_not_applied_at_planned_simulation_time', self.result()['issues'])

    def test_mobility_must_not_run_in_lb_only_arm(self):
        self.action['updates']['hys_db'] = 2
        self.write_lines('actions.jsonl', [self.action])
        self.assertIn('unexpected_app_or_preactivation_control', self.result()['issues'])

    def test_incomplete_cell_csv_rejected(self):
        path = self.root / 'cell_kpi.csv'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1]) + '\n')
        self.assertFalse(self.result()['passed'])


if __name__ == '__main__':
    unittest.main()
