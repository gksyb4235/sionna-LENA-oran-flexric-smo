"""Validate distributed queues and source schedules without simulations."""
import importlib.util, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('validation_sweep.py'));runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
M=json.loads((ROOT/'experiment/khu/validation_20261111/manifest.json').read_text())

class QueueTests(unittest.TestCase):
 def test_partitions_cover_without_overlap(self):
  p=M['partitions'];groups=[set(p[x]) for x in ['local_retry','peer_native','server_container','reuse_verified']]
  self.assertEqual(sum(map(len,groups)),len(set.union(*groups)))
  self.assertEqual(set.union(*groups),{j['job_id'] for j in M['jobs']})
 def test_ids_reject_overlap_and_bounds(self):
  for text in ['1,1','3-1','1-3,3','9999','-1']:
   with self.assertRaises(ValueError):runner.parse_ids(text,len(M['jobs']))
  self.assertEqual(runner.parse_ids('54,63-65,67-68',len(M['jobs'])),[54,63,64,65,67,68])
 def test_mappings_and_llm_retention(self):
  maps=M['evaluation_mapping'];self.assertEqual(len(maps),912)
  for variant in ['full_r2','full_r3','full_r4','full_r5','llm_only','shap_greedy']:
   rows=[r for r in maps if r['variant']==variant];self.assertEqual(len(rows),120)
   self.assertEqual(len({r['episode_id'] for r in rows}),120)
  llm=[r for r in maps if r['variant']=='llm_only']
  self.assertEqual(sum(r['selected_plan_available'] for r in llm),120)
  self.assertEqual(sum(r['predicted_feasible'] for r in llm),32)
 def test_every_schedule_matches_plan_and_history(self):
  import csv,hashlib
  trajectories={}
  for j in M['jobs']:
   path=ROOT/j['schedule'];self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),j['schedule_sha256'])
   with path.open() as f:receipts=[(float(r['time_s']),r['parameter'],float(r['value'])) for r in csv.DictReader(f)]
   self.assertEqual(receipts,[(float(e['time_s']),p,float(v)) for e in j['events'] for p,v in e['updates'].items()])
   state=dict(j['initial_parameters']);events={e['time_s']:e['updates'] for e in j['events']};states=[]
   for t in range(900):state.update(events.get(t,{}));states.append(dict(state))
   trajectories[j['job_id']]=states
  refs={j['candidate_id']:j for j in M['jobs'] if j['kind']=='approve_or_hold'}
  for j in M['jobs']:
   self.assertEqual(trajectories[j['job_id']][:j['t0_s']],trajectories[refs[j['candidate_id']]['job_id']][:j['t0_s']])
  for r in M['evaluation_mapping']:
   if 'plan_provenance' not in r:continue
   p=json.loads((ROOT/r['plan_provenance']).read_text());job=M['jobs'][r['job_index']]
   seq=p['selected_plan']['evaluation']['schedule_controls'] if r['selected_plan_available'] else [p['scenario']['baseline_controls']]*r['horizon_minutes']
   for s,row in enumerate(seq):
    for t in range(r['t0_s']+60*s,r['t0_s']+60*(s+1)):
     self.assertEqual([trajectories[job['job_id']][t][k] for k in M['parameters']],row)
 def test_shards_partition_exactly(self):
  ids=M['partitions']['new_all']
  for n in [2,3,4,7,16]:
   sets=[set(ids[i::n]) for i in range(n)]
   self.assertEqual(sum(map(len,sets)),len(ids));self.assertEqual(set.union(*sets),set(ids))

if __name__=='__main__':unittest.main()
