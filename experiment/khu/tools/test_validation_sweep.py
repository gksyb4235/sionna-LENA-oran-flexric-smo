"""Validate distributed queues and source schedules without simulations."""
import importlib.util, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('validation_sweep.py'));runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
M=json.loads((ROOT/'experiment/khu/validation_20261111/manifest.json').read_text())

class QueueTests(unittest.TestCase):
 def test_partitions_cover_without_overlap(self):
  p=M['partitions'];groups=[set(p[x]) for x in ['local_retry','pc1','pc2','pc3','pc4','server_container','reuse_verified']]
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
 def test_whole_waves_and_optimal_makespan(self):
  model=M['allocation_model'];total=0
  for host,workers in model['workers'].items():
   new=len(M['partitions'][host]);started=model['already_started_jobs'][host];n=new+len(started)
   self.assertEqual(new,workers*model['new_queue_cycles'][host]);self.assertEqual(n,workers*model['cycles'][host]);self.assertEqual(n,model['machine_total_jobs'][host]);total+=n
  self.assertEqual(total,368);self.assertEqual(model['cycles']['pc2'],5)
  self.assertEqual(model['already_started_jobs']['pc1'],M['partitions']['local_retry'])
  self.assertTrue(set(M['partitions']['pc1']).isdisjoint(M['partitions']['local_retry']))
  self.assertEqual(model['cycles']['pc3'],3);self.assertEqual(model['cycles']['pc4'],3)
  self.assertEqual(model['cycles']['server_container'],3)
  self.assertEqual(max([model['cycles'][k] for k in ['pc1','pc2','pc3','pc4']]+[1.5*model['cycles']['server_container']]),6)
  self.assertFalse(any(16*a+110+26*c+84==368 for a in range(1,6) for c in range(1,6)))
  self.assertTrue({'rq3_0009','rq3_0063','rq3_0064','rq3_0065','rq3_0067','rq3_0068'}<=set(M['partitions']['pc4']))
 def test_pc1_prior_retry_guard(self):
  import tempfile
  spec=importlib.util.spec_from_file_location('starter',Path(__file__).with_name('start_validation_native.py'));starter=importlib.util.module_from_spec(spec);spec.loader.exec_module(starter)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'experiment/khu/results/rq3_retry_20261111';out.mkdir(parents=True);proc=root/'proc'
   (out/'launch.json').write_text(json.dumps({'pid':42}));(proc/'42').mkdir(parents=True);(proc/'42/cmdline').write_bytes(b'python3\x00restart_imported_rq3.py\x00--execute')
   self.assertEqual(starter.live_prior_retry_pids(root,M,'pc1',proc),[42]);self.assertEqual(starter.live_prior_retry_pids(root,M,'pc2',proc),[])
   (proc/'42/cmdline').write_bytes(b'unrelated_process')
   job=out/M['partitions']['local_retry'][0];job.mkdir();(job/'execution.json').write_text(json.dumps({'pid':43,'status':'running'}));(proc/'43').mkdir();(proc/'43/cmdline').write_bytes(b'/build/ns3-khu-oran-optimized\x00')
   self.assertEqual(starter.live_prior_retry_pids(root,M,'pc1',proc),[43])
   (proc/'43/cmdline').unlink();self.assertEqual(starter.live_prior_retry_pids(root,M,'pc1',proc),[])
 def test_machine_folders_cover_actual_assignment(self):
  all_ids=[]
  for part,folder in [('pc1','PC1'),('pc2','PC2'),('pc3','PC3'),('pc4','PC4'),('server_container','Server')]:
   data=json.loads((ROOT/'experiment/khu/validation_20261111'/folder/'jobs.json').read_text());rows=data['job_list']
   self.assertEqual(data['jobs'],len(rows));self.assertEqual(data['jobs'],data['cores']*data['cycles'])
   self.assertEqual([r['job_id'] for r in rows if r['launched_by_run_sh']],M['partitions'][part]);all_ids.extend(r['job_id'] for r in rows)
  self.assertEqual(len(all_ids),368);self.assertEqual(len(set(all_ids)),368)

if __name__=='__main__':unittest.main()
