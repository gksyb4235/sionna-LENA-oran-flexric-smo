"""ID range and CPU affinity contracts; execute only small Python fixtures."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

BUNDLE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BUNDLE))
import sweep
from test_sweep_contract import fixture_job,fixture_manifest


class RangeCpuTests(unittest.TestCase):
    def test_range_uses_name_not_priority_row_or_trace_and_includes_endpoints(self):
        manifest=sweep.read_json(BUNDLE/'data/execution_manifest.json')
        jobs=sweep.jobs_in_range(manifest,list(reversed(manifest['jobs'])),51,100)
        self.assertEqual(len(jobs),50)
        self.assertEqual([j['job_id'] for j in jobs],[f'conflict_{i:04d}' for i in range(51,101)])
        self.assertEqual(len(sweep.jobs_in_range(manifest,manifest['jobs'],500,500)),1)
        for first,last in ((0,20),(100,99),(500,599),(1,501)):
            with self.assertRaises(ValueError): sweep.jobs_in_range(manifest,manifest['jobs'],first,last)

    def test_cpu_slots_use_sparse_allowed_ids_without_overlap(self):
        with patch.object(sweep.os,'sched_getaffinity',return_value={2,4,8,9}), \
             patch.object(sweep.shutil,'which',return_value='/usr/bin/taskset'):
            self.assertEqual(sweep.cpu_slots(2,2),[[2,4],[8,9]])
            for cores,workers in ((2,3),(0,1),(1,0)):
                with self.assertRaises(ValueError): sweep.cpu_slots(cores,workers)

    @unittest.skipUnless(sys.platform=='linux' and shutil.which('taskset'),'Requires Linux taskset')
    def test_launched_child_really_has_affinity_and_thread_limit(self):
        cpu=min(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'site').mkdir()
            job=fixture_job();manifest=fixture_manifest(job)
            config={'ns3_root':str(root),'python_site_packages':'site'}
            code='import os,json; print(json.dumps([sorted(os.sched_getaffinity(0)),os.environ["OMP_NUM_THREADS"]]))'
            with patch.object(sweep,'argv_for',return_value=[sys.executable,'-c',code]):
                item=sweep.launch(manifest,config,job,root/'results',{},cpu_ids=[cpu])
            try:
                self.assertEqual(item['process'].wait(timeout=10),0)
            finally:
                sweep.stop_owned(item['process']);item['stdout'].close();item['stderr'].close()
            observed=json.loads((item['attempt']/'stdout.log').read_text())
            self.assertEqual(observed,[[cpu],'1'])
            self.assertEqual(item['record']['cpu_ids'],[cpu])

    @unittest.skipUnless(sys.platform=='linux' and shutil.which('taskset') and len(os.sched_getaffinity(0))>=2,'Requires two allowed CPUs')
    def test_concurrent_runner_reuses_freed_cpu_slot_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'site').mkdir()
            jobs=[fixture_job() for _ in range(3)]
            for i,job in enumerate(jobs): job['job_id']=f'fixture_{i}'
            manifest=fixture_manifest(jobs[0]);manifest['jobs']=jobs
            config=dict(ns3_root=str(root),python_site_packages='site',output_root='results',
                memory_reserve_gib=0,estimated_job_gib=.000001,disk_reserve_gib=0,
                estimated_output_gib_per_job=.000001,startup_stagger_s=0,timeout_s=30)
            def command(manifest,config,job,attempt):
                code='import time,runpy,sys; time.sleep(2.5); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
                return [sys.executable,'-c',code,str(BUNDLE/'tests/dummy_backend.py'),
                        str(attempt/'job.json'),str(attempt/'data'/job['job_id'])]
            with patch.object(sweep,'preflight',return_value={'input_fingerprints':[]}), \
                 patch.object(sweep,'argv_for',side_effect=command):
                first=sweep.run_batch(manifest,config,jobs,workers=2,shard_index=0,shard_count=1,
                                      resume=True,retry_failed=False,cores_per_job=1)
                second=sweep.run_batch(manifest,config,jobs,workers=2,shard_index=0,shard_count=1,
                                       resume=True,retry_failed=False,cores_per_job=1)
            self.assertEqual(first['status'],'complete')
            records=[sweep.read_json(root/'results/jobs'/j['job_id']/'attempt0001/execution.json') for j in jobs]
            self.assertNotEqual(records[0]['cpu_ids'],records[1]['cpu_ids'])
            self.assertEqual(records[0]['cpu_ids'],records[2]['cpu_ids'])
            self.assertEqual(second['skipped_complete'],[j['job_id'] for j in jobs])


if __name__=='__main__': unittest.main()
