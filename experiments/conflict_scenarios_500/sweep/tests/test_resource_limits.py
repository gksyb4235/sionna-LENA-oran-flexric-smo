import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import resource_limits as limits
import sweep

class ResourceTests(unittest.TestCase):
    def test_v2_remaining_memory_and_parent_cpu_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=root/'nested';child.mkdir()
            (root/'cpu.max').write_text('250000 100000')
            (child/'cpu.max').write_text('max 100000')
            (root/'memory.max').write_text('1000')
            (root/'memory.current').write_text('800')
            (child/'memory.max').write_text('max')
            (child/'memory.current').write_text('100')
            with patch.object(limits,'groups',return_value=[(child,'cgroup2'),(root,'cgroup2')]):
                self.assertEqual(limits.cpu_quota(),2.5)
                self.assertEqual(limits.memory_available(),200)
    def test_v1_limits_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'cpu.cfs_quota_us').write_text('100000')
            (root/'cpu.cfs_period_us').write_text('100000')
            (root/'memory.limit_in_bytes').write_text('1000')
            (root/'memory.usage_in_bytes').write_text('1100')
            with patch.object(limits,'groups',return_value=[(root,'cgroup')]):
                self.assertEqual(limits.cpu_quota(),1)
                self.assertEqual(limits.memory_available(),0)
    def test_cpu_quota_rejects_even_with_large_affinity(self):
        with patch.object(sweep,'cpu_quota',return_value=2.5),patch.object(sweep.os,'sched_getaffinity',return_value=set(range(40))),patch.object(sweep.shutil,'which',return_value='/usr/bin/taskset'):
            self.assertEqual(sweep.cpu_slots(1,2),[[0],[1]])
            with self.assertRaisesRegex(ValueError,'quota'): sweep.cpu_slots(1,3)
    def test_resolves_namespaced_cgroup_mount_and_parents(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if str(path)=='/proc/self/cgroup': return '0::/parent/child\n'
            if str(path)=='/proc/self/mountinfo': return '42 23 0:28 /parent /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):
            self.assertEqual(limits.groups('cpu'),[(Path('/sys/fs/cgroup/child'),'cgroup2'),(Path('/sys/fs/cgroup'),'cgroup2')])
