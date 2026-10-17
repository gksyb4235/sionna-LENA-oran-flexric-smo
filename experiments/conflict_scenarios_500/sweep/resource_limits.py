"""Linux cgroup limits in addition to affinity and host MemAvailable."""
from pathlib import Path
def groups(controller):
    result=[]
    try:
        rows=Path('/proc/self/cgroup').read_text().splitlines()
        mounts=Path('/proc/self/mountinfo').read_text().splitlines()
        for row in rows:
            _,controllers,current=row.split(':',2)
            for mount in mounts:
                left,right=mount.split(' - ',1);fields=left.split();kind,_,options=right.split()[:3]
                if not ((kind=='cgroup2' and not controllers) or
                        (kind=='cgroup' and controller in controllers.split(',') and controller in options.split(','))): continue
                base=Path(fields[4]);root=Path(fields[3]);current_path=Path(current)
                if current_path.is_relative_to(root): path=base/current_path.relative_to(root)
                elif current=='/': path=base
                else: continue
                while path.is_relative_to(base):
                    result.append((path,kind))
                    if path==base: break
                    path=path.parent
    except (OSError,ValueError): pass
    return result
def cpu_quota():
    values=[]
    for path,kind in groups('cpu'):
        try:
            if kind=='cgroup2': quota,period=(path/'cpu.max').read_text().split()
            else: quota=(path/'cpu.cfs_quota_us').read_text();period=(path/'cpu.cfs_period_us').read_text()
            if quota.strip()!='max' and float(quota)>0: values.append(float(quota)/float(period))
        except (OSError,ValueError,ZeroDivisionError): pass
    return min(values) if values else None
def memory_available():
    values=[]
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'): values.append(int(line.split()[1])*1024)
    except (OSError,ValueError): pass
    for path,kind in groups('memory'):
        try:
            limit=(path/('memory.max' if kind=='cgroup2' else 'memory.limit_in_bytes')).read_text().strip()
            used=int((path/('memory.current' if kind=='cgroup2' else 'memory.usage_in_bytes')).read_text())
            if limit!='max': values.append(max(0,int(limit)-used))
        except (OSError,ValueError): pass
    return min(values) if values else None
