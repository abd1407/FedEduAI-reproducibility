"""Measure a controlled command; interface bytes are host-wide, not app-attributed."""
import argparse,json,subprocess,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--interface',required=True);p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('Supply a command after --')
path=Path(a.output)
if path.exists():p.error('Output already exists; choose a new name')
def network():
 n=psutil.net_io_counters(pernic=True)
 if a.interface not in n:raise ValueError('Unknown network interface')
 return n[a.interface]
before=network();start=time.perf_counter();peak=0;cpu={};proc=subprocess.Popen(cmd)
try:
 while proc.poll() is None:
  try:
   family=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True);rss=0
   for child in family:
    try:
     rss+=child.memory_info().rss;t=child.cpu_times();cpu[child.pid]=t.user+t.system
    except psutil.Error:pass
   peak=max(peak,rss)
  except psutil.Error:pass
  time.sleep(.25)
except KeyboardInterrupt:
 proc.terminate();proc.wait()
after=network()
result={'exit_code':proc.returncode,'wall_seconds':time.perf_counter()-start,'peak_process_tree_rss_bytes':peak,
 'observed_process_cpu_seconds':sum(cpu.values()),'interface':a.interface,
 'interface_bytes_sent':after.bytes_sent-before.bytes_sent,'interface_bytes_recv':after.bytes_recv-before.bytes_recv,
 'scope':'Host interface counters over command duration; includes other traffic. Use a dedicated interface/host. Short-lived process CPU may be missed.'}
path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
raise SystemExit(proc.returncode or 0)
