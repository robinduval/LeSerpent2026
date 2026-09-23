"""Five held-out episodes in independent 5 Hz processes; no training or selection."""
from pathlib import Path
import concurrent.futures
import json
import hashlib
import subprocess
import sys
import time
import csv
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from snake_rl.metrics import summarize


def main():
 manifest=json.loads((BASE/'checkpoints/selection.json').read_text())
 checkpoint=BASE/'checkpoints/selected.pt'
 checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if checksum!=manifest['selected_sha256']:
  raise SystemExit('Checkpoint changed after validation selection')
 destination=BASE/'runs/final_test'
 if destination.exists():
  raise SystemExit('Final test already exists; do not rerun it for model selection.')
 destination.mkdir()
 config={'purpose':'held-out final test, never used for tuning or selection',
  'seeds':manifest['final_test_seeds'],'clock_hz':5,'max_steps':2000,
  'parallel_processes':5,'training_concurrent':False,
  'selected_sha256':checksum,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
 (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 def run(seed):
  name=f'final_seed_{seed}'
  cmd=[sys.executable,str(BASE/'serpent-algo.py'),'--evaluate','--checkpoint',str(checkpoint),
       '--run-id',name,'--seed',str(seed),'--episodes','1','--max-steps','2000','--trace']
  result=subprocess.run(cmd,cwd=BASE.parent,capture_output=True,text=True)
  (destination/f'process_{seed}.log').write_text(result.stdout+result.stderr)
  if result.returncode:
   raise RuntimeError(f'Final episode {seed} failed: exit {result.returncode}')
  rows=[json.loads(line) for line in (BASE/'runs'/name/'evaluation.jsonl').read_text().splitlines()]
  if len(rows)!=1 or rows[0]['seed']!=seed or rows[0]['model_id']!='selected:'+checksum[:12]:
   raise RuntimeError('Final episode provenance mismatch')
  print(json.dumps(rows[0]),flush=True)
  return rows[0]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  rows=list(pool.map(run,config['seeds']))
 (destination/'evaluation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with (destination/'evaluation.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary=summarize(rows)
 (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
