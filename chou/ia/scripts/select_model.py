"""Select only from equal-budget validation, never from final test results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from snake_rl.agent import DQNAgent
from snake_rl.metrics import select_candidate, summarize, model_selection_key

# Candidate definitions are explicit: no automatic max-record checkpoint scan.
CANDIDATES={
 'validation_ddqn11': 'ddqn11_1m_seed17/candidate_001000000.pt',
 'validation_ordered1m': 'ordered_1m_seed17/candidate_001000000.pt',
 'validation_n3': 'ddqn11_n3_1m_seed17/candidate_001000000.pt',
 'validation_per': 'ddqn11_per_1m_seed17/candidate_001000000.pt',
 'validation_per_n3': 'ddqn11_per_n3_1m_seed17/candidate_001000000.pt',
 'validation_ordered_seed23': 'ordered_1m_seed23/candidate_001000000.pt',
 'validation_ordered2m': 'ordered_5m_seed17/candidate_002000000.pt',
 'validation_ordered5m': 'ordered_5m_seed17/candidate_005000000.pt',
}


def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--install',action='store_true',help='Write the selected inference checkpoint and provenance manifest')
 args=p.parse_args()
 rows={}
 for name,relative in CANDIDATES.items():
  directory=BASE/'runs'/name
  if not (directory/'summary.json').exists():
   raise SystemExit(f'Validation not complete: {name}')
  data=[json.loads(line) for line in (directory/'evaluation.jsonl').read_text().splitlines()]
  if len(data)!=3 or [r['seed'] for r in data]!=[10001,10002,10003]:
   raise SystemExit(f'Unexpected validation seeds/size: {name}')
  if any(r['clock_hz']!=5 or r['evaluation_step_budget']!=600 or r['termination_reason']=='user_quit' for r in data):
   raise SystemExit(f'Invalid validation protocol: {name}')
  path=BASE/'runs'/relative
  expected=path.stem+':'+digest(path)[:12]
  if any(r['model_id']!=expected for r in data):
   raise SystemExit(f'Checkpoint provenance does not match results: {name}')
  rows[name]=data
 selected=select_candidate(rows)
 source=BASE/'runs'/CANDIDATES[selected]
 manifest={
  'selected_validation':selected,'source_checkpoint':str(source.relative_to(BASE)),
  'source_sha256':digest(source),'validation_seeds':[10001,10002,10003],
  'evaluation_hz':5,'evaluation_step_budget':600,
  'criterion':'maximum mean score, median, lower quartile, completions; time only if exact score multiset matches and all games ended naturally',
  'official_aggregation_known':False,
  'external_cutoffs_classability':'unknown; scores at fixed budget are a local validation convention',
  'qualification':'selected on 3 validation episodes; not evidence of completion or first place',
  'final_test_seeds':[20001,20002,20003,20004,20005],
  'final_test_used_for_selection':False,
  'candidates':{name:{'checkpoint':CANDIDATES[name], 'selection_key':model_selection_key(data),
                       'summary':summarize(data)} for name,data in rows.items()}}
 print(json.dumps(manifest,indent=2))
 if args.install:
  target=BASE/'checkpoints'/'selected.pt'
  if target.exists():
   raise SystemExit('selected.pt already exists; explicit archival/removal required before changing delivery')
  agent=DQNAgent.load(source)
  agent.save(target,metadata={'qualification':manifest['qualification'],
                 'source_checkpoint':manifest['source_checkpoint'],
                 'source_sha256':manifest['source_sha256'],
                 'selection_validation':selected},include_replay=False)
  manifest['selected_sha256']=digest(target)
  manifest['training_transitions']=agent.env_steps
  manifest['training_updates']=agent.updates
  manifest['encoder']=agent.config.encoder
  (BASE/'checkpoints'/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
  # A separately evaluated 11-feature model remains explicitly available.
  original=BASE/'runs'/CANDIDATES['validation_ddqn11']
  (BASE/'checkpoints'/'reference_classic11.pt').write_bytes(original.read_bytes())


if __name__=='__main__': main()
