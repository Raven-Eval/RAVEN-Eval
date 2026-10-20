"""Choose two reliable anchors from each of five capability tiers."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from .ranking import aggregate


def load(path):
    scores={}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            key=(row['task_id'],row['model'])
            if key in scores:
                raise ValueError(f'Duplicate task/model score: {key}')
            scores[key]=float(row['skill'])
    return scores


def select(joint, judges, tau=1.):
    models=sorted({model for task,model in joint})
    if len(models)<10:
        raise ValueError('At least ten old models are needed for five tiers with two anchors each')
    # Correlate ranks across tasks, not raw uncentered skills across models.
    def ranks(scores):
        from scipy.stats import rankdata
        by_task={}
        for task,model in scores:
            by_task.setdefault(task,{})[model]=scores[task,model]
        result={}
        for task,values in by_task.items():
            names=sorted(values)
            for model,rank in zip(names,rankdata([-values[m] for m in names],method='average')):
                result[task,model]=rank
        return result
    joint_ranks=ranks(joint);judge_ranks=[ranks(j) for j in judges]
    taskfits={}
    for (task,model),score in joint.items():taskfits.setdefault(task,{})[model]=score
    ordered=[r['model'] for r in aggregate(list(taskfits.values()))]
    quality={}
    for model in models:
        keys=sorted(k for k in joint if k[1]==model)
        if len(keys)<3 or any(set(j)!=set(joint) for j in judges):
            raise ValueError('Require >=3 tasks per model and identical joint/per-judge task-model coverage')
        rhos=[float(spearmanr([joint_ranks[k] for k in keys],[jr[k] for k in keys]).statistic) for jr in judge_ranks]
        if not np.isfinite(rhos).all():
            quality[model]=None
        else:
            quality[model]={'rho':rhos,'q':float(np.mean(rhos)-tau*np.std(rhos))}
    anchors=[];tiers=[]
    for tier in np.array_split(np.array(ordered,dtype=object),5):
        eligible=[str(m) for m in tier if quality[str(m)] is not None]
        if len(eligible)<2:
            raise ValueError('A capability tier has fewer than two anchors with defined reliability')
        chosen=sorted(eligible,key=lambda m:(-quality[m]['q'],m))[:2]
        anchors.extend(chosen);tiers.append({'models':list(tier),'anchors':chosen})
    return anchors,{'tau':tau,'quality':quality,'tiers':tiers}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-scores',type=Path,required=True)
    parser.add_argument('--judge-scores',type=Path,action='append',required=True)
    parser.add_argument('--task-type',choices=['t2v','flt','kft'],required=True)
    parser.add_argument('--tau',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    anchors,details=select(load(args.joint_scores),[load(p) for p in args.judge_scores],args.tau)
    result=json.loads(args.output.read_text()) if args.output.exists() else {}
    if args.task_type in result:
        parser.error('Anchor version already exists; write a new file to change it')
    result[args.task_type]=anchors;result.setdefault('_reliability',{})[args.task_type]=details
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Selected ten {args.task_type} anchors: {args.output}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
