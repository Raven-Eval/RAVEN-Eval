"""Select one of three generated candidates with a fixed anonymous LMM judge."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from . import judge


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True,help='Fixed candidate-selection LMM, separate from final judges')
    parser.add_argument('--api-url',required=True)
    parser.add_argument('--api-key-env',default='LMM_API_KEY')
    parser.add_argument('--work-dir',type=Path,default=Path('runs/selection'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    grouped=defaultdict(list)
    for line in args.candidates.read_text().splitlines():
        if line.strip():
            row=json.loads(line);grouped[row['task_type'],row['task_id'],row['model']].append(row)
    if not grouped:
        parser.error('No candidates')
    if args.output.exists():
        parser.error('Selection output exists; use a new output file')
    root=args.root.resolve();chosen=[]
    for (kind,task_id,model),rows in sorted(grouped.items()):
        if len(rows)!=3 or len({r['candidate'] for r in rows})!=3:
            parser.error('Expected exactly three distinct candidate indices per task/model')
        if len({r['model_version'] for r in rows})!=1:
            parser.error('Candidate model versions differ')
        for row in rows:
            path=Path(row['video_path'])
            if not (path if path.is_absolute() else root/path).is_file():
                parser.error('Candidate video missing')
        task=json.loads((root/'data/tasks'/kind/(task_id+'.json')).read_text())
        if args.dry_run:
            continue
        group_id=hashlib.sha256(json.dumps([kind,task_id,model]).encode()).hexdigest()[:20]
        work=args.work_dir.resolve()/group_id;work.mkdir(parents=True,exist_ok=True)
        guideline=work/'guideline.txt'
        guideline.write_text('Compare candidates from the same video generator. Prioritize task semantic completion and realization of specified details. Apply primary rubric criteria first. Use reference images to check endpoint adherence for I2V. Ignore audio. Return exactly Better, Worse, or Comparable for the first video relative to the second.\n')
        pairfile=work/(task_id+'_pairwise_triples.json')
        if not pairfile.exists():
            triples=[]
            for i in range(3):
                for j in range(i+1,3):
                    triples.append({'triple_id':f'{i}_{j}','generated_video_a':{'model':f'Candidate-{i}','video_path':rows[i]['video_path']},'generated_video_b':{'model':f'Candidate-{j}','video_path':rows[j]['video_path']},'prompt':task['prompt'],'scoring_criteria':task['rubric']})
            pairfile.write_text(json.dumps({'task_id':task_id,'reference_first_frame_path':task['references'].get('first'),'reference_last_frame_path':task['references'].get('last'),'pairwise_comparison_triples':triples},ensure_ascii=False,indent=2)+'\n')
        else:
            saved=json.loads(pairfile.read_text())
            actual={t[f'generated_video_{side}']['video_path'] for t in saved['pairwise_comparison_triples'] for side in ('a','b')}
            if actual!={r['video_path'] for r in rows}:
                parser.error('Selection work directory contains a different candidate set; use a new --work-dir')
        task_type={'t2v':'t2v','flt':'first_last','kft':'keyframe'}[kind]
        status=judge.main(['--root',str(root),'--task-types',task_type,'--task-json',str(pairfile),'--guideline-file',str(guideline),'--model',args.model,'--api-url',args.api_url,'--api-key-env',args.api_key_env,'--output-base','candidate_selector','--state-json',str(work/'state.json'),'--summary-json',str(work/'summary.json'),'--log-dir',str(work/'logs'),'--frame-cache-dir',str(args.work_dir.resolve()/'frames')])
        if status:
            raise RuntimeError('Candidate selection incomplete; check selection logs')
        scores=[0.,0.,0.]
        for triple in json.loads(pairfile.read_text())['pairwise_comparison_triples']:
            i,j=map(int,triple['triple_id'].split('_'))
            forward=triple['candidate_selector_evaluation']['output_text']
            reverse=triple['candidate_selector_reversed_evaluation']['output_text']
            if (forward,reverse)==('Better','Worse'):scores[i]+=1
            elif (forward,reverse)==('Worse','Better'):scores[j]+=1
            else:scores[i]+=.5;scores[j]+=.5
        winner=max(range(3),key=lambda i:(scores[i],-int(rows[i]['candidate'])))
        chosen.append({**rows[winner],'selected':True,'selection':{'judge_model':args.model,'rule':'bidirectional candidate round-robin wins; unresolved pairs 0.5 each; tie by lowest candidate index','candidate_scores':scores}})
    if not args.dry_run:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    print(json.dumps({'groups':len(grouped),'selected':len(chosen),'dry_run':args.dry_run}))
    return 0


if __name__=='__main__':raise SystemExit(main())
