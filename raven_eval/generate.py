"""Generate three candidates through a user-supplied provider adapter."""
import argparse
import importlib
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--task-type',choices=['t2v','flt','kft'],required=True)
    parser.add_argument('--task-id',action='append',default=[])
    parser.add_argument('--model',required=True)
    parser.add_argument('--model-version',required=True)
    parser.add_argument('--provider',required=True,help='Importable module:function')
    parser.add_argument('--parameters',type=Path,help='JSON containing non-secret generation parameters')
    parser.add_argument('--candidates',type=int,default=3)
    parser.add_argument('--output-dir',type=Path,default=Path('runs/generated'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.candidates<1:
        parser.error('--candidates must be positive')
    root=args.root.resolve()
    tasks=[json.loads(p.read_text()) for p in sorted((root/'data/tasks'/args.task_type).glob('*.json'))]
    if args.task_id:
        tasks=[t for t in tasks if t['task_id'] in args.task_id]
        if {t['task_id'] for t in tasks}!=set(args.task_id):
            parser.error('Unknown task ID')
    parameters=json.loads(args.parameters.read_text()) if args.parameters else {}
    if not tasks:
        parser.error('No tasks found')
    if args.dry_run:
        print(json.dumps({'tasks':len(tasks),'candidates':len(tasks)*args.candidates,'model':args.model,'provider':args.provider,'api_calls':0}));return 0
    module,sep,function=args.provider.partition(':')
    if not sep:
        parser.error('--provider must be module:function')
    adapter=getattr(importlib.import_module(module),function)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=output/'candidates.jsonl'
    if manifest.exists():
        parser.error('Candidates manifest exists; use a new output directory')
    # Persist each successful candidate immediately so completed work remains inspectable.
    with manifest.open('w') as stream:
        for task in tasks:
            request={key:task[key] for key in ('task_id','task_type','prompt','task_version')}
            request['references']={side:str(root/path) for side,path in task['references'].items()}
            for index in range(args.candidates):
                folder=output/task['task_type']/task['task_id']/str(index+1);folder.mkdir(parents=True,exist_ok=True)
                path=Path(adapter(task=request,model=args.model,output_dir=folder,candidate_index=index,parameters=parameters)).resolve()
                if not path.is_file():
                    raise FileNotFoundError('Provider did not return an existing video file')
                try:relative=str(path.relative_to(root))
                except ValueError:relative=str(path)
                record={'task_type':task['task_type'],'task_id':task['task_id'],'model':args.model,'model_version':args.model_version,
                        'candidate':index+1,'video_path':relative,'selected':False,'parameters':parameters}
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
    print(f'Candidate manifest: {manifest}; select one candidate per task/model before pair construction.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
