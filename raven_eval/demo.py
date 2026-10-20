"""Offline pipeline demo using synthetic videos and synthetic judge labels."""
import argparse
import json
import subprocess
from pathlib import Path
from . import pairs, ranking


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('runs/demo'))
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1];out=args.output_dir.resolve()
    if out.exists():
        parser.error('Use an empty demo output directory')
    out.mkdir(parents=True)
    models=['Demo-A','Demo-B','Demo-C'];videos={}
    for model,color in zip(models,['red','green','blue']):
        path=out/(model+'.mp4')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',f'color=c={color}:s=96x64:r=8:d=2','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
        videos[model]=str(path)
    manifest=out/'selected_videos.jsonl';rows=[]
    for kind in ['t2v','flt','kft']:
        task=sorted((root/'data/tasks'/kind).glob('*.json'))[0]
        rows.extend({'task_type':kind,'task_id':task.stem,'model':m,'model_version':'synthetic-demo-v1','video_path':videos[m],'selected':True} for m in models)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    pairs.main(['--root',str(root),'--videos',str(manifest),'--output-dir',str(out/'pairs')])
    bases=['demo_judge_1','demo_judge_2','demo_judge_3']
    for p in (out/'pairs').rglob('*_pairwise_triples.json'):
        data=json.loads(p.read_text())
        for triple in data['pairwise_comparison_triples']:
            for base in bases:
                triple[base+'_evaluation']={'output_text':'Better','synthetic':True}
                triple[base+'_reversed_evaluation']={'output_text':'Worse','synthetic':True}
        p.write_text(json.dumps(data,indent=2)+'\n')
    command=[]
    for kind in ['t2v','flt','kft']:command.extend(['--input-dir',str(out/'pairs'/kind)])
    for base in bases:command.extend(['--judge-base',base])
    ranking.main(command+['--output-dir',str(out/'synthetic_results'),'--bootstrap-rounds','3','--bootstrap-pairs','30'])
    (out/'SYNTHETIC_DEMO_ONLY.txt').write_text('Synthetic videos and fabricated preferences for software verification only. Not benchmark results. No API requests were made.\n')
    print(f'Offline demo complete: {out}; these are synthetic results, not published model scores.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
