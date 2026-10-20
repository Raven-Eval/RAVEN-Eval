"""Build task-wise complete graphs or selected-anchor insertion graphs."""
import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path


def build(root, manifest, library_count=None, anchors=None, new_models=()):
    grouped = defaultdict(dict)
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('selected') is not True:
            continue
        key = (row['task_type'], row['task_id'])
        if row['model'] in grouped[key]:
            raise ValueError(f"Multiple selected videos: {key} / {row['model']}")
        video = Path(row['video_path'])
        if not (video if video.is_absolute() else root/video).is_file():
            raise ValueError(f"Missing selected video for {key} / {row['model']}")
        grouped[key][row['model']] = row
    if not grouped:
        raise ValueError('No selected videos in manifest')
    models = {m for videos in grouped.values() for m in videos}
    total = library_count or len(models)
    if total < len(models):
        raise ValueError('Library size cannot be smaller than the supplied model count')
    if total > 25 and (not anchors or not new_models):
        raise ValueError('Above 25 models, provide a frozen --anchors file and --new-model IDs')
    outputs = []
    for (kind, task_id), videos in sorted(grouped.items()):
        task = json.loads((root/'data/tasks'/kind/(task_id+'.json')).read_text())
        comparisons = list(itertools.combinations(sorted(videos), 2))
        if total > 25:
            selected = anchors.get(kind, [])
            if len(set(selected)) != 10:
                raise ValueError(f'{kind}: require ten distinct fixed anchors')
            if set(selected) & set(new_models):
                raise ValueError('A new model cannot be its own anchor')
            for model in new_models:
                if model in videos and not set(selected).issubset(videos):
                    raise ValueError(f'Missing anchor videos for {task_id}')
            comparisons = [(a,b) for a,b in comparisons if (a in new_models and b in selected) or (b in new_models and a in selected)]
        triples = []
        for a,b in comparisons:
            identifier = hashlib.sha256(json.dumps([task_id,a,b],ensure_ascii=False).encode()).hexdigest()[:20]
            triples.append({'triple_id':identifier,'task_id':task_id,
                'generated_video_a':{'model':a,'model_version':videos[a].get('model_version'),'video_path':videos[a]['video_path']},
                'generated_video_b':{'model':b,'model_version':videos[b].get('model_version'),'video_path':videos[b]['video_path']},
                'prompt':task['prompt'],'scoring_criteria':task['rubric']})
        if not triples:
            continue
        outputs.append((kind, task_id, {'task_id':task_id,'task_type':kind,
            'task_version':task['task_version'],'graph_policy':'complete' if total<=25 else 'anchor_insertion',
            'library_model_count':total,'reference_video_path':None,
            'reference_first_frame_path':task['references'].get('first'),
            'reference_last_frame_path':task['references'].get('last'),
            'pairwise_comparison_triples':triples}))
    return outputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--videos',type=Path,required=True,help='Selected-video JSONL manifest')
    parser.add_argument('--output-dir',type=Path,default=Path('runs/pairs'))
    parser.add_argument('--library-model-count',type=int)
    parser.add_argument('--anchors',type=Path,help='JSON: task type to ten old-model IDs')
    parser.add_argument('--new-model',action='append',default=[])
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    root=args.root.resolve()
    try:
        outputs=build(root,args.videos,args.library_model_count,
                      json.loads(args.anchors.read_text()) if args.anchors else None,args.new_model)
        paths=[(args.output_dir/k/(tid+'_pairwise_triples.json'),payload) for k,tid,payload in outputs]
        if not args.dry_run:
            if any(p.exists() for p,d in paths):
                raise ValueError('Output already exists; use another output directory to preserve judgments')
            for p,data in paths:
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'task_files':len(paths),'pairs':sum(len(d['pairwise_comparison_triples']) for p,d in paths),'dry_run':args.dry_run}))
    except (ValueError,KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
