"""Validate public assets, task hashes, links and accidental secret/path leakage."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    counts=Counter();references=set()
    for row in manifest['tasks']:
        path=ROOT/row['path'];assert path.is_file(),row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
        task=json.loads(path.read_text());counts[task['task_type']]+=1
        assert task['task_id']==row['task_id'] and task['prompt'] and task['rubric']
        for reference in task['references'].values():
            assert not Path(reference).is_absolute() and (ROOT/reference).is_file(),reference
            references.add(reference)
    assert counts=={'t2v':150,'flt':50,'kft':50} and len(references)==150
    for kind,n in [('t2v',20),('flt',14),('kft',20),('overall',20)]:
        with (ROOT/'results/published'/(kind+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==n and [int(r['rank']) for r in rows]==list(range(1,n+1))
        assert [float(r['score']) for r in rows]==sorted([float(r['score']) for r in rows],reverse=True)
    links=0
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('http:','https:','mailto:','#')):continue
            relative=target.split('#')[0]
            linked=path.parent/unquote(relative)
            assert linked.exists(),(str(path.relative_to(ROOT)),target)
            if '#' in target and linked.suffix=='.md':
                anchor=target.split('#',1)[1]
                headings=[]
                for heading in re.findall(r'^#+\s+(.+)$',linked.read_text(),re.M):
                    slug=re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-')
                    headings.append(slug)
                assert anchor in headings,(path.name,target)
            links+=1
    for image in ['teaser.png','framework.png','dataset_statistics.png']:
        assert (ROOT/'docs/assets'/image).stat().st_size>1000
    for row in json.loads((ROOT/'docs/assets/examples/manifest.json').read_text()):
        assert (ROOT/row['video']).is_file() and (ROOT/row['poster']).is_file()
    forbidden=[('/mnt/'+'shared-storage-user/'),('jia'+'ziheng')]
    secret=re.compile(r'\bsk-[A-Za-z0-9_-]{16,}\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    checked=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in {'.git','__pycache__','runs','.venv'} or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts):continue
        assert not path.is_symlink(),path
        assert path.stat().st_size<100*1024*1024,path.name
        if path.suffix in {'.py','.md','.txt','.json','.jsonl','.csv','.toml','.yml','.yaml','.cff'} or path.name.startswith('.env'):
            content=path.read_text(encoding='utf-8-sig')
            assert not any(term in content for term in forbidden),f'Local identity/storage reference: {path.relative_to(ROOT)}'
            assert not secret.search(content),f'Possible embedded credential: {path.relative_to(ROOT)}'
            checked+=1
    print(json.dumps({'tasks':dict(counts),'reference_images':len(references),'local_markdown_links':links,'text_files_scanned':checked,'published_model_rows':74,'result':'passed'},indent=2))


if __name__=='__main__':main()
