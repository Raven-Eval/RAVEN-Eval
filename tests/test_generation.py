import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from raven_eval import generate, select
from raven_eval.anchors import select as choose_anchors
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class GenerationTests(unittest.TestCase):
    def test_generation_excludes_private_rubric(self):
        observed=[]
        def adapter(**request):
            observed.append(request)
            self.assertNotIn('rubric',request['task'])
            output=request['output_dir']/'video.mp4';output.write_bytes(b'fixture');return output
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generate.importlib,'import_module',return_value=types.SimpleNamespace(generate=adapter)):
                with contextlib.redirect_stdout(io.StringIO()):
                    generate.main(['--root',str(ROOT),'--task-type','t2v','--task-id','T2V_L1_D1_K1_0006','--model','test','--model-version','test-v1','--provider','test:generate','--output-dir',tmp])
            rows=[json.loads(line) for line in (Path(tmp)/'candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3);self.assertEqual(len(observed),3)
            self.assertTrue(all(not r['selected'] for r in rows))

    def test_candidate_selection_uses_resolved_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);candidate=tmp/'video.mp4';candidate.write_bytes(b'fixture')
            manifest=tmp/'candidates.jsonl'
            rows=[{'task_type':'t2v','task_id':'T2V_L1_D1_K1_0006','model':'test','model_version':'v1','candidate':i+1,'video_path':str(candidate),'selected':False} for i in range(3)]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            def offline_judge(argv):
                path=Path(argv[argv.index('--task-json')+1]);data=json.loads(path.read_text())
                for t in data['pairwise_comparison_triples']:
                    t['candidate_selector_evaluation']={'output_text':'Better'}
                    t['candidate_selector_reversed_evaluation']={'output_text':'Worse'}
                path.write_text(json.dumps(data));return 0
            with patch.object(select.judge,'main',side_effect=offline_judge):
                with contextlib.redirect_stdout(io.StringIO()):
                    select.main(['--root',str(ROOT),'--candidates',str(manifest),'--output',str(tmp/'selected.jsonl'),'--work-dir',str(tmp/'selection'),'--model','offline-selector','--api-url','https://example.invalid/responses'])
            selected=json.loads((tmp/'selected.jsonl').read_text())
            self.assertTrue(selected['selected']);self.assertEqual(selected['candidate'],1)

    def test_five_tier_anchor_selection(self):
        rng=np.random.default_rng(0)
        joint={(f't{t}',f'm{m}'):float(rng.normal()) for t in range(12) for m in range(15)}
        judges=[{key:value+float(rng.normal(scale=.03)) for key,value in joint.items()} for _ in range(3)]
        anchors,metadata=choose_anchors(joint,judges)
        self.assertEqual(len(set(anchors)),10)
        self.assertTrue(all(len(tier['anchors'])==2 for tier in metadata['tiers']))


if __name__=='__main__':unittest.main()
