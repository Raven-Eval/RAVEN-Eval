import argparse
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raven_eval import judge
from raven_eval.pairs import build

ROOT=Path(__file__).resolve().parents[1]


class PairPolicyTests(unittest.TestCase):
    def test_complete_until_25_then_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task_id='test';folder=root/'data/tasks/t2v';folder.mkdir(parents=True)
            (folder/'test.json').write_text(json.dumps({'task_id':task_id,'task_type':'t2v','task_version':'test','references':{},'prompt':'demo','rubric':'demo'}))
            video=root/'video.mp4';video.write_bytes(b'fixture')
            models=[f'old-{i}' for i in range(10)]+['new']
            manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps({'task_type':'t2v','task_id':task_id,'model':m,'video_path':str(video),'selected':True})+'\n' for m in models))
            full=build(root,manifest,25)
            self.assertEqual(len(full[0][2]['pairwise_comparison_triples']),55)
            with self.assertRaisesRegex(ValueError,'Above 25'):
                build(root,manifest,26)
            sparse=build(root,manifest,26,{'t2v':models[:-1]},['new'])
            self.assertEqual(len(sparse[0][2]['pairwise_comparison_triples']),10)
            for triple in sparse[0][2]['pairwise_comparison_triples']:
                self.assertIn('new',[triple['generated_video_a']['model'],triple['generated_video_b']['model']])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        cls.video=cls.folder/'video.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=red:s=64x48:r=8:d=2','-c:v','libx264',str(cls.video)],check=True)
        cls.frame=cls.folder/'reference.png'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(cls.video),'-frames:v','1',str(cls.frame)],check=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_task_payloads_hidden_and_uniform8(self):
        judge.ROOT=ROOT;judge.SAMPLING_MODE='uniform';judge.FRAME_COUNT=8
        judge.FRAME_CACHE_DIR=self.folder/'frames'
        judge.TASK_SPECS={k:{'reference_mode':v} for k,v in [('t2v','none'),('keyframe','first'),('first_last','first_last')]}
        triple={'generated_video_a':{'model':'SecretGeneratorA','video_path':str(self.video)},'generated_video_b':{'model':'SecretGeneratorB','video_path':str(self.video)},'prompt':'Show a red scene.','scoring_criteria':'The scene is red.'}
        args=argparse.Namespace(show_model_names=False,image_detail='high',model='test-judge',max_output_tokens=128,reasoning_effort='default')
        data={'reference_first_frame_path':str(self.frame),'reference_last_frame_path':str(self.frame)}
        for task_type,expected in [('t2v',16),('keyframe',17),('first_last',18)]:
            payload=judge.build_payload(task_type,data,triple,'Follow the rubric.',args)
            text=json.dumps(payload)
            self.assertNotIn('SecretGeneratorA',text);self.assertNotIn('SecretGeneratorB',text)
            content=payload['input'][0]['content']
            self.assertEqual(sum(c['type']=='input_image' for c in content),expected)
        args.show_model_names=True
        self.assertIn('SecretGeneratorA',judge.build_instruction(triple,'Follow the rubric.',args))

    def test_readonly_preview_and_bidirectional_write(self):
        pairdir=self.folder/'pairs';pairdir.mkdir(exist_ok=True)
        path=pairdir/'test_pairwise_triples.json'
        path.write_text(json.dumps({'task_id':'test','pairwise_comparison_triples':[{'triple_id':'ab','generated_video_a':{'model':'HiddenA','video_path':str(self.video)},'generated_video_b':{'model':'HiddenB','video_path':str(self.video)},'prompt':'Red.','scoring_criteria':'Red scene.'}]}))
        argv=['--root',str(ROOT),'--task-types','t2v','--pair-dir',str(pairdir),'--model','test-judge','--api-url','https://example.invalid/v1/responses','--allow-no-api-key','--state-json',str(self.folder/'state.json'),'--summary-json',str(self.folder/'summary.json'),'--log-dir',str(self.folder/'logs'),'--frame-cache-dir',str(self.folder/'frames')]
        before={p.relative_to(self.folder):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(judge.main(argv+['--dry-run']),0)
        after={p.relative_to(self.folder):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        with patch.object(judge,'call_openai',return_value={'output_text':'Better'}) as request:
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(judge.main(argv),0)
            self.assertEqual(request.call_count,2)
        saved=json.loads(path.read_text())['pairwise_comparison_triples'][0]
        for direction in ['evaluation','reversed_evaluation']:
            result=saved['test_judge_blind_uniform8_v1_'+direction]
            self.assertEqual(result['output_text'],'Better')
            self.assertFalse(result['judge_config']['show_model_names'])


if __name__=='__main__':unittest.main()
