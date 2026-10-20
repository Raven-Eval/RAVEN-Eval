import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from raven_eval.ranking import Task, merge, objective, fit, aggregate, bootstrap, load_task, connected


class RankingTests(unittest.TestCase):
    def test_direction_swap(self):
        mass, confident=merge('win','loss')
        np.testing.assert_array_equal(mass,[1,0,0]); self.assertTrue(confident)
        mass, confident=merge('tie','tie')
        np.testing.assert_array_equal(mass,[0,0,1]); self.assertTrue(confident)
        for a,b in [('win','win'),('loss','loss'),('tie','win')]:
            mass, confident=merge(a,b)
            np.testing.assert_array_equal(mass,[.5,.5,0]); self.assertFalse(confident)

    def test_pair_reliability(self):
        triple={'generated_video_a':{'model':'A'},'generated_video_b':{'model':'B'}}
        for base,labels in [('j1',('Better','Worse')),('j2',('Comparable','Comparable')),('j3',('Better','Better'))]:
            triple[base+'_evaluation']={'output_text':labels[0]}
            triple[base+'_reversed_evaluation']={'output_text':labels[1]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.json';path.write_text(json.dumps({'pairwise_comparison_triples':[triple]}))
            task=load_task(path,['j1','j2','j3'])
        # lambda=2/3, average outcome=(1.5,.5,1)/3.
        np.testing.assert_allclose(task.weights[0],np.array([1.5,.5,1])*2/9)
        self.assertEqual(task.confident,[2])

    def test_gradient(self):
        params=np.array([.8,-.4,-.4,.2]);indices=np.array([[0,1],[0,2],[1,2]])
        mass=np.array([[.5,.2,.3],[.6,.1,.0],[.0,.0,1.]])
        actual=objective(params,indices,mass,3)[1]
        expected=approx_derivative(lambda x:objective(x,indices,mass,3)[0],params).ravel()
        np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_centering_and_stability(self):
        task=Task('demo',['A','B','C'],[('A','B'),('A','C'),('B','C')],np.array([[1.,0,0],[1.,0,0],[1.,0,0]]),[3,3,3],3)
        scores,gamma=fit(task)
        self.assertGreater(scores['A'],scores['B']);self.assertGreater(scores['B'],scores['C'])
        self.assertAlmostEqual(sum(scores.values()),0)
        # Extreme logits stay finite (no exp overflow).
        loss,grad=objective(np.array([800.,-800.,0.,0.]),np.array([[0,1],[0,2],[1,2]]),task.weights,3)
        self.assertTrue(np.isfinite(loss));self.assertTrue(np.isfinite(grad).all())

    def test_mean_minus_std_and_bootstrap(self):
        result=aggregate([{'A':1.,'B':0.},{'A':-1.,'B':0.}])
        self.assertEqual(result[0]['model'],'B');self.assertEqual(result[1]['score'],-1.)
        task=Task('demo',['A','B','C'],[('A','B'),('A','C'),('B','C')],np.array([[1.,0,0],[1.,0,0],[1.,0,0]]),[3,3,3],3)
        a=bootstrap([task],3,9,123);b=bootstrap([task],3,9,123)
        self.assertEqual(a,b);self.assertEqual(set(a[0]),set(task.models))
        self.assertFalse(connected(task,np.array([1,0,0])))
        self.assertEqual(set(fit(task,np.array([1,0,0]))[0]),set(task.models))


if __name__=='__main__':unittest.main()
