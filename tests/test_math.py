"""Offline mathematical checks: python -m unittest discover -s tests -p test_math.py -v"""
import unittest
import numpy as np
from math_core import (softmax, probability_metrics, attention_reference, rmsnorm,
                       aggregate_heads, affine_fit, affine_apply, toy_attention, toy_training)


class MathTests(unittest.TestCase):
    def test_softmax_normalizes(self):
        for t in [0,0.01,0.3,1,3]: self.assertAlmostEqual(softmax([-2,3,1],t).sum(),1,14)
    def test_stability(self): np.testing.assert_allclose(softmax([10000,10001]),softmax([0,1]))
    def test_shift_invariance(self): np.testing.assert_allclose(softmax([2,3,4]),softmax([-3,-2,-1]))
    def test_temperature_ranking(self):
        for t in [.01,.3,1,3]: self.assertGreater(softmax([0,1,2],t)[2],softmax([0,1,2],t)[1])
    def test_greedy_tie(self): np.testing.assert_array_equal(softmax([3,3,1],0),[1,0,0])
    def test_invalid_softmax(self):
        for x,t in [([],1),([float('nan')],1),([1],-1),([1],float('inf'))]:
            with self.assertRaises(ValueError): softmax(x,t)
    def test_odds_formula(self):
        p=softmax([10.25,8.50],.3)
        self.assertAlmostEqual(p[0]/p[1],np.exp(1.75/.3),10)
    def test_uniform_vocabulary(self): self.assertAlmostEqual(100/49152,.0020345052083333335)
    def test_rounding_logits_changes_distribution(self):
        self.assertFalse(np.array_equal(softmax([1.0004,1.0001],.01),softmax([1,1],.01)))
    def test_tv_js_identity(self): self.assertEqual(probability_metrics([.2,.8],[.2,.8]),{'tv':0.,'js_nats':0.})
    def test_tv_js_disjoint(self):
        m=probability_metrics([1,0],[0,1]); self.assertEqual(m['tv'],1);self.assertAlmostEqual(m['js_nats'],np.log(2))
    def test_invalid_metrics(self):
        with self.assertRaises(ValueError):probability_metrics([.2,.2],[.5,.5])
    def test_attention_mask(self):
        t=toy_attention();w=np.array(t['weights']);self.assertEqual(np.triu(w,1).sum(),0);np.testing.assert_allclose(w.sum(-1),1)
    def test_toy_first_position(self): np.testing.assert_array_equal(toy_attention()['output'][0],[2,0])
    def test_attention_weighted_sum(self):
        t=toy_attention();np.testing.assert_allclose(np.array(t['weights'])@np.array(t['v']),t['output'])
    def test_score_scale(self):
        t=toy_attention();np.testing.assert_allclose(t['scores'],np.array(t['q'])@np.array(t['k']).T/np.sqrt(2))
    def test_mean_is_distribution_max_is_not(self):
        h=np.array([[[1,0],[.9,.1]],[[1,0],[.1,.9]]]);np.testing.assert_allclose(aggregate_heads(h,'mean').sum(-1),1);self.assertAlmostEqual(aggregate_heads(h,'max')[1].sum(),1.8)
    def test_rmsnorm_learned_scale(self):
        x=np.array([3.,4.]);np.testing.assert_allclose(rmsnorm(x,[2,1]),x/np.sqrt(12.5+1e-5)*[2,1])
    def test_affine_identity(self):
        x=np.arange(12.).reshape(4,3);np.testing.assert_allclose(affine_apply(x,affine_fit(x,x)),x)
    def test_affine_intercept(self):
        x=np.arange(12.).reshape(4,3);fit=affine_fit(x,x+[2,-1,3]);np.testing.assert_allclose(affine_apply([1,4,9],fit),[3,3,12])
    def test_affine_dual_matches_primal(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(8,3));y=rng.normal(size=(8,3));fit=affine_fit(x,y,.3);xc=x-x.mean(0);delta=y-x;coef=np.linalg.solve(xc.T@xc+fit['effective_penalty']*np.eye(3),xc.T@(delta-delta.mean(0)));h=np.array([.2,.8,-1]);expected=h+delta.mean(0)+(h-x.mean(0))@coef;np.testing.assert_allclose(affine_apply(h,fit),expected)
    def test_training_loss_decreases(self):
        for r in toy_training(10)['steps']:self.assertLess(r['next_loss'],r['loss'])
    def test_gradient_finite_difference(self):
        t=toy_training();r=t['steps'][0];w=np.array(r['weights_before']);x=np.array(t['input']);target=t['target_id'];g=np.array(r['gradient']);epsilon=1e-6
        for i in range(3):
            for j in range(2):
                delta=np.zeros_like(w);delta[i,j]=epsilon
                fd=(-np.log(softmax((w+delta)@x)[target])+np.log(softmax((w-delta)@x)[target]))/(2*epsilon)
                self.assertAlmostEqual(fd,g[i,j],8)
    def test_optimizer_update(self):
        t=toy_training();r=t['steps'][0];np.testing.assert_allclose(r['weights_after'],np.array(r['weights_before'])-t['learning_rate']*np.array(r['gradient']))
    def test_parameter_accounting(self):
        self.assertEqual(49152*576+30*(884736+2654208+1152)+576,134515008)
    def test_cache_accounting(self): self.assertEqual(30*3*64*2*4,46080)


if __name__=='__main__':unittest.main()
