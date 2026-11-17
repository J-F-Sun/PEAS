import unittest
import numpy as np
from peas.evaluation.archive import errors,read,SCENES,METHODS

class MetricTests(unittest.TestCase):
    def test_run_macro_order(self):
        ref=np.array([[1.,2.],[1.,2.]])
        pred=np.stack([ref*np.array([[1.1],[1.3]]),ref*np.array([[1.3],[1.1]])])
        e=errors(pred,ref)
        np.testing.assert_allclose(e,[[10,30],[30,10]])
        self.assertAlmostEqual(e.mean(1).std(ddof=1),0.)
        self.assertGreater(e.std(0,ddof=1).mean(),0.)
    def test_alignment(self):
        for scene in SCENES:
            for method in METHODS:
                d=read(scene,method);self.assertEqual(d['prediction'].shape[1],20)
                self.assertEqual((d['times'][0],d['times'][-1]),(0,20))
    def test_mc_mean(self):
        for scene in SCENES:
            d=read(scene,'AMS (mean of 128)');self.assertEqual(len(d['run_predictions']),128)
            np.testing.assert_allclose(d['run_predictions'].mean(0),d['prediction'][0],rtol=1e-12,atol=1e-15)
    def test_zero_reference_rejected(self):
        with self.assertRaises(ValueError):errors(np.zeros((1,2,3)),np.zeros((2,3)))

class InferenceTests(unittest.TestCase):
    def test_one_checkpoint_per_scene(self):
        import torch
        from peas.common.paths import repository_root
        from peas.common.checkpoints import load_model
        torch.set_num_threads(1)
        for scene in SCENES:
            d=read(scene,'PEAS');seed=int(d['seeds'][0])
            model,_=load_model(repository_root()/f'checkpoints/peas/{scene}/seed_{seed}.pt',device='cpu')
            idx=np.array([0,len(d['times'])//2,len(d['times'])-1])
            p=model.predict(d['times'][idx],d['samples'][:2])
            np.testing.assert_allclose(p,d['prediction'][0,:2][:,idx],rtol=2e-4,atol=2e-6)

if __name__=='__main__':unittest.main()
