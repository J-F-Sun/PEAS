"""Validate checkpoint selection without a costly full training run."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch
from peas.ablations import selection
from peas.common.paths import repository_root

class SelectionTests(TestCase):
    def test_earliest_minimum_and_rng_preservation(self):
        with np.load(repository_root()/'data/validation/canonical.npz') as d:
            ref=d['reference'].copy()
        class Model:
            factor=0.
            def predict(self,times,points):
                torch.rand(7)
                return ref*(1+self.factor)
        model=Model(); saved=[]
        def save(path,model,state,provenance):
            if Path(path).name=='selected.pt':saved.append(state['epoch'])
        with TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            def train(config,**kwargs):
                for epoch,factor in [(0,.5),(3500,.2),(3750,.1),(4000,.1),(9000,.15),(9000,.15)]:
                    model.factor=factor;before=torch.get_rng_state().clone()
                    selection.training.save_checkpoint(path,model,{'epoch':epoch}, {})
                    self.assertTrue(torch.equal(before,torch.get_rng_state()))
                return {'phase':'complete'}
            original=selection.training.save_checkpoint
            with patch.object(selection.training,'save_checkpoint',save),patch.object(selection.training,'train',train):
                selection.train_selected({'scenario':'canonical','ablation':{'variant':'no_features'}},output_root=folder)
                self.assertIs(selection.training.save_checkpoint,save)
            self.assertIs(selection.training.save_checkpoint,original)
            audit=json.loads((Path(folder)/'validation.json').read_text())
            self.assertEqual(saved,[3500,3750]);self.assertEqual(audit['best_epoch'],3750)
            self.assertEqual(len(audit['scores']),4)
