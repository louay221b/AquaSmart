import sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_personalized import experiment, adapt, r

class TestPersonalized(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42); self.S={}
        for i in range(3):
            x=rng.normal(size=(45,6)); y=.5*x[:,0]+i+2
            self.S[str(i)]=((x[:30],y[:30]),(x[30:],y[30:]),None)
        self.opts=dict(rounds_grid=(2,4),lr_grid=(.05,),epochs_grid=(1,),alpha_grid=(.1,),personal_epochs=(0,1,3),personal_lrs=(.01,))
    def test_zero_epochs_preserves_model(self):
        w=np.arange(7,dtype=float); models=adapt(w,self.S,.1,0,0,.01)
        for a in models.values(): np.testing.assert_array_equal(a,w); self.assertIsNot(a,w)
    def test_masked_equivalence_and_no_test(self):
        x=experiment(self.S,0,**self.opts)
        self.assertLess(x['global_max_param_diff'],1e-8)
        self.assertLess(x['personal_max_param_diff'],1e-8)
        self.assertEqual(x['cost_model_only']['personalization_extra_communication_bytes'],0)
        self.assertLessEqual(x['validation']['personalized_fedavg']['micro'],x['validation']['fedavg_tuned']['micro']+1e-12)
    def test_repeatable_and_macro_selection(self):
        a=experiment(self.S,1,selection='macro',**self.opts); b=experiment(self.S,1,selection='macro',**self.opts)
        self.assertEqual(a,b)
        self.assertLessEqual(a['validation']['personalized_fedavg']['macro'],a['validation']['fedavg_tuned']['macro']+1e-12)

if __name__=='__main__': unittest.main()
