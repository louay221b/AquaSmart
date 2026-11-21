import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_cost_benchmark import trial
from baseline_tls import np,tls
class TestBaseline(unittest.TestCase):
    def test_paired(self):
        v={'C000':np.array([.1,.2]),'C001':np.array([.3,-.4])}
        before=(tls._server_worker,tls._client_worker)
        a=trial(2,2,0,'plain_tls',v,30); b=trial(2,2,0,'secure_tls',v,30)
        self.assertEqual(before,(tls._server_worker,tls._client_worker))
        self.assertEqual(a['tls_application_frames'],4);self.assertEqual(b['tls_application_frames'],12)
        self.assertLess(a['tls_application_frame_bytes'],b['tls_application_frame_bytes'])
        self.assertLessEqual(a['max_sum_error'],a['bound']);self.assertLessEqual(b['max_sum_error'],b['bound'])
if __name__=='__main__':unittest.main()
