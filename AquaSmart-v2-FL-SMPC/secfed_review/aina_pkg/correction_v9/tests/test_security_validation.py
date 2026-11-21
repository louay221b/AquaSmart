"""Security checks against V7 implementation, without changing V7/V8."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_security_validation as sec


class TestSecurityValidation(unittest.TestCase):
    def test_signature_tamper_and_old_session(self):
        data = sec.inputs(seed=42, n=3, dim=3)
        for kind in ('signature', 'masked'):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(sec.SecureAggregationError,
                                            'Invalid authenticated client signature'):
                    sec._direct_signature_forgery(data, kind)
        with self.assertRaises(sec.SecureAggregationError):
            sec._replay_old_session(data)

    def test_network_security_scenarios(self):
        rows = sec.suite(seed=0, n=3, dim=3, timeout=12)
        self.assertEqual(len(rows), 8)
        for row in rows:
            with self.subTest(scenario=row['scenario']):
                self.assertTrue(row['passed'], row)
        successes = [x for x in rows if x['observed'] == 'accepted']
        self.assertEqual({x['scenario'] for x in successes},
                         {'normal_mtls', 'dropout_before_upload'})
        self.assertTrue(all(x['details']['max_sum_error'] <=
                            x['details']['rounding_bound'] for x in successes))


if __name__ == '__main__':
    unittest.main()
