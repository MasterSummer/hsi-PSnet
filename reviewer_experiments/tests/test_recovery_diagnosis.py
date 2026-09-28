import unittest
import numpy as np
from diagnose_recovery import analyze


class RecoveryDiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.RandomState(157).uniform(.1, 5, size=(4, 5, 6)).astype(np.float32)
        self.chw = np.moveaxis(self.raw, -1, 0).copy()

    def test_identical_cube_detected_without_mutation(self):
        raw, old = self.raw.copy(), self.chw.copy()
        report = analyze(old, raw)
        self.assertIn('MAT_HWC_to_CHW', report['exact_matching_hypotheses'])
        np.testing.assert_array_equal(old, self.chw)
        np.testing.assert_array_equal(raw, self.raw)
        self.assertFalse(report['merge_authorized'])

    def test_masking_is_only_reported_as_a_clue(self):
        old = self.chw.copy(); old[:, 0, 0] = 0
        report = analyze(old, self.raw)
        self.assertEqual(report['spatial_zero_pattern']['all_bands_zero_pixels'], 1)
        self.assertEqual(report['exact_matching_hypotheses'], [])
        self.assertIn('MAT_HWC_to_CHW', report['nonzero_preserving_hypotheses'])
        self.assertFalse(report['merge_authorized'])

    def test_spatial_flip_is_detected(self):
        report = analyze(self.chw[:, ::-1, :], self.raw)
        self.assertIn('reverse_CHW_axes_1', report['exact_matching_hypotheses'])

    def test_changed_values_cannot_pass_mask_or_layout_check(self):
        report = analyze(self.chw * 2 + .3, self.raw)
        self.assertEqual(report['exact_matching_hypotheses'], [])
        self.assertEqual(report['nonzero_preserving_hypotheses'], [])

    def test_nonfinite_rejected(self):
        self.chw[0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            analyze(self.chw, self.raw)


if __name__ == '__main__':
    unittest.main()
