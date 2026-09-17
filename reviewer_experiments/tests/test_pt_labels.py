import unittest

import pandas as pd

from reviewer_experiments.pt_labels import audit_pt_labels, stored_label_value


class PTLabelsTest(unittest.TestCase):
    def test_schemes_and_rejection_of_inconsistent_records(self):
        frame = pd.DataFrame(dict(treatment=['mock', 'infected', 'infected', 'infected'],
                                  dpi=[2, 2, 4, 6], stored_label=[0, 1, 2, 3]))
        self.assertEqual(audit_pt_labels(frame)['encoding'], 'mock0_infected123')
        frame.stored_label = [3, 0, 1, 2]
        self.assertEqual(audit_pt_labels(frame)['encoding'], 'infected012_mock3')
        frame.stored_label = [0, 0, 1, 2]
        with self.assertRaisesRegex(ValueError, 'Mixed or unknown'):
            audit_pt_labels(frame)
        frame.stored_label = [3, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, 'Mixed or unknown'):
            audit_pt_labels(frame)

    def test_labels_must_be_finite_integer_scalars(self):
        for value in [float('nan'), float('inf'), 1.5, [1, 2], None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                stored_label_value(value)
        self.assertEqual(stored_label_value(0), 0)


if __name__ == '__main__':
    unittest.main()
