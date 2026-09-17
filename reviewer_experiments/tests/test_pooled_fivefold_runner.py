import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_controlled_ablations as runner
from reviewer_experiments.core import assign_plant_folds
from reviewer_experiments.metrics import plant_predictions


class PooledFiveFoldRunnerTest(unittest.TestCase):
    def test_default_invokes_one_pooled_task_one_seed_and_five_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root/'tasks'
            tasks.mkdir()
            frame = pd.DataFrame([dict(plant_id=f'p{i}', fold=i+1) for i in range(5)])
            for task in ['all_dpi']:
                frame.to_csv(tasks/f'{task}.csv', index=False)
            args = ['run_controlled_ablations.py', '--task-dir', str(tasks), '--output', str(root/'out')]
            with patch('sys.argv', args), patch.object(runner, 'build_model'), \
                 patch.object(runner, 'trainable_parameters', return_value=1), \
                 patch.object(runner, 'run_matrix', return_value=[]) as train, \
                 patch.object(runner, 'verify_predictions'), patch.object(runner, 'evaluate_predictions'), \
                 patch.object(runner, 'label_oof_evaluation') as label:
                runner.main()
            self.assertEqual(train.call_args.args[4], [157])
            self.assertEqual(train.call_args.args[5], [1, 2, 3, 4, 5])
            self.assertEqual(train.call_args.args[2], ['all_dpi'])
            self.assertEqual(len(train.call_args.args[2])*len(train.call_args.args[3])*len(train.call_args.args[5]), 25)
            manifest = json.loads((root/'out/experiment_manifest.json').read_text())
            self.assertEqual(manifest['identity']['folds'], [1, 2, 3, 4, 5])
            self.assertEqual(manifest['identity']['evaluation_design'], 'five_fold_plant_oof_single_seed')
            label.assert_called_once_with((root/'out/evaluation').resolve(), 157)

    def test_single_run_sd_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([dict(model='psnet_full', seeds=1, balanced_accuracy_mean=.75,
                               balanced_accuracy_std=0.)]).to_csv(root/'metrics_summary.csv', index=False)
            pd.DataFrame([dict(model='psnet_full', seed=157, balanced_accuracy=.75)]).to_csv(root/'metrics_by_seed.csv', index=False)
            runner.label_oof_evaluation(root, 157)
            result = pd.read_csv(root/'metrics_summary.csv')
            self.assertTrue(np.isnan(result.balanced_accuracy_std.iloc[0]))
            self.assertTrue((root/'oof_metrics.csv').exists())
            note = json.loads((root/'evaluation_design.json').read_text())
            self.assertEqual(note['training_runs_per_model_task'], 5)

    def test_incomplete_fivefold_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root/'tasks'
            task_dir.mkdir()
            rows = [dict(sample_id=f's{i}', plant_id=f'p{i}', label=i%2, fold=i) for i in range(1, 6)]
            pd.DataFrame(rows).to_csv(task_dir/'all_dpi.csv', index=False)
            paths = []
            for record in rows:
                path = root/'runs/psnet_full/all_dpi/seed_157'/f"fold_{record['fold']}"/'predictions.csv'
                path.parent.mkdir(parents=True)
                pd.DataFrame([{**record, 'model': 'psnet_full', 'task': 'all_dpi', 'seed': 157, 'prob_infected': .6}]).to_csv(path, index=False)
                path.with_name('best.pt').touch()
                paths.append(path)
            runner.verify_predictions(paths, task_dir)
            with self.assertRaisesRegex(ValueError, 'all five folds'):
                runner.verify_predictions(paths[:-1], task_dir)

    def test_dates_and_leaves_stay_together_and_aggregate_per_plant(self):
        rows = []
        for label in (0, 1):
            for plant in range(10):
                for dpi in (2, 4, 6):
                    for leaf in (3, 4):
                        rows.append(dict(plant_id=f'{label}_{plant}', label=label,
                                         treatment='infected' if label else 'mock', dpi=dpi, leaf=leaf,
                                         model='psnet_full', task='all_dpi', seed=157, prob_infected=dpi/10))
        frame = assign_plant_folds(pd.DataFrame(rows), n_splits=5, seed=157)
        self.assertTrue(frame.groupby('plant_id').fold.nunique().eq(1).all())
        self.assertEqual(set(frame.fold), {1, 2, 3, 4, 5})
        aggregated = plant_predictions(frame)
        self.assertEqual(len(aggregated), 20)
        np.testing.assert_allclose(aggregated.prob_infected, .4)


if __name__ == '__main__':
    unittest.main()
