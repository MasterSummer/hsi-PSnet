import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from run_revision import PRESETS, commands
from reviewer_experiments.models import MODEL_NAMES
from summarize_revision import summarize


class RevisionDeliveryTest(unittest.TestCase):
    def test_presets_have_supported_models_and_expected_budgets(self):
        budgets={'paper-minimal':10,'main':480,'priority':45,'recovered':135,'controls':25,'full':720}
        for name,(tasks,models,seeds) in PRESETS.items():
            self.assertLessEqual(set(models.split(',')),set(MODEL_NAMES))
            self.assertEqual(len(tasks.split(','))*len(models.split(','))*len(seeds)*5,budgets[name])
        a=argparse.Namespace(preset='controls',seeds=None,task_dir=None,
            split_dir=Path('/tmp/split with spaces'),rgb_root=Path('/tmp/rgb'),output=Path('/tmp/out'),
            device='cpu',epochs=1,patience=1,batch_size=2,workers=0)
        cmds=list(commands(a));self.assertEqual(len(cmds),1)
        self.assertIn('/tmp/split with spaces',cmds[0])

    def make_run(self,root,seed=157):
        run=root/f'seed_{seed}';run.mkdir(parents=True)
        identity=dict(models=['psnet_full'],tasks=['dpi_2'],seeds=[seed],source_sha256={},
            training={'dim':64},band_spec='all',top_k_bands=None)
        (run/'experiment_manifest.json').write_text(json.dumps(dict(identity=identity,task_sha256={'dpi_2':'fixture'})))
        rows=[]
        for fold in range(1,6):
            for label in (0,1):
                rows.append(dict(sample_id=f's{fold}_{label}',plant_id=f'p{fold}_{label}',
                    label=label,fold=fold,model='psnet_full',task='dpi_2',seed=seed,prob_infected=.9 if label else .1))
        frame=pd.DataFrame(rows)
        (run/'task_snapshot').mkdir()
        frame.to_csv(run/'task_snapshot/dpi_2.csv',index=False)
        for fold in range(1,6):
            target=run/'runs/psnet_full/dpi_2'/f'seed_{seed}'/f'fold_{fold}'
            target.mkdir(parents=True)
            frame[frame.fold==fold].to_csv(target/'predictions.csv',index=False)
            (target/'best.pt').touch()
        return run

    def test_summary_checks_all_folds_and_preserves_single_seed_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_run(root/'runs')
            summarize(root/'runs',root/'summary')
            row=pd.read_csv(root/'summary/summary.csv').iloc[0]
            self.assertEqual(row.balanced_accuracy_mean,1)
            self.assertTrue(np.isnan(row.balanced_accuracy_sd))
            self.assertEqual(row.n_seeds,1)
            (root/'runs/seed_157/runs/psnet_full/dpi_2/seed_157/fold_5/predictions.csv').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                summarize(root/'runs',root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_different_protocols_not_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_run(root/'runs');run=self.make_run(root/'runs',257)
            path=run/'experiment_manifest.json';m=json.loads(path.read_text())
            m['identity']['training']['dim']=128;path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Cannot pool'):
                summarize(root/'runs',root/'summary')


if __name__=='__main__':unittest.main()
