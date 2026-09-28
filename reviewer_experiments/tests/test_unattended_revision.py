import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_unattended_revision import choose_cohort, main, paper_minimal


class UnattendedTest(unittest.TestCase):
    def test_minimal_runs_only_psnet_with_spectra_between_tasks(self):
        with patch('run_unattended_revision.execute') as run:
            stages = []
            paper_minimal(Path('tasks'), Path('selected.csv'), Path('out'), 'cuda:0',
                          lambda name, **kwargs: stages.append(name), {'cohort':'recovered'})
        self.assertEqual(stages, ['training_all_dpi','summary_all_dpi','spectral_analysis',
                                  'training_dpi_2','summary_dpi_2'])
        calls = run.call_args_list
        self.assertEqual(len(calls), 5)
        for index, task in ((0, 'all_dpi'), (3, 'dpi_2')):
            args = calls[index].args
            self.assertEqual(args[args.index('--models')+1], 'psnet_full')
            self.assertEqual(args[args.index('--tasks')+1], task)
            self.assertEqual(args[args.index('--seed')+1], '157')
            self.assertNotIn('--top-k-bands', args)
        self.assertEqual(calls[2].args[0], 'reviewer_experiments.spectral_reanalysis')
        self.assertEqual(calls[2].args[2], Path('selected.csv'))
        self.assertTrue(calls[2].kwargs['module'])

    def test_mismatch_falls_back_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('recover_revision_data.merge', side_effect=ValueError('numeric mismatch')), patch('reviewer_experiments.core.prepare_tasks') as prepare:
                tasks, decision=choose_cohort(root/'base.csv',root/'new',root/'rgb',root,'prefer-recovered')
            self.assertEqual(tasks,root/'archived_tasks')
            self.assertFalse(decision['recovered_included'])
            self.assertIn('numeric mismatch',decision['reason'])
            prepare.assert_called_once()

    def test_strict_mode_never_falls_back(self):
        with patch('recover_revision_data.merge',side_effect=ValueError('mismatch')), patch('reviewer_experiments.core.prepare_tasks') as prepare:
            with self.assertRaisesRegex(ValueError,'mismatch'):
                choose_cohort(Path('base'),Path('new'),Path('rgb'),Path('out'),'require-recovered')
            prepare.assert_not_called()

    def test_passed_merge_uses_recovered_tasks(self):
        with patch('recover_revision_data.merge'),patch('reviewer_experiments.core.prepare_tasks') as prepare:
            tasks,decision=choose_cohort(Path('base'),Path('new'),Path('rgb'),Path('out'),'prefer-recovered')
        self.assertEqual(tasks,Path('out/recovered_cohort/tasks'))
        self.assertTrue(decision['recovered_included'])
        prepare.assert_not_called()

    def test_failed_job_writes_status_but_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'job'
            with patch('sys.argv',['runner','--split-dir',tmp,'--output',str(output)]),patch('run_unattended_revision.job',side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError,'test failure'): main()
            self.assertEqual(json.loads((output/'STATUS.json').read_text())['state'],'failed')
            self.assertTrue((output/'FAILED.txt').is_file())
            self.assertFalse((output/'DONE.txt').exists())

    def test_success_marks_actual_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'job'
            with patch('sys.argv',['runner','--split-dir',tmp,'--output',str(output)]),patch('run_unattended_revision.job',return_value={'cohort':'archived','recovered_included':False}):
                main()
            status=json.loads((output/'STATUS.json').read_text())
            self.assertEqual(status['state'],'completed')
            self.assertFalse(status['recovered_included'])
            self.assertTrue((output/'DONE.txt').exists())


if __name__=='__main__':unittest.main()
