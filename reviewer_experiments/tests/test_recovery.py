import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat

from recover_revision_data import export_mat, identify, merge


class RecoveryTest(unittest.TestCase):
    def fixture(self, root):
        old, rgb, new = root/'old', root/'rgb', root/'new'
        for p in (old, rgb, new): p.mkdir()
        cube = np.arange(120, dtype=np.float32).reshape(4,5,6)
        rows = []
        for dpi in (2,4,6):
            for treatment in ('Infected','Healthy'):
                stem = f'Plant1_{treatment}_Leaf3_Day{dpi}'
                np.save(old/f'{stem}.npy',cube)
                Image.new('RGB',(8,8)).save(rgb/f'{stem}.jpg')
                rows.append({**identify(stem), 'rgb_path':str(rgb/f'{stem}.jpg'),
                    'hsi_path':str(old/f'{stem}.npy'),'hsi_layout':'HWC',
                    'acquisition_date':f'd{dpi}','imaging_session':f'd{dpi}','symptom_status':'unknown'})
        pd.DataFrame(rows).to_csv(root/'base.csv',index=False)
        np.save(new/'Plant1_Infected_Leaf3_Day2.npy',cube)
        np.save(new/'Plant2_Infected_Leaf3_Day2.npy',cube)
        Image.new('RGB',(8,8)).save(rgb/'Plant2_Infected_Leaf3_Day2.jpg')
        return new,rgb,cube

    def test_mat_export_keeps_raw_values_and_nominal_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); cube=np.arange(24).reshape(2,3,4)
            mat=r/'Plant69_Infected_Leaf3_Day2.mat'
            savemat(mat,{'H':cube,'parameters':{'Wavelengths':np.arange(4)+400}})
            with zipfile.ZipFile(r/'data.zip','w') as z:z.write(mat,mat.name)
            export_mat(r/'data.zip',r/'out')
            np.testing.assert_array_equal(np.load(r/'out'/f'{mat.stem}.npy'),cube)
            self.assertFalse((r/'out'/'metadata.csv').exists())

    def test_mismatch_blocks_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);new,rgb,cube=self.fixture(r)
            np.save(new/'Plant1_Infected_Leaf3_Day2.npy',cube+1)
            with self.assertRaisesRegex(ValueError,'Overlap check failed'):
                merge(r/'base.csv',new,rgb,r/'out')
            self.assertFalse((r/'out/metadata.csv').exists())
            self.assertFalse(json.loads((r/'out/overlap_audit.json').read_text())['passed'])

    def test_matching_overlap_not_duplicated_and_old_data_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);new,rgb,_=self.fixture(r)
            with patch('recover_revision_data.prepare_tasks') as prepare:
                merge(r/'base.csv',new,rgb,r/'out')
            merged=pd.read_csv(r/'out/metadata.csv')
            self.assertEqual(len(merged),7)
            self.assertFalse(merged.sample_id.duplicated().any())
            self.assertIn('/old/',merged.hsi_path.iloc[0])
            prepare.assert_called_once()

    def test_no_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);new,rgb,_=self.fixture(r)
            (new/'Plant1_Infected_Leaf3_Day2.npy').unlink()
            with self.assertRaisesRegex(ValueError,'no overlap'):
                merge(r/'base.csv',new,rgb,r/'out')


if __name__=='__main__':unittest.main()
