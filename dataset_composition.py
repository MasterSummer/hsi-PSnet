"""Count Run-1 paired observations and unique biological plants, without training."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from reviewer_experiments.core import read_metadata, TASK_DPI

NAMES={'dpi_2':'2 dpi','dpi_4':'4 dpi','dpi_6':'6 dpi',
       'presymptomatic_2_4':'Joint 2+4 dpi','all_dpi':'All dates pooled'}


def write_composition(metadata, output, cohort):
    frame=read_metadata(metadata)
    if frame.duplicated(['plant_id','leaf_id','dpi']).any():
        raise ValueError('Repeated biological plant/leaf/date observation')
    expected=[]
    for dpi in (2,4,6):
        for treatment,count in [('infected',72),('mock',24)]:
            for plant in range(1,count+1):
                for leaf in (3,4):
                    expected.append(dict(sample_id=f'run1_{treatment}_p{plant:03d}_l{leaf}_d{dpi}',
                        plant_id=f'run1_{treatment}_p{plant:03d}',treatment=treatment,leaf_id=f'L{leaf}',dpi=dpi))
    expected=pd.DataFrame(expected)
    # This expectation is specific to the documented Run-1 design.
    unexpected=set(frame.sample_id)-set(expected.sample_id)
    if unexpected:
        raise ValueError(f'Unexpected observation IDs for the documented Run-1 design: {sorted(unexpected)[:10]}')
    expected_by_id=expected.set_index('sample_id')
    actual_by_id=frame.set_index('sample_id')
    for column in ('plant_id','treatment','leaf_id','dpi'):
        if not actual_by_id[column].equals(expected_by_id.loc[actual_by_id.index,column]):
            raise ValueError(f'Metadata inconsistent with sample IDs: {column}')
    missing=expected[~expected.sample_id.isin(frame.sample_id)]
    rows=[]
    for task,dates in TASK_DPI.items():
        subset=frame[frame.dpi.isin(dates)]
        inoc=subset[subset.treatment=='infected']; mock=subset[subset.treatment=='mock']
        a,b=int(inoc.plant_id.nunique()),int(mock.plant_id.nunique())
        rows.append(dict(task=task,task_name=NAMES[task],inoculated_plants=a,mock_plants=b,
            total_plants=a+b,inoculated_observations=len(inoc),mock_observations=len(mock),
            total_observations=len(subset),expected_plants=96,expected_observations=192*len(dates),
            missing_observations=int(missing.dpi.isin(dates).sum())))
    table=pd.DataFrame(rows)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'dataset_composition.csv',index=False)
    missing.to_csv(output/'missing_observations.csv',index=False)
    day2=table[table.task=='dpi_2'].iloc[0]
    caption=('Classification tasks and plant-level dataset composition. Plant numbers were counted from the '
             f'{cohort} paired-data manifest. At 2 dpi, the total is {day2.inoculated_plants} inoculated + '
             f'{day2.mock_plants} mock = {day2.total_plants} plants. The experimental design comprised 96 plants. ')
    if day2.missing_observations:
        caption+=f'The paired-data subset at 2 dpi lacks {day2.missing_observations} planned observations, listed in missing_observations.csv. '
    else:
        caption+='All 192 planned paired observations at 2 dpi are present in this manifest. '
    caption+='Plants observed on multiple dates are counted once in pooled tasks. Observation counts refer to paired leaf/date records, not independent plants.'
    (output/'caption_en.txt').write_text(caption+'\n')
    lines=['# Classification tasks and plant-level dataset composition','',f'Cohort: {cohort}','',
        '| Task | Inoculated plants | Mock plants | Total plants | Paired observations | Missing observations |',
        '|---|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.task_name} | {row.inoculated_plants} | {row.mock_plants} | {row.total_plants} | {row.total_observations} | {row.missing_observations} |')
    (output/'dataset_composition.md').write_text('\n'.join(lines)+'\n\n'+caption+'\n')
    return table


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cohort',choices=['archived','recovered'],default='archived')
    a=p.parse_args();write_composition(a.metadata,a.output,a.cohort)
