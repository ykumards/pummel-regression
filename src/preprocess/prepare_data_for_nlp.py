"""
- Script takes input years and output years and loads the corresponding year's
(processed) visits tables. 
- It then sorts the visits based on krypht.
- merges the visits table with the combined diagnosis table 
- sets the target 
"""
import os
import sys
import time
import gc

import pandas as pd
import numpy as np
import dask
from dask import dataframe as dd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from multiprocessing import cpu_count
n_cores = cpu_count()

RANDOM_STATE = 2
do_split = True
load_temp = False
target_years = [2018]
input_years = [2012, 2013, 2014, 2015, 2016, 2017]
all_years = input_years + target_years

starttime = time.time()

logging.info(f'between years {input_years[0]} to {target_years[-1]}')

assert len(target_years) > 0, "at least 1 target year needed"
assert len(input_years) > 0, "at least 1 input year needed"

VOLUME_PATH = Path("/home/ykuv/pummel_data/volume/")


df_basics = pd.concat([pd.read_csv(VOLUME_PATH/Path(f"basics_norm_{year}.csv")) for year in all_years])

df_basics.sort_values(['krypht', 'ika'], ascending=False, inplace=True)
logging.info(f"basics total rows before removing duplicates is {df_basics.shape[0]}")

df_basics.drop_duplicates(['krypht', 'sukup'], keep='first', inplace=True)
logging.info(f"basics total rows after removing duplicates is {df_basics.shape[0]}")

if load_temp == False:
    combined_diag = pd.read_csv(VOLUME_PATH/'merged_diagnosis_grouped.csv')
    logging.info("diagnosis data loaded...")
    
    # Processing input tables
    df_x = pd.read_csv(
        VOLUME_PATH/Path(f"visits_norm_{input_years[0]}.csv"), parse_dates=['tupva'])
    df_y = pd.read_csv(
        VOLUME_PATH/Path(f"visits_norm_{target_years[0]}.csv"), parse_dates=['tupva'])

    for year in target_years[1:]:
        df_y = pd.concat([df_y, pd.read_csv(
            VOLUME_PATH/Path(f"visits_norm_{year}.csv"))], parse_dates=['tupva'])

    for year in input_years[1:]:
        df_x = pd.concat([df_x, pd.read_csv(
            VOLUME_PATH/Path(f"visits_norm_{year}.csv"), parse_dates=['tupva'])])

    logging.info("visits data loaded...")

    logging.info("dropping rows with missing krypht id...")
    df_x = df_x.query("krypht != -1")
    df_y = df_y.query("krypht != -1")

    logging.info("merging visits with diag..."),
    df_x.set_index('isoid', inplace=True)
    df_y.set_index('isoid', inplace=True)
    combined_diag.set_index('isoid', inplace=True)

    df_x = df_x.join(combined_diag, on='isoid', how='left')
    df_y = df_y.join(combined_diag, on='isoid', how='left')
    logging.info("done.")

    del combined_diag
    gc.collect()


    # extract only physical visits for target
    logging.info(
        f"number of rows in df_y considering all visits: {df_y.shape[0]}")
    df_y = df_y.query(
        "palvelumuoto == 'T11' and (yhteystapa == 'R10' or yhteystapa == 'R40')")
    logging.info(
        f"number of rows in df_y considering only physical visits: {df_y.shape[0]}")

    # sorting needed for time shift calc
    logging.info("sorting df_x by krypht and tupva to get time deltas...")
    df_x.sort_values(['krypht', 'tupva'], inplace=True)
    df_x['days_from_prev'] = df_x['tupva'] - df_x['tupva'].shift(1)
    df_x['days_from_prev'] = df_x['days_from_prev'].dt.total_seconds() / 86400.0

    df_y.sort_values(['krypht', 'tupva'], inplace=True)
    df_y['days_from_prev'] = df_y['tupva'] - df_y['tupva'].shift(1)
    df_y['days_from_prev'] = df_y['days_from_prev'].dt.total_seconds() / 86400.0
    logging.info("done.")

    df_x.reset_index(inplace=True); df_x.set_index('krypht', inplace=True)
    df_y.reset_index(inplace=True); df_y.set_index('krypht', inplace=True)

    # convert to str
    df_x['diagnosis'].fillna('<UNK>', inplace=True)
    df_y['diagnosis'].fillna('<UNK>', inplace=True)
    df_x['palvelumuoto'].fillna('<UNK>', inplace=True)
    df_x['yhteystapa'].fillna('<UNK>', inplace=True)
    
    df_x['diagnosis'] = df_x['diagnosis'].apply(lambda row: str(row))
    df_x['days_from_prev'] = df_x['days_from_prev'].apply(lambda row: str(row))
    df_x['yhteystapa'] = df_x['yhteystapa'].apply(lambda row: str(row))
    df_x['palvelumuoto'] = df_x['palvelumuoto'].apply(lambda row: str(row))
    df_x['tupva'] = df_x['tupva'].apply(lambda row: str(row))
    df_x['isoid'] = df_x['isoid'].apply(lambda row: str(row))

    df_y['diagnosis'] = df_y['diagnosis'].apply(lambda row: str(row))
    df_y['days_from_prev'] = df_y['days_from_prev'].apply(lambda row: str(row))


    df_x['diag_length_per_visit'] = df_x['diagnosis'].apply(lambda row: len(row.split()))

    tqdm.pandas()

    logging.info("grouping on krypht... ")
    def _diag_grouper(paritition):
        diag_seq = ";".join(paritition['diagnosis'].values.tolist())
        return diag_seq

    def _time_delta_grouper(partition):
        time_delta_seq = partition['days_from_prev'].values.tolist()
        time_delta_seq[0] = '0.0'
        return ";".join(time_delta_seq)

    def _yh_grouper(partition):
        yh_seq = ";".join(partition['yhteystapa'].values.tolist())
        return yh_seq

    def _pal_grouper(partition):
        pal_seq = ";".join(partition['palvelumuoto'].values.tolist())
        return pal_seq

    def _admittime_grouper(partition):
        pal_seq = ";".join(partition['tupva'].values.tolist())
        return pal_seq

    def _isoid_grouper(partition):
        pal_seq = ";".join(partition['isoid'].values.tolist())
        return pal_seq

    print("grouping df_x...")
    grouped_max_visit_length = df_x.groupby('krypht').progress_apply(lambda partition: max(partition['diag_length_per_visit'].values))
    grouped_diag = df_x.groupby('krypht').progress_apply(_diag_grouper)
    grouped_time_delta = df_x.groupby('krypht').progress_apply(_time_delta_grouper)
    grouped_yh = df_x.groupby('krypht').progress_apply(_yh_grouper)
    grouped_pal = df_x.groupby('krypht').progress_apply(_pal_grouper)
    grouped_admittime = df_x.groupby('krypht').progress_apply(_admittime_grouper)
    grouped_isoid = df_x.groupby('krypht').progress_apply(_isoid_grouper)

    grouped_df_x = pd.DataFrame(list(zip(grouped_diag.index, 
                                         grouped_diag.values, 
                                         grouped_time_delta,
                                         grouped_max_visit_length,
                                         grouped_yh,
                                         grouped_pal, 
                                         grouped_admittime, 
                                         grouped_isoid)),
                                columns = ['krypht', 
                                           'X_seq', 
                                           'days_from_prev',
                                           'max_num_diagnoses_per_visit',
                                           'yh', 'pal', 'admit_time', 'isoid'])
    print("writing df_x_temp...")
    grouped_df_x.to_csv(VOLUME_PATH/'df_x_temp.csv', index=False)

    print("grouping df_y...")

    grouped_diag = df_y.groupby('krypht').progress_apply(_diag_grouper)
    grouped_time_delta = df_y.groupby('krypht').progress_apply(_time_delta_grouper)
    grouped_df_y = pd.DataFrame(list(zip(grouped_diag.index, 
                                         grouped_diag.values, 
                                         grouped_time_delta)),
                                columns = ['krypht', 
                                           'Y_seq', 
                                           'Y_days_from_prev']) 

    print('writing df_y_temp...')
    grouped_df_y.to_csv(VOLUME_PATH/'df_y_temp.csv', index=False)

    logging.info("grouping on krypht done... ")

else:
    grouped_df_x = pd.read_csv(VOLUME_PATH/'df_x_temp.csv')
    grouped_df_y = pd.read_csv(VOLUME_PATH/'df_y_temp.csv')

grouped_df_x.set_index('krypht', inplace=True)
grouped_df_y.set_index('krypht', inplace=True)
df_basics.set_index('krypht', inplace=True)

grouped_df_y['target_y'] = grouped_df_y['Y_seq'].apply(
    lambda row: len(row.split(';')))

merged_x_y = grouped_df_x.loc[:, ['X_seq', 'days_from_prev', 
                                  'max_num_diagnoses_per_visit',
                                  'yh', 'pal', 'admit_time', 'isoid']].merge(
    grouped_df_y.loc[:, ['Y_seq', 'Y_days_from_prev', 'target_y']], how='left', on='krypht')

merged_x_y = merged_x_y.merge(df_basics, how='left', on='krypht')

merged_x_y.fillna({
    'target_y': 0.0, # should be 0 because it means no physical visits this year
    'sukup': merged_x_y.sukup.mode(),
    'ika': merged_x_y.ika.median(),
}, inplace=True)


def _time_delta_mean(delta_seq):
    delta_seq = np.array(delta_seq.split(';')).astype(np.float)
    return np.sum(delta_seq)/delta_seq.shape[0]

merged_x_y['prev_num_visits'] = merged_x_y['X_seq'].apply(lambda row: len(row.split(';')))
# merged_x_y = merged_x_y.query("prev_num_visits > 2 and prev_num_visits <= 100")

merged_x_y['time_delta_mean'] = merged_x_y['days_from_prev'].apply(lambda row: _time_delta_mean(row))
merged_x_y['yh'] = merged_x_y['yh'].astype(str)
merged_x_y['yh'] = merged_x_y['yh'].apply(lambda row: row.strip())
merged_x_y['pal'] = merged_x_y['pal'].astype(str)
merged_x_y['pal'] = merged_x_y['pal'].apply(lambda row: row.strip())

print(merged_x_y.head(5))

logging.info("writing to file")


#### Validation split logic
if do_split:
    print("doing split...")
    # Add a column to the dataset for split
    train_size = 0.8
    test_size = 0.1
    heldout_size = 0.1

    train, test = train_test_split(merged_x_y, test_size=(
        test_size + heldout_size), random_state=RANDOM_STATE)
    test, heldout = train_test_split(test, test_size=0.5, random_state=RANDOM_STATE)
    train, val = train_test_split(train, test_size=0.1, random_state=RANDOM_STATE)

    train['split'] = 'train'
    test['split'] = 'test'
    val['split'] = 'val'
    heldout['split'] = 'heldout'

    pd.concat([train, test, val, heldout])\
      .to_csv(VOLUME_PATH/Path(
          f'NLPized_data_{input_years[0]}_to_{target_years[-1]}_w_split_seq.csv'),
        index=True)
else:
    merged_x_y.to_csv(
        VOLUME_PATH/Path(f'NLPized_data_{input_years[0]}_to_{target_years[-1]}_seq.csv'), index=True) #index is krypht

print(f'done in {(time.time() - starttime)/60:.4f} minutes.')
