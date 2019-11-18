import time
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
import datetime
import lightgbm as lgb
import numpy as np
import scipy
import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import scipy
from scipy import sparse


from utils import *

args = load_config('config/default.yml')
sys.path.insert(0, '../src/')
warnings.simplefilter('ignore')
VOLUME_PATH = Path('/home/ykuv/pummel_data/')

seed_everything(args.seed)

with timer('loaded and truncated dataframe'):
    df = pd.read_csv(VOLUME_PATH/'NLPized_data_2014_to_2016_w_split_seq.csv')
    df['prev_num_visits'] = df['X_seq'].apply(lambda row: len(row.split(';')))

    df_len_before = df.shape[0]
    df = df.query(f"\tprev_num_visits > 2 and prev_num_visits <= {args.max_seq_length}")
    print(f"\tdropping {df_len_before - df.shape[0]} rows with < 2 or > {args.max_seq_length} prev visits")

    df_tr = df.query("split == 'train'")
    df_tr = df_tr.sample(n=args.num_samples, random_state=args.seed)
    df_rest = df.query("split != 'train'")
    df = pd.concat([df_tr, df_rest])

    df['target_y'] = df['target_y'].apply(lambda row: row if row <= args.topcap else args.topcap)
    print(f"\tdataframe size: {df.shape[0]}")
    print(f"\ttrain dataframe size: {df_tr.shape[0]}")    


    def _truncator(row):
        visits = row.split(';')
        return ";".join(visits[-args.max_seq_length:])

    def _time_delta_mean(delta_seq):
        delta_seq = np.array(delta_seq.split(';')).astype(np.float)
        return np.sum(delta_seq)/delta_seq.shape[0]

    df['X_seq'] = df['X_seq'].apply(lambda row: _truncator(row))
    df['pal'] = df['pal'].apply(lambda row: _truncator(row))
    df['yh'] = df['yh'].apply(lambda row: _truncator(row))
    df['time_delta_mean'] = df['days_from_prev'].apply(lambda row: _time_delta_mean(row))
    df['days_from_prev'] = df['days_from_prev'].apply(lambda row: _truncator(row))


with timer('prepare count-vectors'):
    df['X_text'] = df['X_seq'].apply(lambda row: row.replace(';', ' '))
    cv_text = TfidfVectorizer(lowercase=False, tokenizer=lambda row: row.split())
    X_text_train = cv_text.fit_transform(df.query("split == 'train'")['X_text'].values)

    X_text_val = cv_text.transform(df.query("split == 'val'")['X_text'].values)
    X_text_heldout = cv_text.transform(df.query("split == 'heldout'")['X_text'].values)

with timer('prepare numerical features'):
    X_num_train = df.query("split == 'train'").loc[:, ['sukup', 'ika', 'time_delta_mean']]
    X_num_val = df.query("split == 'val'").loc[:, ['sukup', 'ika', 'time_delta_mean']]
    X_num_heldout = df.query("split == 'heldout'").loc[:, ['sukup', 'ika', 'time_delta_mean']]


with timer('prepare ml-ready tensors'):
    x_train = scipy.sparse.hstack((
        X_text_train, 
        X_num_train
    ))
    y_train = df.query("split == 'train'")['target_y'].values

    x_val = scipy.sparse.hstack((
        X_text_val, 
        X_num_val
    ))
    y_val = df.query("split == 'val'")['target_y'].values

    x_heldout = scipy.sparse.hstack((
        X_text_heldout, 
        X_num_heldout
    ))
    y_heldout = df.query("split == 'heldout'")['target_y'].values

    feature_names = cv_text.get_feature_names()
    feature_names = np.hstack([
        feature_names, 
        ['gender', 'age', 'time_delta_mean']])
    print(f"\tx shape: {x_train.shape}")


with timer('LASSO regression'):
    clf = Lasso()
    clf.fit(x_train, y_train)

    preds = clf.predict(x_val)
    err = mse(y_val, preds)
    err_mae = mae(y_val, preds)
    print("Lasso Results"); print("="*90)
    print(f'\tvalid MSE: {err}')
    print(f'valid MAE: {err_mae}')
    
    tr_preds = clf.predict(x_train)
    tr_err = mse(y_train, tr_preds)
    tr_err_mae = mae(y_train, tr_preds)
    print(f'\tTrain MSE: {tr_err}')
    print(f'\tTrain MAE: {tr_err_mae}')
       
    heldout_preds_lasso = clf.predict(x_heldout)
    heldout_err = mse(y_heldout, heldout_preds_lasso)
    heldout_err_mae = mae(y_heldout, heldout_preds_lasso)
    print(f"\theldout MSE: {heldout_err}")
    print(f"\theldout MAE: {heldout_err_mae}")
    print(f"\tR2 for ho: {metrics.r2_score(np.array(y_heldout), np.array(heldout_preds_lasso))}")
    print(f"\tMax residual error for ho: {np.max(np.abs(np.array(y_heldout) - np.array(heldout_preds_lasso)))}")
    print(f"\tExplained variance for ho: {metrics.explained_variance_score(y_heldout, heldout_preds_lasso)}")
    print(f"\tCorr. for ho: {scipy.stats.stats.pearsonr(y_heldout, heldout_preds_lasso)}")
    print(f"\tSpearman Corr. for ho: {scipy.stats.stats.spearmanr(y_heldout, heldout_preds_lasso)}")

with timer('run lightgbm baseline'):

    params = {
        'boosting_type': 'gbdt',
        'objective': 'poisson',
        'metric': 'mse',
        'num_leaves': 128,
        'max_depth': -1, 
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'verbose': 0, 
        'num_threads': 18,
        'seed': args.seed,
        'early_stopping_round': 100
    }
    n_estimators = 10000

    n_iters = 1
    preds_buf = []
    err_buf = []

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_val, label=y_val)
    watchlist = [d_valid]

    model = lgb.train(
        params=params, 
        train_set=d_train, 
        num_boost_round=n_estimators, 
        valid_sets=[d_valid, d_train],
        feature_name=list(feature_names),
        valid_names=['valid', 'train'],
        verbose_eval=500
    )


    preds = model.predict(x_val)
    err = mse(y_val, preds)
    err_mae = mae(y_val, preds)
    
    heldout_preds_lgbm = model.predict(x_heldout)
    heldout_err = mse(y_heldout, heldout_preds_lgbm)
    heldout_err_mae = mae(y_heldout, heldout_preds_lgbm)

    print("LGBM Results"); print("="*90)
    print(f"\theldout MSE: {heldout_err}")
    print(f"\theldout MAE: {heldout_err_mae}")
    print(f"\tR2 for ho: {metrics.r2_score(np.array(y_heldout), np.array(heldout_preds_lgbm))}")
    print(f"\tMax residual error for ho: {np.max(np.abs(np.array(y_heldout) - np.array(heldout_preds_lgbm)))}")
    print(f"\tExplained variance for ho: {metrics.explained_variance_score(y_heldout, heldout_preds_lgbm)}")
    print(f"\tPearson Corr. for ho: {scipy.stats.stats.pearsonr(y_heldout, heldout_preds_lgbm)}")
    print(f"\tSpearman Corr. for ho: {scipy.stats.stats.spearmanr(y_heldout, heldout_preds_lgbm)}")
