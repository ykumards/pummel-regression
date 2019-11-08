import sys, os
import datetime
from pathlib import Path
import warnings
warnings.simplefilter('ignore')

import copy
import re
import torch
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy

from dataset import *
from vocab import *
from trainer import *
from model.ehr_lstm import *
from utils import *

starttime = time.time()

args = load_config('config/default.yml')
seed_everything(args.seed)
inference_model_filepath = '../output/NeuripsLSTM_14_16_50k_v3_small_rnn_log.pth'


with timer('loaded dataframe'):
    df = pd.read_csv(args.data_filepath)
    df['prev_num_visits'] = df['X_seq'].apply(lambda row: len(row.split(';')))

    df_len_before = df.shape[0]
    df = df.query(f"prev_num_visits > 2 and prev_num_visits <= {args.max_seq_length}")
    print(f"dropping {df_len_before - df.shape[0]} rows with < 2 or > {args.max_seq_length} prev visits")

    df_tr = df.query("split == 'train'")
    df_tr = df_tr.sample(n=int(args.num_samples), random_state=args.seed).iloc[:-1, :]
    df_rest = df.query("split != 'train'")
    df = pd.concat([df_tr, df_rest])
    df['target_y'] = df['target_y'].clip(upper=args.topcap)

print(f"Dataframe shape: {df.shape}")

with timer('truncating at max seq length'):
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


with timer('built dataset and vectors'):
    dataset = EHRDataset.load_dataset_and_make_vectorizer(df)
    vectorizer = dataset.vectorizer


rnn_model = EHR_LSTM(
    embed_size=args.embedding_dim, 
    input_vocab_size=vectorizer.diagnoses_vocab_len,
    rnn_hidden_size=args.rnn_hidden_size,
    hidden_size=args.hidden_size,
    max_diag_length_per_visit=args.max_diag_length_per_visit,
    dropout_p=args.dropout_p,
    use_adder=args.use_adder,
    use_mh_attn=args.use_mh_attn,
    do_bn=args.do_bn
)


rnn_model.load_state_dict(torch.load(Path(inference_model_filepath)))
print("model loaded...")

trainer = Trainer(dataset=dataset, model=rnn_model, args=args)
trainer.train_state['model_filename'] = Path(inference_model_filepath)
trainer.heldout()

print("HO loss: {0:.5f}".format(trainer.train_state['heldout_loss']))
print("HO Accuracy: {0:.5f}%".format(trainer.train_state['heldout_acc']))

df_ho = df.query("split == 'heldout'")
heldout_preds = trainer.train_state['heldout_preds']
heldout_true = trainer.train_state['heldout_true']

df_ho_mean_pred = np.ones(df_ho.shape[0]) * df_ho.target_y.mean()
print()
print(f"MSE for ho: {metrics.mean_squared_error(heldout_true, heldout_preds)}")
print(f"MAE for ho: {metrics.mean_absolute_error(heldout_true, heldout_preds)}")
print(f"R2 for ho: {metrics.r2_score(heldout_true, heldout_preds)}")
print(f"Max residual error for ho: {np.max(np.abs(np.array(heldout_true) - np.array(heldout_preds)))}")
print(f"Explained variance for ho: {metrics.explained_variance_score(heldout_true, heldout_preds)}")
print(f"Corr. for ho: {scipy.stats.stats.pearsonr(heldout_true, heldout_preds)}")
print()
