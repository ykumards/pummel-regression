import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.blocks import *


class EHR_LSTM(nn.Module):
    """
    LSTM model that takes in a sequence as input and predicts the number of times
    the patient will visit the hospital in the nect year. So output is a real-number (using ReLU)
    """

    def __init__(
        self,
        embed_size: int,
        input_vocab_size: int,
        rnn_hidden_size: int,
        hidden_size: int,
        max_diag_length_per_visit: int,
        bidirectional=False,
        padding_idx=0,
        batch_first=True,
        dropout_p=0.5,
        use_adder=False,
        use_mh_attn=False,
        do_bn=True,
    ):
        super().__init__()

        self.do_bias = not do_bn
        self.do_bn = do_bn
        self.rnn_hidden_size = rnn_hidden_size
        self.hidden_size = hidden_size

        self.use_adder = use_adder
        self.use_mh_attn = use_mh_attn

        self.gender_embed = nn.Embedding(num_embeddings=3, embedding_dim=1)

        self.diag_embed = nn.Embedding(
            num_embeddings=input_vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx,
        )

        self.embed_conv = EmbeddingConvolution(
            in_channels=max_diag_length_per_visit,
            dropout_p=dropout_p / 2,
            kernel_size=1,
        )

        self.embed_sum = EmbeddingAdder()

        self.rnn = nn.LSTM(
            input_size=embed_size + 4,
            hidden_size=rnn_hidden_size,
            batch_first=batch_first,
            dropout=0.1,
            bidirectional=bidirectional,
        )

        self.attn_alpha = nn.Parameter(torch.empty(1))
        self.attn = nn.MultiheadAttention(rnn_hidden_size, 8)  # 800/8

        self.fc_w_feats = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=hidden_size // 2,
            bias=self.do_bias,
        )

        self.ln = nn.LayerNorm(hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, 1, bias=True)
        self.activation = nn.PReLU()
        self._droput_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        print(f"using adder is {self.use_adder}")
        print(f"using mh_attn is {self.use_mh_attn}")

    def forward(
        self,
        x_diag: torch.LongTensor,
        x_lengths: torch.LongTensor,
        x_timedels: torch.FloatTensor = None,
        x_age: torch.FloatTensor = None,
        x_gender: torch.LongTensor = None,
        x_time_delta_mean: torch.FloatTensor = None,
        all_input_zeros: bool = False,
    ) -> torch.Tensor:

        batch_size, max_seq_length, _ = x_diag.shape

        if all_input_zeros:
            x_diag = torch.zeros_like(x_diag)

        #### Embeddings
        x_embed = self.diag_embed(x_diag)

        ## rNN Layers
        # This does early fusion with feature replication
        x_gender = (
            self.gender_embed(x_gender)
            .unsqueeze(1)
            .expand(batch_size, max_seq_length, -1)
        )
        x_age = x_age.unsqueeze(1).expand(batch_size, max_seq_length, -1)
        x_time_delta_mean = x_time_delta_mean.unsqueeze(1).expand(
            batch_size, max_seq_length, -1
        )

        if self.use_adder:
            x_embed_sum = self.embed_sum(x_embed)
        else:
            x_embed_sum = self.embed_conv(x_embed.permute(0, 2, 1, 3)).squeeze(1)

        # we replicate the scalar features across every time-step
        x_embed_sum = torch.cat(
            [x_embed_sum, x_gender, x_age, x_time_delta_mean, x_timedels.unsqueeze(2)],
            dim=2,
        )

        rnn_outs, _ = self.rnn(x_embed_sum)
        rnn_outs = rnn_outs.contiguous().view(
            batch_size, max_seq_length, self.rnn_hidden_size
        )

        attn_out, _ = self.attn(rnn_outs, rnn_outs, rnn_outs)
        rnn_attn_max = F.adaptive_max_pool1d(attn_out.permute(0, 2, 1), 1).squeeze()

        rnn_out = gather_last_relevant_hidden(rnn_outs, x_lengths)

        if self.use_mh_attn:
            rnn_out = (
                1 - F.sigmoid(self.attn_alpha)
            ) * rnn_out + rnn_attn_max * F.sigmoid(self.attn_alpha)

        z = self.fc_w_feats(rnn_out)

        if self.do_bn:
            z = self.ln(z)

        z = self.dropout(self.activation(z))
        y_pred = self.out(z)
        y_out = F.relu(y_pred.view(-1))

        return y_out
