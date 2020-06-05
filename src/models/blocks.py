import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_last_relevant_hidden(hiddens, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)


def gather_all_relevant_hidden(hiddens, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy()
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(hiddens[batch_index, :column_index])
    return torch.stack(out)


def elementwise_scaled_by_length(hiddens, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy()
    out = []
    for batch_idx, xlen in enumerate(x_lengths):
        out.append(hiddens[batch_idx, :, :] / np.sqrt(xlen))
    return torch.stack(out)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class EmbeddingAdder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sum(dim=2)


class EmbeddingConvolution(nn.Module):
    # https://github.com/pytorch/pytorch/issues/3867#issuecomment-407663012
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        dropout_p=0.3,
        inter_channels=16,
        padding_layer=nn.ZeroPad2d,
    ):
        super().__init__()

        # input (N, Cin, H, W)
        # output (N, Cout, H, W)
        # we treat the visitation dimension as channels
        # so Cout will be 1
        # for same convolution P = ((S-1)*W-S+F)/2

        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.in_cnn = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=(kernel_size, kernel_size),
            ),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Dropout2d(dropout_p),
        )

        self.mid_cnn = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=(kernel_size, kernel_size),
            ),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Dropout2d(dropout_p),
        )

        self.out_cnn = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=1,
                kernel_size=(kernel_size, kernel_size),
            ),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout2d(dropout_p),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.in_cnn(x)
        skip = x
        x = self.mid_cnn(x)
        x = self.out_cnn(x + skip)
        return x
