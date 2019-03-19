import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")

PLOT = False
TRAIN_SIZE = 10**3
VALI_SIZE  = 10**3
EP_LOG = 2

def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return TrainData(feats, targs), scale


def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

    # Train-test split (T, train_size, batch_size, loss_func)

    # ANDREA --> Smaller training size
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    #train_cfg = TrainConfig(T, 10**3, batch_size, nn.MSELoss())

    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi: json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {
        "encoder_hidden_size": encoder_hidden_size,
        "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs
    }
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi: json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad], lr=learning_rate)
    decoder_optimizer = optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad], lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):

    BA = t_cfg.batch_size
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / BA))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / BA:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):


        print(e_i, end='\r')

        # ANDREA --> The training set is now chosen at random
        #print(len(train_data), t_cfg.train_size, train_data[0][0][:2])
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        perm_idx = np.random.choice(perm_idx, size=TRAIN_SIZE)

        #for t_i in range(0, t_cfg.train_size, BA):
        for t_i in range(0, TRAIN_SIZE, BA):
            batch_idx = perm_idx[t_i:(t_i + BA)]
            
            ################################################################
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)
            ################################################################

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // BA] = loss
            # if (j / BA) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / BA, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        # Log stuffs every EP_LOG epochs
        if e_i % EP_LOG == 0:

            y_test_pred = predict(net, train_data, t_cfg.train_size, BA, t_cfg.T, on_train=False, on_eval=True)
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:t_cfg.train_size+len(y_test_pred)]
            val_loss = np.mean(np.square(val_loss))

            y_train_pred = predict(net, train_data, t_cfg.train_size, BA, t_cfg.T, on_train=True)
            tra_loss = y_train_pred - train_data.targs[:len(y_train_pred)]
            tra_loss = np.mean(np.square(tra_loss))

            #logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {val_loss}.")
            logger.info(f"Epoch {e_i:d}, train loss: {tra_loss:3.3f}, val loss: {val_loss:3.3f}.")
            
            if PLOT:
                plt.figure()
                plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs, label="True")
                plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred, label='Predicted - Train')
                t0 = t_cfg.T + len(y_train_pred)
                plt.plot(range(t0, t0 + VALI_SIZE), y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, : ] = train_data.feats[b_slc, :]

        #### KEEP IT ZEROS
        #y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False, on_eval=False):
    out_size  = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))
        #y_pred = np.zeros((VALI_SIZE, out_size))

    if on_eval: #AB
        y_pred = np.zeros((VALI_SIZE, out_size))

    for y_i in range(0, len(y_pred), batch_size):

        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                # ANDREA --> The validation set is chosen at random
                # b_idx = np.random.randint(0, len(t_dat.feats)-train_size)
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            
            ## Leave it zeros
            # y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred





