# -*- coding: utf-8 -*-
"""
Master Thesis Machine Learning
Gwen Hirsch
2022

"""
#---------------------------------------------------
#acd imports
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import time
import argparse
import os
import math
import itertools
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from torch.optim import lr_scheduler
from abc import abstractmethod

#own imports
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import networkx as nx
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pytz
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import gc
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
#---------------------------------------------------

#---------------------------------------------------
# utils
#---------------------------------------------------
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.distributions as tdist
# from torch.autograd import Variable
# from sklearn.metrics import roc_auc_score
# from collections import defaultdict

def my_softmax(input, axis=1):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: From https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: From https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: From https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def encode_onehot(labels):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(
    preds, num_atoms, num_edge_types, add_const=False, eps=1e-16
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target, binary=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    _, preds = preds.max(-1)
    if binary:
        preds = (preds >= 1).long()
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return float(correct) / (target.size(0) * target.size(1))


def calc_auroc(pred_edges, GT_edges):
    pred_edges = 1 - pred_edges[:, :, 0]
    return roc_auc_score(
        GT_edges.cpu().detach().flatten(),
        pred_edges.cpu().detach().flatten(),  # [:, :, 1]
    )


def kl_latent(args, prob, log_prior, predicted_atoms):
    if args.prior != 1:
        return kl_categorical(prob, log_prior, predicted_atoms)
    else:
        return kl_categorical_uniform(prob, predicted_atoms, args.edge_types)


def get_observed_relations_idx(num_atoms):
    length = (num_atoms ** 2) - num_atoms * 2
    remove_idx = np.arange(length)[:: num_atoms - 1][1:] - 1
    idx = np.delete(np.linspace(0, length - 1, length), remove_idx)
    return idx


def mse_per_sample(output, target):
    mse_per_sample = F.mse_loss(output, target, reduction="none")
    mse_per_sample = torch.mean(mse_per_sample, dim=(1, 2, 3)).cpu().data.numpy()
    return mse_per_sample


def edge_accuracy_per_sample(preds, target):
    _, preds = preds.max(-1)
    acc = torch.sum(torch.eq(preds, target), dim=1, dtype=torch.float64,) / preds.size(
        1
    )
    return acc.cpu().data.numpy()


def auroc_per_num_influenced(preds, target, total_num_influenced):
    preds = 1 - preds[:, :, 0]
    preds = preds.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    preds_per_num_influenced = defaultdict(list)
    targets_per_num_influenced = defaultdict(list)

    for idx, k in enumerate(total_num_influenced):
        preds_per_num_influenced[k].append(preds[idx])
        targets_per_num_influenced[k].append(target[idx])

    auc_per_num_influenced = np.zeros((max(preds_per_num_influenced) + 1))
    for num_influenced, elem in preds_per_num_influenced.items():
        auc_per_num_influenced[num_influenced] = roc_auc_score(
            np.vstack(targets_per_num_influenced[num_influenced]).flatten(),
            np.vstack(elem).flatten(),
        )

    return auc_per_num_influenced


def edge_accuracy_observed(preds, target, num_atoms=5):
    idx = get_observed_relations_idx(num_atoms)
    _, preds = preds.max(-1)
    correct = preds[:, idx].eq(target[:, idx]).cpu().sum()
    return float(correct) / (target.size(0) * len(idx))


def calc_auroc_observed(pred_edges, GT_edges, num_atoms=5):
    idx = get_observed_relations_idx(num_atoms)
    pred_edges = pred_edges[:, :, 1]
    return roc_auc_score(
        GT_edges[:, idx].cpu().detach().flatten(),
        pred_edges[:, idx].cpu().detach().flatten(),
    )


def kl_normal_reverse(prior_mean, prior_std, mean, log_std, downscale_factor=1):
    std = softplus(log_std) * downscale_factor
    d = tdist.Normal(mean, std)
    prior_normal = tdist.Normal(prior_mean, prior_std)
    return tdist.kl.kl_divergence(d, prior_normal).mean()


def sample_normal_from_latents(latent_means, latent_logsigmas, downscale_factor=1):
    latent_sigmas = softplus(latent_logsigmas) * downscale_factor
    eps = torch.randn_like(latent_sigmas)
    latents = latent_means + eps * latent_sigmas
    return latents


def softplus(x):
    return torch.log(1.0 + torch.exp(x))


def distribute_over_GPUs(args, model, num_GPU=None):
    ## distribute over GPUs
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


def create_rel_rec_send(args, num_atoms):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    if args.unobserved > 0 and args.model_unobserved == 1:
        num_atoms -= args.unobserved

    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    if args.cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    return rel_rec, rel_send


def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list


# Latent Temperature Experiment utils
def get_uniform_parameters_from_latents(latent_params):
    n_params = latent_params.shape[1]
    logit_means = latent_params[:, : n_params // 2]
    logit_widths = latent_params[:, n_params // 2 :]
    means = sigmoid(logit_means)
    widths = sigmoid(logit_widths)
    mins, _ = torch.min(torch.cat([means, 1 - means], dim=1), dim=1, keepdim=True)
    widths = mins * widths
    return means, widths


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def sample_uniform_from_latents(latent_means, latent_width):
    latent_dist = tdist.uniform.Uniform(
        latent_means - latent_width, latent_means + latent_width
    )
    latents = latent_dist.rsample()
    return latents


def get_categorical_temperature_prior(mid, num_cats, to_torch=True, to_cuda=True):
    categories = [mid * (2.0 ** c) for c in np.arange(num_cats) - (num_cats // 2)]
    if to_torch:
        categories = torch.Tensor(categories)
    if to_cuda:
        categories = categories.cuda()
    return categories


def kl_uniform(latent_width, prior_width):
    eps = 1e-8
    kl = torch.log(prior_width / (latent_width + eps))
    return kl.mean()


def get_uniform_logprobs(inferred_mu, inferred_width, temperatures):
    latent_dist = tdist.uniform.Uniform(
        inferred_mu - inferred_width, inferred_mu + inferred_width
    )
    cdf = latent_dist.cdf(temperatures)
    log_prob_default = latent_dist.log_prob(inferred_mu)
    probs = torch.where(
        cdf * (1 - cdf) > 0.0, log_prob_default, torch.full(cdf.shape, -8).cuda()
    )
    return probs.mean()


def get_preds_from_uniform(inferred_mu, inferred_width, categorical_temperature_prior):
    categorical_temperature_prior = torch.reshape(
        categorical_temperature_prior, [1, -1]
    )
    preds = (
        (categorical_temperature_prior > inferred_mu - inferred_width)
        * (categorical_temperature_prior < inferred_mu + inferred_width)
    ).double()
    return preds


def get_correlation(a, b):
    numerator = torch.sum((a - a.mean()) * (b - b.mean()))
    denominator = torch.sqrt(torch.sum((a - a.mean()) ** 2)) * torch.sqrt(
        torch.sum((b - b.mean()) ** 2)
    )
    return numerator / denominator


def get_offdiag_indices(num_nodes):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

#---------------------------------------------------
# utils_unobserved
#---------------------------------------------------

# import torch
# import torch.nn.functional as F
# from collections import defaultdict

# from model import utils


def remove_unobserved(args, data, mask_idx):
    data = torch.cat(
        (data[:, :mask_idx, :, :], data[:, mask_idx + args.unobserved :, :, :],), dim=1,
    )
    return data


def baseline_mean_imputation(args, data_encoder, mask_idx):
    target_unobserved = data_encoder[:, mask_idx, :, :]
    data_encoder = remove_unobserved(args, data_encoder, mask_idx)

    unobserved = torch.mean(data_encoder, dim=1).unsqueeze(1)

    mse_unobserved = F.mse_loss(
        torch.squeeze(unobserved), torch.squeeze(target_unobserved)
    )

    data_encoder = torch.cat(
        (data_encoder[:, :mask_idx, :], unobserved, data_encoder[:, mask_idx:, :],),
        dim=1,
    )

    return data_encoder, unobserved, mse_unobserved


def baseline_remove_unobserved(
    args, data_encoder, data_decoder, mask_idx, relations, predicted_atoms
):
    data_encoder = remove_unobserved(args, data_encoder, mask_idx)
    data_decoder = remove_unobserved(args, data_decoder, mask_idx)

    predicted_atoms -= args.unobserved
    observed_relations_idx = get_observed_relations_idx(args.num_atoms)
    relations = relations[:, observed_relations_idx]

    return data_encoder, data_decoder, predicted_atoms, relations


def add_unobserved_to_data(args, data, unobserved, mask_idx, diff_data_enc_dec):
    if diff_data_enc_dec:
        data = torch.cat(
            (
                data[:, :mask_idx, :],
                torch.unsqueeze(unobserved[:, :, -1, :], 2).repeat(
                    1, 1, args.timesteps, 1
                ),  # start predicting unobserved path from last point predicted
                data[:, mask_idx + 1 :, :],
            ),
            dim=1,
        )
    else:
        data = torch.cat(
            (data[:, :mask_idx, :], unobserved, data[:, mask_idx + 1 :, :],), dim=1,
        )

    return data


def calc_mse_observed(args, output, target, mask_idx):
    output_observed = remove_unobserved(args, output, mask_idx)
    target_observed = remove_unobserved(args, target, mask_idx)
    return F.mse_loss(output_observed, target_observed)


def calc_performance_per_num_influenced(args, relations, output, target, logits, prob, mask_idx, losses):
    if args.model_unobserved == 1:
        num_atoms = args.num_atoms - args.unobserved
    else:
        num_atoms = args.num_atoms

    influenced_idx_relations = list(
        range(num_atoms - 2, num_atoms ** 2, num_atoms - 1)
    )[: num_atoms - 1]
    influenced_idx = relations[:, influenced_idx_relations]

    ## calculate performance based on how many particles are influenced by unobserved one
    total_num_influenced = torch.sum(influenced_idx, 1).tolist()
    if args.model_unobserved != 1 and args.unobserved > 0:
        observed_idx = get_observed_relations_idx(args.num_atoms).astype(int)
        acc_per_sample = edge_accuracy_per_sample(logits[:, observed_idx, :], relations[:, observed_idx])

        output_observed = remove_unobserved(args, output, mask_idx)
        target_observed = remove_unobserved(args, target, mask_idx)
        mse_per_sample = mse_per_sample(output_observed, target_observed)

        auroc_per_num_infl = auroc_per_num_influenced(prob[:, observed_idx, :], relations[:, observed_idx], total_num_influenced)
    else:
        acc_per_sample = edge_accuracy_per_sample(logits, relations)
        mse_per_sample = mse_per_sample(output, target)
        auroc_per_num_infl= auroc_per_num_influenced(prob, relations, total_num_influenced)

    if losses["acc_per_num_influenced"] == 0:
        losses["acc_per_num_influenced"] = defaultdict(list)
        losses["mse_per_num_influenced"] = defaultdict(list)
        losses["auroc_per_num_influenced"] = defaultdict(list)

    for idx, k in enumerate(total_num_influenced):
        losses["acc_per_num_influenced"][k].append(acc_per_sample[idx])
        losses["mse_per_num_influenced"][k].append(mse_per_sample[idx])

    for idx, elem in enumerate(auroc_per_num_infl):
        losses["auroc_per_num_influenced"][idx].append(elem)

    return losses


#---------------------------------------------------
# model_loader
#---------------------------------------------------

# import os
# import torch
# import torch.optim as optim
# from torch.optim import lr_scheduler

# from model.modules import *
# from model.MLPEncoder import MLPEncoder
# from model.CNNEncoder import CNNEncoder
# from model.MLPEncoderUnobserved import MLPEncoderUnobserved
# from model.EncoderGlobalTemp import CNNEncoderGlobalTemp

# from model.MLPDecoder import MLPDecoder
# from model.RNNDecoder import RNNDecoder
# from model.SimulationDecoder import SimulationDecoder
# from model.DecoderGlobalTemp import MLPDecoderGlobalTemp, SimulationDecoderGlobalTemp

# from model import utils


def load_distribution(args):
    edge_probs = torch.randn(
        torch.Size([args.num_atoms ** 2 - args.num_atoms, args.edge_types]),
        device=args.device.type,
        requires_grad=True,
    )
    return edge_probs


def load_encoder(args):
    if args.global_temp:
        encoder = CNNEncoderGlobalTemp(
            args,
            args.dims,
            args.encoder_hidden,
            args.edge_types,
            args.encoder_dropout,
            args.factor,
        )
    elif args.unobserved > 0 and args.model_unobserved == 0:
        encoder = MLPEncoderUnobserved(
            args,
            args.timesteps * args.dims,
            args.encoder_hidden,
            args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor,
        )
    else:
        if args.encoder == "mlp":
            encoder = MLPEncoder(
                args,
                args.timesteps * args.dims,
                args.encoder_hidden,
                args.edge_types,
                do_prob=args.encoder_dropout,
                factor=args.factor,
            )
        elif args.encoder == "cnn":
            encoder = CNNEncoder(
                args,
                args.dims,
                args.encoder_hidden,
                args.edge_types,
                args.encoder_dropout,
                args.factor,
            )

    encoder, num_GPU = distribute_over_GPUs(args, encoder, num_GPU=args.num_GPU)
    if args.load_folder:
        print("Loading model file")
        args.encoder_file = os.path.join(args.load_folder, "encoder.pt")
        encoder.load_state_dict(torch.load(args.encoder_file, map_location=args.device))

    return encoder


def load_decoder(args, loc_max, loc_min, vel_max, vel_min):
    if args.global_temp:
        if args.decoder == "mlp":
            decoder = MLPDecoderGlobalTemp(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
                latent_dim=args.latent_dim,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoderGlobalTemp(
                loc_max, loc_min, vel_max, vel_min, args.suffix
            )
    else:
        if args.decoder == "mlp":
            decoder = MLPDecoder(
                args,
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "rnn":
            decoder = RNNDecoder(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

    decoder, num_GPU = distribute_over_GPUs(args, decoder, num_GPU=args.num_GPU)
    # print("Let's use", num_GPU, "GPUs!")

    if args.load_folder:
        print("Loading model file")
        args.decoder_file = os.path.join(args.load_folder, "decoder.pt")
        decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
        args.save_folder = False

    return decoder


def load_model(args, loc_max, loc_min, vel_max, vel_min):

    decoder = load_decoder(args, loc_max, loc_min, vel_max, vel_min)

    if args.use_encoder:
        encoder = load_encoder(args)
        edge_probs = None
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr,
        )
    else:
        encoder = None
        edge_probs = load_distribution(args)
        optimizer = optim.Adam(
            [{"params": edge_probs, "lr": args.lr_z}]
            + [{"params": decoder.parameters(), "lr": args.lr}]
        )

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )

    return (
        encoder,
        decoder,
        optimizer,
        scheduler,
        edge_probs,
    )


#---------------------------------------------------
# modules
#---------------------------------------------------

# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import torch

# from model import utils

class MLP(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, use_batch_norm=True, final_linear=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.use_batch_norm = use_batch_norm
        self.final_linear = final_linear
        if self.final_linear:
            self.fc_final = nn.Linear(n_out, n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.final_linear:
            x = self.fc_final(x)
        if self.use_batch_norm:
            return self.batch_norm(x)
        else:
            return x


class CNN(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


#---------------------------------------------------
# Encoder
#---------------------------------------------------

# from abc import abstractmethod
# import torch

# from model.modules import *


class Encoder(nn.Module):
    def __init__(self, args, factor=True):
        super(Encoder, self).__init__()
        self.args = args
        self.factor = factor

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(
            inputs.size(0) * receivers.size(1), inputs.size(2), inputs.size(3)
        )
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(
            inputs.size(0) * senders.size(1), inputs.size(2), inputs.size(3)
        )
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders,receivers], dim=2)
        return edges

    @abstractmethod
    def forward(self, inputs, rel_rec, rel_send, mask_idx=None):
        pass

#---------------------------------------------------
# MLPEncoder
#---------------------------------------------------

# import torch

# from model.modules import *
# from model.Encoder import Encoder

class MLPEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, factor)

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)



#---------------------------------------------------
# CNNEncoder
#---------------------------------------------------


# from model.modules import *
# from model.Encoder import Encoder

_EPS = 1e-10


class CNNEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True, n_in_mlp1=None
    ):
        super().__init__(args, factor)

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)

        if n_in_mlp1 is None:
            n_in_mlp1 = n_hid
        self.mlp1 = MLP(n_in_mlp1, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)

        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)


#---------------------------------------------------
# MLPEncoderUnobserved
#---------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model import utils, utils_unobserved
# from model.MLPEncoder import MLPEncoder

_EPS = 1e-10

class MLPEncoderUnobserved(MLPEncoder):
    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, n_in, n_hid, n_out, do_prob, factor)

        self.unobserved = args.unobserved

        self.lstm1 = nn.LSTM(
            (args.num_atoms - self.unobserved) * args.dims,
            n_hid,
            bidirectional=True,
            dropout=do_prob,
        )
        self.lstm2 = nn.LSTM(n_hid * 2, args.dims, bidirectional=False, dropout=do_prob)

        self.init_weights()
        print("Using unobserved encoder.")

    def evaluate_unobserved(self, unobserved, target):
        return F.mse_loss(torch.squeeze(unobserved), torch.squeeze(target))

    def calc_unobserved_q(self, unobserved):
        ### Gaussian prior
        unobserved_mu = self.fc_mu(unobserved)
        unobserved_log_sigma = self.fc_logsigma(unobserved)

        unobserved = sample_normal_from_latents(
            unobserved_mu,
            unobserved_log_sigma,
            downscale_factor=self.args.prior_downscale,
        )

        loss_kl_latent = kl_normal_reverse(
            0,
            1,
            unobserved_mu,
            unobserved_log_sigma,
            downscale_factor=self.args.prior_downscale,
        )
        return unobserved, loss_kl_latent

    def forward(self, inputs, rel_rec, rel_send, mask_idx=0):
        timesteps = inputs.size(2)

        # input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        observed = remove_unobserved(self.args, inputs, mask_idx)

        observed = observed.permute(2, 0, 1, 3)
        observed = observed.reshape(observed.size(0), observed.size(1), -1)
        unobserved, _ = self.lstm1(observed)
        unobserved, _ = self.lstm2(unobserved)
        unobserved = unobserved.unsqueeze(0).permute(2, 0, 1, 3)
        unobserved = torch.reshape(
            unobserved, [unobserved.size(0), unobserved.size(1), timesteps, -1]
        )
        # output shape: [num_sims, num_atoms, num_timesteps, num_dims]

        target_unobserved = inputs[:, mask_idx, :, :]
        mse_unobserved = self.evaluate_unobserved(unobserved, target_unobserved)

        data_encoder = torch.cat(
            (inputs[:, :mask_idx, :], unobserved, inputs[:, mask_idx + 1 :, :],), dim=1,
        )

        output = super().forward(data_encoder, rel_rec, rel_send)

        return (output, unobserved, mse_unobserved)



#---------------------------------------------------
# EncoderGLobalTemp
#---------------------------------------------------

# from model.modules import *
# from model.MLPEncoder import MLPEncoder
# from model.CNNEncoder import CNNEncoder


class CNNEncoderGlobalTemp(CNNEncoder):
    def __init__(
        self,
        args,
        n_in,
        n_hid,
        n_out,
        do_prob=0.0,
        factor=True,
        latent_dim=2,
        latent_sample_dim=1,
        num_atoms=5,
        num_timesteps=49,
    ):
        super().__init__(
            args,
            n_in,
            n_hid,
            n_out,
            do_prob,
            factor,
            n_in_mlp1=n_hid + latent_sample_dim,
        )

        self.mlp4_confounder = MLP(
            n_in * num_timesteps * num_atoms,
            n_hid,
            latent_dim,
            do_prob,
            use_batch_norm=False,
            final_linear=True,
        )
        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)

        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x_latent_input = inputs.view(inputs.size(0), 1, -1)
        latents = self.mlp4_confounder(x_latent_input).squeeze(1)

        inferred_mu, inferred_width = utils.get_uniform_parameters_from_latents(latents)
        latent_sample = utils.sample_uniform_from_latents(inferred_mu, inferred_width)
        l = latent_sample.view(latent_sample.size(0), 1, latent_sample.size(1)).repeat(
            1, x.size(1), 1
        )
        l = l.detach()
        # l = latents.view(latents.size(0), 1, latents.size(1)).repeat(1, x.size(1), 1)

        x = self.mlp1(torch.cat([x, l], 2))  # 2-layer ELU net per node
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x), latent_sample, inferred_mu, inferred_width



#---------------------------------------------------
# EncoderGlobalTemp
#---------------------------------------------------

# from model.modules import *
# from model.MLPEncoder import MLPEncoder
# from model.CNNEncoder import CNNEncoder


class CNNEncoderGlobalTemp(CNNEncoder):
    def __init__(
        self,
        args,
        n_in,
        n_hid,
        n_out,
        do_prob=0.0,
        factor=True,
        latent_dim=2,
        latent_sample_dim=1,
        num_atoms=5,
        num_timesteps=49,
    ):
        super().__init__(
            args,
            n_in,
            n_hid,
            n_out,
            do_prob,
            factor,
            n_in_mlp1=n_hid + latent_sample_dim,
        )

        self.mlp4_confounder = MLP(
            n_in * num_timesteps * num_atoms,
            n_hid,
            latent_dim,
            do_prob,
            use_batch_norm=False,
            final_linear=True,
        )
        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)

        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x_latent_input = inputs.view(inputs.size(0), 1, -1)
        latents = self.mlp4_confounder(x_latent_input).squeeze(1)

        inferred_mu, inferred_width = utils.get_uniform_parameters_from_latents(latents)
        latent_sample = utils.sample_uniform_from_latents(inferred_mu, inferred_width)
        l = latent_sample.view(latent_sample.size(0), 1, latent_sample.size(1)).repeat(
            1, x.size(1), 1
        )
        l = l.detach()
        # l = latents.view(latents.size(0), 1, latents.size(1)).repeat(1, x.size(1), 1)

        x = self.mlp1(torch.cat([x, l], 2))  # 2-layer ELU net per node
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x), latent_sample, inferred_mu, inferred_width



#---------------------------------------------------
# MLPDecoder
#---------------------------------------------------

# import torch

# from model.modules import *

class MLPDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self,
        args,
        n_in_node,
        edge_types,
        msg_hid,
        msg_out,
        n_hid,
        do_prob=0.0,
        skip_first=False,
    ):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]
        )
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

    def single_step_forward(
        self, single_timestep_inputs, rel_rec, rel_send, single_timestep_rel_type
    ):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
        )

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i : i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [
            rel_type.size(0),
            inputs.size(1),
            rel_type.size(1),
            rel_type.size(2),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1).expand(
            sizes
        )  # copy relations over sequence length

        time_steps = inputs.size(1)
        assert pred_steps <= time_steps
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(
                last_pred, rel_rec, rel_send, curr_rel_type
            )
            preds.append(last_pred)

        sizes = [
            preds[0].size(0),
            preds[0].size(1) * pred_steps,
            preds[0].size(2),
            preds[0].size(3),
        ]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, : (inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()



#---------------------------------------------------
# SimulationDecoder
#---------------------------------------------------

# import torch.nn as nn
# import torch


class SimulationDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoder, self).__init__()

        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        self.interaction_type = suffix

        if "_springs" in self.interaction_type:
            print("Using spring simulation decoder.")
            self.interaction_strength = 0.1
            # original simulation used sample_freq, _delta_T = 100, 0.001
            # we use 1, 0.1 instead for computational efficiency
            self.sample_freq = 1
            self._delta_T = 0.1
            self.box_size = 5.0
        else:
            print("Simulation type could not be inferred from suffix.")

        self.out = None

        # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

        self._max_F = 0.1 / self._delta_T

    def unnormalize(self, loc, vel):
        loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
        vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
        return loc, vel

    def renormalize(self, loc, vel):
        loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
        vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        return loc, vel

    def clamp(self, loc, vel):
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -torch.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = torch.abs(vel[under])

        return loc, vel

    def get_offdiag_indices(self, num_nodes):
        """Linear off-diagonal indices."""
        ones = torch.ones(num_nodes, num_nodes)
        eye = torch.eye(num_nodes, num_nodes)
        offdiag_indices = (ones - eye).nonzero().t()
        offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
        return offdiag_indices

    def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = self.get_offdiag_indices(inputs.size(1))
        edges = torch.zeros(relations.size(0), inputs.size(1) * inputs.size(1))

        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1), inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            forces_size = -self.interaction_strength * edges
            pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)

            # Tricks for parallel processing of time steps
            pair_dist = pair_dist.view(
                inputs.size(0), (inputs.size(2) - 1), inputs.size(1), inputs.size(1), 2,
            )
            forces = (forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(3)

            forces = forces.view(
                inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2
            )

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out



#---------------------------------------------------
# DecoderGlobalTemp
#---------------------------------------------------


# from model.modules import *
# from model.SimulationDecoder import SimulationDecoder
# from model import utils


class MLPDecoderGlobalTemp(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self,
        n_in_node,
        edge_types,
        msg_hid,
        msg_out,
        n_hid,
        do_prob=0.0,
        skip_first=False,
        latent_dim=32,
    ):
        super(MLPDecoderGlobalTemp, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            # [nn.Linear(2 * n_in_node + latent_dim, msg_hid) for _ in range(edge_types)]
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]
        )
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

    def single_step_forward(
        self,
        single_timestep_inputs,
        latents,
        rel_rec,
        rel_send,
        single_timestep_rel_type,
    ):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
        )

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i : i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, latents, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [
            rel_type.size(0),
            inputs.size(1),
            rel_type.size(1),
            rel_type.size(2),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1).expand(
            sizes
        )  # copy relations over sequence length

        time_steps = inputs.size(1)
        assert pred_steps <= time_steps
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(
                last_pred, latents, rel_rec, rel_send, curr_rel_type
            )
            preds.append(last_pred)

        sizes = [
            preds[0].size(0),
            preds[0].size(1) * pred_steps,
            preds[0].size(2),
            preds[0].size(3),
        ]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, : (inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class SimulationDecoderGlobalTemp(SimulationDecoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoderGlobalTemp, self).__init__(
            loc_max, loc_min, vel_max, vel_min, suffix
        )

    def forward(self, inputs, relations, latents, rel_rec, rel_send, pred_steps=1):
        temperature = latents.unsqueeze(2)
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = get_offdiag_indices(inputs.size(1))
        edges = torch.zeros(relations.size(0), inputs.size(1) * inputs.size(1))

        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1), inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            if "_springs" in self.interaction_type:
                forces_size = -temperature * edges
                pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)

                # Tricks for parallel processing of time steps
                pair_dist = pair_dist.view(
                    inputs.size(0),
                    (inputs.size(2) - 1),
                    inputs.size(1),
                    inputs.size(1),
                    2,
                )
                forces = (forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(3)
            else:  # charged particle sim
                e = (-1) * (edges * 2 - 1)
                forces_size = -temperature * e

                l2_dist_power3 = torch.pow(self.pairwise_sq_dist(loc), 3.0 / 2.0)
                l2_dist_power3 = self.set_diag_to_one(l2_dist_power3)

                l2_dist_power3 = l2_dist_power3.view(
                    inputs.size(0), (inputs.size(2) - 1), inputs.size(1), inputs.size(1)
                )
                forces_size = forces_size.unsqueeze(1) / (l2_dist_power3 + _EPS)

                pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)
                pair_dist = pair_dist.view(
                    inputs.size(0),
                    (inputs.size(2) - 1),
                    inputs.size(1),
                    inputs.size(1),
                    2,
                )
                forces = (forces_size.unsqueeze(-1) * pair_dist).sum(3)

            forces = forces.view(
                inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2
            )

            if "_charged" in self.interaction_type:  # charged particle sim
                # Clip forces
                forces[forces > self._max_F] = self._max_F
                forces[forces < -self._max_F] = -self._max_F

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out



#---------------------------------------------------
# argparser
#---------------------------------------------------


# import argparse
# import torch
# import datetime
# import numpy as np


def parse_args(
    seed = 969491451,
    GPU_to_use=None,
    epochs = 3, #<--- #500
    batch_size=128,
    lr=0.0005,
    lr_decay=200,
    gamma=0.5,
    training_samples=0,
    test_samples=0,
    shuffle_traindata=True, 
    prediction_steps=10,
    encoder_hidden=256,
    decoder_hidden=256,
    encoder='mlp',
    decoder='mlp', 
    prior=1,
    edge_types=2, #?
    dont_use_encoder=False,
    lr_z=0.1,
    global_temp=False,
    alpha=2,
    num_cats=3,
    unobserved=0,
    model_unobserved=0,
    dont_shuffle_unobserved=False,
    teacher_forcing=0,
    suffix='_energy1', #<--- #_springs5_s200
    timesteps= 24, #<---
    num_atoms= 7, #<---
    dims= 1, #<---
    datadir='./data',
    save_folder='logs',
    expername="",
    sym_save_folder="../logs",
    load_folder='',
    test_time_adapt=False,
    lr_logits=0.01,
    num_tta_steps=100,
    dont_skip_first=False,
    temp=0.5,
    hard=False,
    no_validate=False,
    no_cuda=False,
    var=5e-7,
    encoder_dropout=0.0,
    decoder_dropout=0.0,
    no_factor=False
    ):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", 
                        type=int, 
                        default=seed, #42,
                        help="Random seed.")
    parser.add_argument(
        "--GPU_to_use", type=int, default=GPU_to_use, help="GPU to use for training"
    )

    ############## training hyperparameter ##############
    parser.add_argument(
        "--epochs", type=int, 
        default = epochs,
        # default=500, 
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=batch_size, #128, 
        help="Number of samples per batch."
    )
    parser.add_argument(
        "--lr", type=float, default=lr, #0.0005, 
        help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_decay",
        type=int,
        default=lr_decay,#200,
        help="After how epochs to decay LR by a factor of gamma.",
    )
    parser.add_argument("--gamma", type=float, default=gamma,#0.5
                        help="LR decay factor.")
    parser.add_argument(
        "--training_samples", type=int, default=training_samples,
        help="If 0 use all data available, otherwise reduce number of samples to given number"
    )
    parser.add_argument(
        "--test_samples", type=int, default=test_samples,
        help="If 0 use all data available, otherwise reduce number of samples to given number"
    )
    parser.add_argument(
        "--shuffle_traindata", 
        action="store_true",
        default=shuffle_traindata,
        help="If False, DataLoader for training data will provide shuffled batches, unshuffled o.w."
    )
    parser.add_argument(
        "--prediction_steps",
        type=int,
        default=prediction_steps,#10,
        metavar="N",
        help="Num steps to predict before re-using teacher forcing.",
    )

    ############## architecture ##############
    parser.add_argument(
        "--encoder_hidden", type=int, default=encoder_hidden,#256, 
        help="Number of hidden units."
    )
    parser.add_argument(
        "--decoder_hidden", type=int, default=decoder_hidden,#256, 
        help="Number of hidden units."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=encoder,#"mlp",
        help="Type of path encoder model (mlp or cnn).",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=decoder,#"mlp",
        help="Type of decoder model (mlp, rnn, or sim).",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=prior,#1,
        help="Weight for sparsity prior (if == 1, uniform prior is applied)",
    )
    parser.add_argument(
        "--edge_types",
        type=int,
        default=edge_types,#2,
        help="Number of different edge-types to model",
    )

    ########### Different variants for variational distribution q ###############
    parser.add_argument(
        "--dont_use_encoder",
        action="store_true",
        default=dont_use_encoder,
        help="If true, replace encoder with distribution to be estimated",
    )
    parser.add_argument(
        "--lr_z",
        type=float,
        default=lr_z,#0.1,
        help="Learning rate for distribution estimation.",
    )

    ### global latent temperature ###
    parser.add_argument(
        "--global_temp",
        action="store_true",
        default=global_temp,
        help="Should we model temperature confounding?",
    )
    parser.add_argument(
        "--load_temperatures",
        help="Should we load temperature data?",
        action="store_true",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=alpha,
        help="Middle value of temperature distribution.",
    )
    parser.add_argument(
        "--num_cats",
        type=int,
        default=num_cats,
        help="Number of categories in temperature distribution.",
    )

    ### unobserved time-series ###
    parser.add_argument(
        "--unobserved",
        type=int,
        default=unobserved,
        help="Number of time-series to mask from input.",
    )
    parser.add_argument(
        "--model_unobserved",
        type=int,
        default=model_unobserved,
        help="If 0, use NRI to infer unobserved particle. "
        "If 1, removes unobserved from data. "
        "If 2, fills empty slot with mean of observed time-series (mean imputation)",
    )
    parser.add_argument(
        "--dont_shuffle_unobserved",
        action="store_true",
        default=dont_shuffle_unobserved,
        help="If true, always mask out last particle in trajectory. "
        "If false, mask random particle.",
    )
    parser.add_argument(
        "--teacher_forcing",
        type=int,
        default=teacher_forcing,
        help="Factor to determine how much true trajectory of "
        "unobserved particle should be used to learn prediction.",
    )

    ############## loading and saving ##############
    parser.add_argument(
        "--suffix",
        type=str,
        default=suffix,
        # default="_springs5",
        help='Suffix for training data.',
    )
    parser.add_argument(
        "--timesteps", type=int, default=timesteps, #49, 
        help="Number of timesteps in input."
    )
    parser.add_argument(
        "--num_atoms", type=int, default= num_atoms,#5, 
        help="Number of time-series in input."
    )
    parser.add_argument(
        "--dims", type=int, default= dims,#4, 
        help="Dimensionality of input."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=datadir,#"./data",
        help="Name of directory where data is stored.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=save_folder,
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--expername",
        type=str,
        default=expername,
        help="If given, creates a symlinked directory by this name in logdir"
        "linked to the results file in save_folder"
        "(be careful, this can overwrite previous results)",
    )
    parser.add_argument(
        "--sym_save_folder",
        type=str,
        default=sym_save_folder,
        help="Name of directory where symlinked named experiment is created."
    )
    parser.add_argument(
        "--load_folder",
        type=str,
        default="",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )

    ############## fine tuning ##############
    parser.add_argument(
        "--test_time_adapt",
        action="store_true",
        default=test_time_adapt,
        help="Test time adapt q(z) on first half of test sequences.",
    )
    parser.add_argument(
        "--lr_logits",
        type=float,
        default=lr_logits, #0.01,
        help="Learning rate for test-time adapting logits.",
    )
    parser.add_argument(
        "--num_tta_steps",
        type=int,
        default=num_tta_steps, #100,
        help="Number of test-time-adaptation steps per batch.",
    )

    ############## almost never change these ##############
    parser.add_argument(
        "--dont_skip_first",
        action="store_true",
        default=dont_skip_first,
        help="If given as argument, do not skip first edge type in decoder, i.e. it represents no-edge.",
    )
    parser.add_argument(
        "--temp", type=float, default=temp, help="Temperature for Gumbel softmax."
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        default=hard,
        help="Uses discrete samples in training forward pass.",
    )
    parser.add_argument(
        "--no_validate", action="store_true", default=no_validate, help="Do not validate results throughout training."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=no_cuda, #False, 
        help="Disables CUDA training."
    )
    parser.add_argument("--var", type=float, default=var, help="Output variance.")
    parser.add_argument(
        "--encoder_dropout",
        type=float,
        default=encoder_dropout, #0.0,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--decoder_dropout",
        type=float,
        default=decoder_dropout, #0.0,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--no_factor",
        action="store_true",
        default=no_factor,
        help="Disables factor graph model.",
    )
    ########################################################
    parser.add_argument('-f')
    
    args = parser.parse_args()
    args.test = True

    ### Presets for different datasets ###
    if (
        "fixed" in args.suffix
        or "uninfluenced" in args.suffix
        or "influencer" in args.suffix
        or "conf" in args.suffix
    ):
        args.dont_shuffle_unobserved = True

    if "netsim" in args.suffix:
        args.dims = 1
        args.num_atoms = 15
        args.timesteps = 200
        args.no_validate = True
        args.test = False

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    args.validate = not args.no_validate
    args.shuffle_unobserved = not args.dont_shuffle_unobserved
    args.skip_first = not args.dont_skip_first
    args.use_encoder = not args.dont_use_encoder
    # args.time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")#.isoformat()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        torch.cuda.manual_seed(args.seed)
        args.num_GPU = 1  # torch.cuda.device_count()
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size

    return args


#---------------------------------------------------
# logger
#---------------------------------------------------

# import time
# import os
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import pandas as pd
# from collections import defaultdict
# import itertools


class Logger:
    def __init__(self, args):
        self.args = args

        self.train_losses = pd.DataFrame()
        self.train_losses_idx = 0

        self.test_losses = pd.DataFrame()
        self.test_losses_idx = 0

        if args.validate:
            self.val_losses = pd.DataFrame()
            self.val_losses_idx = 0
        else:
            self.val_losses = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

        self.create_log_path(args)

    def create_log_path(self, args, add_path_var=""):

        print(type(args.time))
        args.log_path = os.path.join(args.save_folder, args.time)#add_path_var, args.time)

        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)

        if args.expername != "":
            sympath = os.path.join(args.sym_save_folder, args.expername)
            if os.path.islink(sympath):
                os.remove(sympath)
            ## check whether args.log_path is absolute path and if not concatenate with current working directory
            if os.path.isabs(args.log_path):
                log_link = args.log_path
            else:
                log_link = os.path.join(os.getcwd(), args.log_path)
            os.symlink(log_link, sympath)

        self.log_file = os.path.join(args.log_path, "log.txt")
        self.write_to_log_file(args)

        args.encoder_file = os.path.join(args.log_path, "encoder.pt")
        args.decoder_file = os.path.join(args.log_path, "decoder.pt")
        args.optimizer_file = os.path.join(args.log_path, "optimizer.pt")

        args.plotdir = os.path.join(args.log_path, "plots")
        if not os.path.exists(args.plotdir):
            os.makedirs(args.plotdir)

    def save_checkpoint(self, args, encoder, decoder, optimizer, specifier="", rnn=None):
        args.encoder_file = os.path.join(args.log_path, "encoder" + specifier + ".pt")
        args.decoder_file = os.path.join(args.log_path, "decoder" + specifier + ".pt")
        args.optimizer_file = os.path.join(
            args.log_path, "optimizer" + specifier + ".pt"
        )

        if encoder is not None:
            torch.save(encoder.state_dict(), args.encoder_file)
        if decoder is not None:
            torch.save(decoder.state_dict(), args.decoder_file)
        if rnn is not None:
            args.rnn_file = os.path.join(args.log_path, "rnn" + specifier + ".pt")
            torch.save(rnn.state_dict(), args.rnn_file)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), args.optimizer_file)

    def write_to_log_file(self, string):
        """
        Write given string in log-file and print as terminal output
        """
        print(string)
        cur_file = open(self.log_file, "a")
        print(string, file=cur_file)
        cur_file.close()

    def create_log(
        self,
        args,
        encoder=None,
        decoder=None,
        rnn=None,
        accuracy=None,
        optimizer=None,
        final_test=False,
        test_losses=None,
    ):

        print("Saving model and log-file to " + args.log_path)

        # Save losses throughout training and plot
        self.train_losses.to_pickle(os.path.join(self.args.log_path, "train_loss"))#if error occurs, omit .csv again
        self.train_losses.to_csv(os.path.join(self.args.log_path, "train_loss.csv"), index=False)

        if self.val_losses is not None:
            self.val_losses.to_pickle(os.path.join(self.args.log_path, "val_loss"))
            self.val_losses.to_csv(os.path.join(self.args.log_path, "val_loss.csv"), index=False)


        if accuracy is not None:
            np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

        specifier = ""
        if final_test:
            pd_test_losses = pd.DataFrame(
                [
                    [k] + [np.mean(v)]
                    for k, v in test_losses.items()
                    if type(v) != defaultdict
                ],
                columns=["loss", "score"],
            )
            pd_test_losses.to_pickle(os.path.join(self.args.log_path, "test_loss"))
            pd_test_losses.to_csv(os.path.join(self.args.log_path, "test_loss.csv"), index=False)

            pd_test_losses_per_influenced = pd.DataFrame(
                list(
                    itertools.chain(
                        *[
                            [
                                [k]
                                + [idx]
                                + [np.mean(list(itertools.chain.from_iterable(elem)))]
                                for idx, elem in sorted(v.items())
                            ]
                            for k, v in test_losses.items()
                            if type(v) == defaultdict
                        ]
                    )
                ),
                columns=["loss", "num_influenced", "score"],
            )
            pd_test_losses_per_influenced.to_pickle(
                os.path.join(self.args.log_path, "test_loss_per_influenced")
            )

            specifier = "final"

        # Save the model checkpoint
        self.save_checkpoint(args, encoder, decoder, optimizer, specifier=specifier, rnn=rnn)

    def draw_loss_curves(self):
        for i in self.train_losses.columns:
            plt.figure()
            plt.plot(self.train_losses[i], "-b", label="train " + i)

            if self.val_losses is not None and i in self.val_losses:
                plt.plot(self.val_losses[i], "-r", label="val " + i)

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")

            # save image
            plt.savefig(os.path.join(self.args.log_path, i + ".png"))
            plt.close()

    def append_train_loss(self, loss):
        for k, v in loss.items():
            self.train_losses.at[str(self.train_losses_idx), k] = np.mean(v)
        self.train_losses_idx += 1

    def append_val_loss(self, val_loss):
        for k, v in val_loss.items():
            self.val_losses.at[str(self.val_losses_idx), k] = np.mean(v)
        self.val_losses_idx += 1

    def append_test_loss(self, test_loss):
        for k, v in test_loss.items():
            if type(v) != defaultdict:
                self.test_losses.at[str(self.test_losses_idx), k] = np.mean(v)
        self.test_losses_idx += 1

    def result_string(self, trainvaltest, epoch, losses, t=None):
        string = ""
        if trainvaltest == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + trainvaltest + "\t \t"

        for loss, value in losses.items():
            if type(value) == defaultdict:
                string += loss + " "
                for idx, elem in sorted(value.items()):
                    string += str(idx) + ": {:.10f} \t".format(
                        np.mean(list(itertools.chain.from_iterable(elem)))
                    )
            elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
                string += loss + " {:.10f} \t".format(np.mean(value))

        if t is not None:
            string += "time: {:.4f}s \t".format(time.time() - t)

        return string




#---------------------------------------------------
# data_loader
#---------------------------------------------------


# import os
# import numpy as np
# import torch
# from torch.utils.data.dataset import TensorDataset
# from torch.utils.data import DataLoader


def load_data(args):
    loc_max, loc_min, vel_max, vel_min = None, None, None, None
    train_loader, valid_loader, test_loader = None, None, None

    if "kuramoto" in args.suffix:
        train_loader, valid_loader, test_loader = load_ode_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir,
        )
    elif "netsim" in args.suffix:
        train_loader, loc_max, loc_min = load_netsim_data(
            batch_size=args.batch_size_multiGPU, 
            datadir=args.datadir
        )
    elif "springs" in args.suffix:
        (
            train_loader,
            valid_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        ) = load_springs_data(
            args, 
            args.batch_size_multiGPU, 
            args.suffix, 
            datadir=args.datadir
        )
    elif "energy" in args.suffix:
        train_loader, valid_loader, test_loader = load_energy_data(
            args,
            suffix=args.suffix,
            batch_size=args.batch_size_multiGPU,
            datadir=args.datadir
        )
    else:
        raise NameError("Unknown data to be loaded")

    return train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min


def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1


def denormalize(x, x_min, x_max): # my addition to later look at prediction
    return (x + 1) * (x_max - x_min)/2 + x_min


def remove_unobserved_from_data(loc, vel, edge, args):
    loc = loc[:, :, :, : -args.unobserved]
    vel = vel[:, :, :, : -args.unobserved]
    edge = edge[:, : -args.unobserved, : -args.unobserved]
    return loc, vel, edge


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )


def data_preparation(
    loc,
    vel,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    temperature=None,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    feat = np.concatenate([loc, vel], axis=3)
    edges = np.reshape(edges, [-1, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)

    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)

    edges = edges[:, off_diag_idx]

    if temperature is not None:
        dataset = TensorDataset(feat, edges, temperature)
    else:
        dataset = TensorDataset(feat, edges)

    return dataset


def load_springs_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    print("Loading data from {}".format(datadir))
    loc_train = np.load(os.path.join(datadir, "loc_train" + suffix + ".npy"))
    vel_train = np.load(os.path.join(datadir, "vel_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))

    loc_valid = np.load(os.path.join(datadir, "loc_valid" + suffix + ".npy"))
    vel_valid = np.load(os.path.join(datadir, "vel_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))

    loc_test = np.load(os.path.join(datadir, "loc_test" + suffix + ".npy"))
    vel_test = np.load(os.path.join(datadir, "vel_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    if args.load_temperatures:
        temperatures_train, temperatures_valid, temperatures_test = load_temperatures(
            suffix=suffix, datadir=datadir
        )
    else:
        temperatures_train, temperatures_valid, temperatures_test = None, None, None

    # [num_samples, num_timesteps, num_dims, num_atoms]
    if args.training_samples != 0:
        loc_train = loc_train[: args.training_samples]
        vel_train = vel_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        loc_test = loc_test[: args.test_samples]
        vel_test = vel_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(args.num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_train,
    )
    valid_data = data_preparation(
        loc_valid,
        vel_valid,
        edges_valid,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_valid,
    )
    test_data = data_preparation(
        loc_test,
        vel_test,
        edges_test,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        args.num_atoms,
        temperature=temperatures_test,
    )
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    return (
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    )


def load_temperatures(suffix="", datadir="data"):
    temperatures_train = np.load(
        os.path.join(datadir, "temperatures_train" + suffix + ".npy")
    )
    temperatures_valid = np.load(
        os.path.join(datadir, "temperatures_valid" + suffix + ".npy")
    )
    temperatures_test = np.load(
        os.path.join(datadir, "temperatures_test" + suffix + ".npy")
    )

    temperatures_train = torch.FloatTensor(temperatures_train)
    temperatures_valid = torch.FloatTensor(temperatures_valid)
    temperatures_test = torch.FloatTensor(temperatures_test)

    return temperatures_train, temperatures_valid, temperatures_test


def load_ode_data(args, batch_size=1, suffix="", datadir="data"):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    feat_train = np.load(os.path.join(datadir, "feat_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))
    feat_valid = np.load(os.path.join(datadir, "feat_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))
    feat_test = np.load(os.path.join(datadir, "feat_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    if args.training_samples != 0:
        feat_train = feat_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        feat_test = feat_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )  # , num_workers=8
    # )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size
    )  # , num_workers=8) ##THIS

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_for_ncg(datadir, data_index, suffix):
    """Data loading for Neural Granger Causality method (one example at a time)."""
    feat_train = np.load(os.path.join(datadir, "feat_train_small" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train_small" + suffix + ".npy"))
    return feat_train[data_index], edges_train[data_index]


def load_netsim_data(batch_size=1, datadir="data"):
    print("Loading data from {}".format(datadir))

    subject_id = [1, 2, 3, 4, 5]

    print("Loading data for subjects ", subject_id)

    loc_train = torch.zeros(len(subject_id), 15, 200)
    edges_train = torch.zeros(len(subject_id), 15, 15)

    for idx, elem in enumerate(subject_id):
        fileName = "sim3_subject_%s.npz" % (elem)
        ld = np.load(os.path.join(datadir, "netsim", fileName))
        loc_train[idx] = torch.FloatTensor(ld["X_np"])
        edges_train[idx] = torch.LongTensor(ld["Gref"])

    # [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = loc_train.unsqueeze(-1)

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    loc_train = normalize(loc_train, loc_min, loc_max)

    # Exclude self edges
    num_atoms = loc_train.shape[1]

    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = torch.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = (edges_train + 1) // 2
    edges_train = edges_train[:, off_diag_idx]

    train_data = TensorDataset(loc_train, edges_train)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return (train_data_loader, loc_max, loc_min)

def unpack_batches(args, minibatch):
    if args.load_temperatures:
        (data, relations, temperatures) = minibatch
    else:
        (data, relations) = minibatch
        temperatures = None
    if args.cuda:
        data, relations = data.cuda(), relations.cuda()
        if args.load_temperatures:
            temperatures = temperatures.cuda()
    return data, relations, temperatures


#### own addition for energy data
def load_energy_data(args, batch_size=1, suffix="", datadir="data"):
    print("Loading data from {}".format(datadir))
    feat_train = np.load(os.path.join(datadir, "feat_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(datadir, "edges_train" + suffix + ".npy"))
    feat_valid = np.load(os.path.join(datadir, "feat_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(datadir, "edges_valid" + suffix + ".npy"))
    feat_test  = np.load(os.path.join(datadir, "feat_test" + suffix + ".npy"))
    edges_test  = np.load(os.path.join(datadir, "edges_test" + suffix + ".npy"))
    if 'lstm' in suffix:
        target_train = np.load(os.path.join(datadir, "target_train" + suffix + ".npy"))
        target_valid = np.load(os.path.join(datadir, "target_valid" + suffix + ".npy"))
        target_test  = np.load(os.path.join(datadir, "target_test" + suffix + ".npy"))
  
    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]
    
    if args.training_samples != 0:
        feat_train = feat_train[: args.training_samples]
        edges_train = edges_train[: args.training_samples]

    if args.test_samples != 0:
        feat_test = feat_test[: args.test_samples]
        edges_test = edges_test[: args.test_samples]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)
    
    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
    
    if 'lstm' not in suffix:
        train_data = TensorDataset(feat_train, edges_train)
        valid_data = TensorDataset(feat_valid, edges_valid)
        test_data = TensorDataset(feat_test, edges_test)
    else: 
        target_train = torch.FloatTensor(target_train)
        target_valid = torch.FloatTensor(target_valid) 
        target_test = torch.FloatTensor(target_test)
        train_data = TensorDataset(feat_train, target_train)
        valid_data = TensorDataset(feat_valid, target_valid)
        test_data = TensorDataset(feat_test, target_test)

    train_data_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=args.shuffle_traindata #probably what i need without timedims
        # shuffle=True #original
    )  # , num_workers=8
    # )
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size
    )  # , num_workers=8) ##THIS

    return train_data_loader, valid_data_loader, test_data_loader


#---------------------------------------------------
# forward_pass_and_eval
#---------------------------------------------------

# from __future__ import division
# from __future__ import print_function

# from collections import defaultdict
# import time
# import torch

# import numpy as np

# from model.modules import *
# from model import utils, utils_unobserved


def test_time_adapt(
    args,
    logits,
    decoder,
    data_encoder,
    rel_rec,
    rel_send,
    predicted_atoms,
    log_prior,
):
    with torch.enable_grad():
        tta_data_decoder = data_encoder.detach()

        if args.use_encoder:
            ### initialize q(z) with q(z|x)
            tta_logits = logits.detach()
            tta_logits.requires_grad = True
        else:
            ### initialize q(z) randomly
            tta_logits = torch.randn_like(
                logits, device=args.device.type, requires_grad=True
            )

        tta_optimizer = torch.optim.Adam(
            [{"params": tta_logits, "lr": args.lr_logits}]
        )
        tta_target = data_encoder[:, :, 1:, :].detach()

        ploss = 0
        for i in range(args.num_tta_steps):
            tta_optimizer.zero_grad()

            tta_edges = gumbel_softmax(tta_logits, tau=args.temp, hard=False)

            tta_output = decoder(
                tta_data_decoder, tta_edges, rel_rec, rel_send, args.prediction_steps
            )

            loss = nll_gaussian(tta_output, tta_target, args.var)

            prob = my_softmax(tta_logits, -1)

            if args.prior != 1:
                loss += kl_categorical(prob, log_prior, predicted_atoms) 
            else:
                loss += kl_categorical_uniform(
                    prob, predicted_atoms, args.edge_types
                ) 

            loss.backward()
            tta_optimizer.step()
            ploss += loss.cpu().detach()

            if i == 0:
                first_loss = loss.cpu().detach()
            if (i + 1) % 10 == 0:
                print(i, ": ", ploss / 10)
                ploss = 0

    print("Fine-tuning improvement: ", first_loss - loss.cpu().detach())

    return tta_logits

def forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    relations,
    rel_rec,
    rel_send,
    hard,
    data_encoder=None,
    data_decoder=None,
    edge_probs=None,
    testing=False,
    log_prior=None,
    temperatures=None
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))

    #################### INPUT DATA ####################
    diff_data_enc_dec = False
    if data_encoder is not None and data_decoder is not None:
        diff_data_enc_dec = True

    if data_encoder is None:
        data_encoder = data
    if data_decoder is None:
        data_decoder = data

    #################### DATA WITH UNOBSERVED TIME-SERIES ####################
    predicted_atoms = args.num_atoms
    if args.unobserved > 0:
        if args.shuffle_unobserved:
            mask_idx = np.random.randint(0, args.num_atoms)
        else:
            mask_idx = args.num_atoms - 1

        ### baselines ###
        if args.model_unobserved == 1:
            (
                data_encoder,
                data_decoder,
                predicted_atoms,
                relations,
            ) = baseline_remove_unobserved(
                args, data_encoder, data_decoder, mask_idx, relations, predicted_atoms
            )
            unobserved = 0
        if args.model_unobserved == 2:
            (
                data_encoder,
                unobserved,
                losses["mse_unobserved"],
            ) = baseline_mean_imputation(args, data_encoder, mask_idx)
            data_decoder = add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
    else:
        mask_idx = 0
        unobserved = 0

    #################### TEMPERATURE INFERENCE ####################
    if args.global_temp:
        ctp = args.categorical_temperature_prior
        cmax = ctp[-1]
        uniform_prior_mean = cmax
        uniform_prior_width = cmax 

    #################### ENCODER ####################
    if args.use_encoder:
        if args.unobserved > 0 and args.model_unobserved == 0:
            ## model unobserved time-series
            (
                logits,
                unobserved,
                losses["mse_unobserved"],
            ) = encoder(data_encoder, rel_rec, rel_send, mask_idx=mask_idx)
            data_decoder = add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
        elif args.global_temp:
            (logits, temperature_samples, 
                    inferred_mean, inferred_width) = encoder(
                            data_encoder, rel_rec, rel_send)
            temperature_samples *= 2 * cmax
            inferred_mean *= 2 * cmax 
            inferred_width *= 2 * cmax
        else:
            ## model only the edges
            logits = encoder(data_encoder, rel_rec, rel_send)
    else:
        logits = edge_probs.unsqueeze(0).repeat(data_encoder.shape[0], 1, 1)

    if args.test_time_adapt and args.num_tta_steps > 0 and testing:
        assert args.unobserved == 0, "No implementation for test-time adaptation when there are unobserved time-series."
        logits = test_time_adapt(
            args,
            logits,
            decoder,
            data_encoder,
            rel_rec,
            rel_send,
            predicted_atoms,
            log_prior,
        )

    edges = gumbel_softmax(logits, tau=args.temp, hard=hard)
    prob = my_softmax(logits, -1)

    target = data_decoder[:, :, 1:, :]

    #################### DECODER ####################
    if args.decoder == "rnn":
        output = decoder(
            data_decoder,
            edges,
            rel_rec,
            rel_send,
            pred_steps=args.prediction_steps,
            burn_in=True,
            burn_in_steps=args.timesteps - args.prediction_steps,
        )
    else:
        if args.global_temp:
            output = decoder(
                data_decoder, 
                edges, 
                temperature_samples, 
                rel_rec, 
                rel_send, 
                args.prediction_steps
            )
        else:
            output = decoder(
                data_decoder,
                edges,
                rel_rec,
                rel_send,
                args.prediction_steps,
            )

    #################### LOSSES ####################
    if args.unobserved > 0:
        if args.model_unobserved != 1:
            losses["mse_observed"] = calc_mse_observed(
                args, output, target, mask_idx
            )

            if not args.shuffle_unobserved:
                losses["observed_acc"] = edge_accuracy_observed(
                    logits, relations, num_atoms=args.num_atoms
                )
                losses["observed_auroc"] = calc_auroc_observed(
                    prob, relations, num_atoms=args.num_atoms
                )

    if args.global_temp:
        losses['loss_kl_temp'] = kl_uniform(inferred_width, uniform_prior_width)
        losses['temp_logprob'] = get_uniform_logprobs(
                inferred_mean.flatten(), inferred_width.flatten(), temperatures)
        targets = torch.eq(torch.reshape(ctp, [1, -1]), torch.reshape(temperatures, [-1, 1])).double()
        preds = get_preds_from_uniform(inferred_mean, inferred_width, ctp)

        losses['temp_precision'] = torch.sum(targets * preds) / torch.sum(preds)
        losses['temp_recall'] = torch.sum(targets * preds) / torch.sum(targets)
        losses['temp_corr'] = get_correlation(inferred_mean.flatten(), temperatures)

    ## calculate performance based on how many particles are influenced by unobserved one/last one
    if not args.shuffle_unobserved and args.unobserved > 0:
        losses = calc_performance_per_num_influenced(
            args,
            relations,
            output,
            target,
            logits,
            prob,
            mask_idx,
            losses
        )

    #################### MAIN LOSSES ####################
    ### latent losses ###
    losses["loss_kl"] = kl_latent(args, prob, log_prior, predicted_atoms)
    losses["acc"] = edge_accuracy(logits, relations)
    losses["auroc"] = calc_auroc(prob, relations)

    ### output losses ###
    losses["loss_nll"] = nll_gaussian(
        output, target, args.var
    ) 

    losses["loss_mse"] = F.mse_loss(output, target)

    total_loss = losses["loss_nll"] + losses["loss_kl"]
    total_loss += args.teacher_forcing * losses["mse_unobserved"]
    if args.global_temp:
        total_loss += losses['loss_kl_temp']
    losses["loss"] = total_loss

    losses["inference time"] = time.time() - start

    return losses, output, unobserved, edges

def forward_pass_rnn(
    args,
    rnn,
    data,
    target
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    
    output = rnn(data).unsqueeze(1)
    
    losses["loss_mse"] = F.mse_loss(output, target)
    losses["inference time"] = time.time() - start

    return losses, output