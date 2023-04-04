# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import warnings


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)
warnings.filterwarnings("ignore")


def positive(x):
    return np.maximum(x, 1e-14)


def chebyshev(true, pred):  # average Chebyshev distance
    return np.mean(np.max(np.abs(true - pred), axis=1))


def clark(true, pred):  # average Clark distance
    return np.mean(np.sqrt(np.sum((true - pred) ** 2 / positive((true + pred) ** 2), axis=1)))


def canberra(true, pred):  # average Canberra metric
    return np.mean(np.sum(np.abs(true - pred) / positive((true + pred)), axis=1))


def kld(true, pred):  # average Kullback-Leibler divergence
    return np.mean(np.sum(true * np.log(positive(true / positive(pred))), axis=1))


def cosine(true, pred):  # average Cosine coefficient
    return np.mean(np.sum(true * pred, axis=1) / positive(np.sum(true ** 2, axis=1) ** 0.5) / positive(
        np.sum(pred ** 2, axis=1) ** 0.5))


def intersection(true, pred):  # average intersection similarity
    return np.mean(np.sum(np.minimum(true, pred), axis=1))
