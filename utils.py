"""
Author: Dimas Ahmad
Description: This file contains utility functions for data processing and evaluation.
Source: Aside from the data preprocessing section, all is from the original implementation of the paper.
"""

import torch
import numpy as np


# Data Preprocessing
def squeeze_data(data, UNK=-99.99):
    T, N, M = data.shape
    lists_considered = []
    returns = data[:, :, 0]
    for i in range(N):
        returns_i = returns[:, i]
        if np.sum(returns_i != UNK) > 0:
            lists_considered.append(i)
    return data[:, lists_considered, :], lists_considered


def get_data(data_path, split_list, subset):
    subset2col = {
        'flow+fund_mom+sentiment': list(range(56, 60)) + [47],
        'fund_ex_mom_flow': [59] + [x for x in range(46, 58) if x not in (list(range(54, 58)) + [47])],
        'stock': range(46),
        'fund': range(46, 59),
        'fund+sentiment': range(46, 60),
        'stock+fund': range(59),
        'F_r12_2+sentiment': [58, 59],
        'stock+sentiment': [59] + list(range(0, 46)),
        'stock+fund+sentiment': range(60),
        'F_r12_2+flow+sentiment': [47, 58, 59]
    }

    dataset = np.load(data_path)
    data = dataset['data']
    column_considered = [0] + [x + 1 for x in subset2col[subset]]
    data = data[:, :, column_considered]
    data, list_considered = squeeze_data(data[split_list])
    return data, list_considered


def get_tensors(data, UNK=-99.99):
    ret = torch.tensor(data[:, :, 0])
    individualFeature = torch.tensor(data[:, :, 1:])
    macroFeature = torch.empty((data.shape[0], 0))
    mask = (ret != UNK)

    input_macro_tile = macroFeature.unsqueeze(1).repeat(1, ret.shape[1], 1)
    input_macro_masked = input_macro_tile[mask]
    input_masked = individualFeature[mask]
    input_concat = torch.concat([input_masked, input_macro_masked], dim=1)
    return_masked = ret[mask]

    return input_concat, return_masked, mask


def get_dataset(data_path, split_list, subset):
    datasets = []
    masks = []
    for split in split_list:
        data, list_considered = get_data(data_path, split, subset)
        input_concat, return_masked, mask = get_tensors(data)
        datasets.append(torch.utils.data.TensorDataset(input_concat, return_masked))
        masks.append(mask)
    return datasets, masks


def get_dataloader(datasets, batch_size, num_workers=4, shuffle=False):
    dataloaders = []
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                 shuffle=shuffle)
        dataloaders.append(dataloader)
    return dataloaders


def get_crossval_dataloaders(data_path, split_lists, subset, batch_size, num_workers=4, shuffle=False):
    crossval_loaders = []
    masks = []
    for split_list in split_lists:
        datasets, mask = get_dataset(data_path, split_list, subset)
        dataloaders = get_dataloader(datasets, batch_size, num_workers, shuffle)
        dict = {'datasets': datasets,
                'dataloaders': dataloaders}
        masks.append(mask)
        crossval_loaders.append(dict)
    return crossval_loaders, masks


def unload_data(dataloader):
    X_tensors = []
    y_tensors = []

    for i, (X, y) in enumerate(dataloader):
        X_tensors.append(X)
        y_tensors.append(y)

    return torch.cat(X_tensors, dim=0), torch.cat(y_tensors, dim=0)

# For visualization purposes
class FirmChar:
    def __init__(self):
        self._category = ['Fund mom', 'Fund char', 'Fund Family', 'Sentiment']
        self._category2variables = {
            'Fund mom': ['F_ST_Rev', 'F_r2_1', 'F_r12_2'],
            'Fund char': ['ages', 'flow', 'exp_ratio', 'tna', 'turnover'],
            'Fund Family': ['Family_TNA', 'fund_no', 'Family_r12_2', 'Family_flow', 'Family_age'],
            'Sentiment': ['sentiment', 'RecCFNAI', 'sentiment_lsq', 'sentiment_lad', 'CFNAI_orth', 'leading'],
        }
        self._variable2category = {}
        for category in self._category:
            for var in self._category2variables[category]:
                self._variable2category[var] = category
        self._category2color = {
            'Fund mom': 'blue',
            'Fund char': 'plum',
            'Fund Family': 'lime',
            'Sentiment': 'darkgreen'
        }
        self._color2category = {value: key for key, value in self._category2color.items()}

    def getColorLabelMap(self):
        return {var: self._category2color[self._variable2category[var]] for var in self._variable2category}


# Sharpe's evaluation and portfolio construction
def evaluate_sharpe(r_pred, r_masked, mask):
    # convert into numpy array
    if isinstance(r_pred, torch.Tensor):
        r_pred = r_pred.detach().cpu().numpy()
    if isinstance(r_masked, torch.Tensor):
        r_masked = r_masked.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    portfolio = construct_long_short_portfolio(r_pred, r_masked, mask, low=0.1, high=0.1)  # equally weighted
    return sharpe(portfolio)


def sharpe(r):
    return np.mean(r / r.std())


def construct_decile_portfolios(w, R, mask, value=None, decile=10):
    N_i = np.sum(mask.astype(int), axis=1)
    N_i_cumsum = np.cumsum(N_i)
    w_split = np.split(w, N_i_cumsum)[:-1]
    R_split = np.split(R, N_i_cumsum)[:-1]

    # value weighted
    value_weighted = False
    if value is not None:
        value_weighted = True
        value = value[mask]
        value_split = np.split(value, N_i_cumsum)[:-1]

    portfolio_returns = []

    for j in range(mask.shape[0]):
        R_j = R_split[j]
        w_j = w_split[j]
        if value_weighted:
            value_j = value_split[j]
            R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
        else:
            R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
        R_w_j_sorted = sorted(R_w_j, key=lambda t: t[1])
        n_decile = N_i[j] // decile
        R_decile = []
        for i in range(decile):
            R_decile_i = 0.0
            value_sum_i = 0.0
            for k in range(n_decile):
                R_decile_i += R_w_j_sorted[i * n_decile + k][0] * R_w_j_sorted[i * n_decile + k][2]
                value_sum_i += R_w_j_sorted[i * n_decile + k][2]
            R_decile.append(R_decile_i / value_sum_i)
        portfolio_returns.append(R_decile)
    return np.array(portfolio_returns)


def construct_long_short_portfolio(w, R, mask, value=None, low=0.1, high=0.1, normalize=True):
    # use masked R and value
    N_i = np.sum(mask.astype(int), axis=1)
    N_i_cumsum = np.cumsum(N_i)
    w_split = np.split(w, N_i_cumsum)[:-1]
    R_split = np.split(R, N_i_cumsum)[:-1]

    # value weighted
    value_weighted = False
    if value is not None:
        value_weighted = True
        value_split = np.split(value, N_i_cumsum)[:-1]

    portfolio_returns = []

    for j in range(mask.shape[0]):
        R_j = R_split[j]
        w_j = w_split[j]
        if value_weighted:
            value_j = value_split[j]
            R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
        else:
            R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
        R_w_j_sorted = sorted(R_w_j, key=lambda t: t[1])
        n_low = int(low * N_i[j])
        n_high = int(high * N_i[j])

        if n_high == 0.0:
            portfolio_return_high = 0.0
        else:
            portfolio_return_high = 0.0
            value_sum_high = 0.0
            for k in range(n_high):
                portfolio_return_high += R_w_j_sorted[-k - 1][0] * R_w_j_sorted[-k - 1][2]
                value_sum_high += R_w_j_sorted[-k - 1][2]
            if normalize:
                portfolio_return_high /= value_sum_high

        if n_low == 0:
            portfolio_return_low = 0.0
        else:
            portfolio_return_low = 0.0
            value_sum_low = 0.0
            for k in range(n_low):
                portfolio_return_low += R_w_j_sorted[k][0] * R_w_j_sorted[k][2]
                value_sum_low += R_w_j_sorted[k][2]
            if normalize:
                portfolio_return_low /= value_sum_low
        if np.isnan(portfolio_return_high) or np.isnan(portfolio_return_low) or np.isinf(
                portfolio_return_high) or np.isinf(portfolio_return_low):
            print(portfolio_return_high)
            print(portfolio_return_low)

        portfolio_returns.append(portfolio_return_high - portfolio_return_low)
    return np.array(portfolio_returns)
