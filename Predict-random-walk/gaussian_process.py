"""
File: gaussian_process.py
Author: Chuncheng Zhang
Date: 2024-05-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-05-13 ------------------------
# Requirements and constants
import argparse
import matplotlib
import contextlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print

from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from util import logger, root
from util.options import GaussianProcessOption
from util.generate_data import generate_random_series

# %% ---- 2024-05-13 ------------------------
# Function and class


def fit(df: pd.DataFrame):
    total_length = len(df)
    training_samples = total_length // 10

    testing_index = list(df.index)

    # Only train on the first half data
    all_index = list(df.index[:int(total_length * 0.7)])
    np.random.shuffle(all_index)
    training_index = all_index[:training_samples]

    # --------------------
    x_training = df.iloc[training_index]['x'].to_numpy()[:, np.newaxis]
    y_training = df.iloc[training_index]['value'].to_numpy()[:, np.newaxis]

    # --------------------
    x_testing = df.iloc[testing_index]['x'].to_numpy()[:, np.newaxis]
    y_testing = df.iloc[testing_index]['value'].to_numpy()[:, np.newaxis]

    # --------------------
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(x_training, y_training)

    mean_prediction, std_prediction = gaussian_process.predict(
        x_testing, return_std=True)

    mse = metrics.mean_squared_error(y_pred=mean_prediction, y_true=y_testing)
    logger.debug(f'MSE of {gaussian_process} is {mse}')

    df['meanPred'] = None
    df['stdPred'] = None
    df['training'] = False
    df.loc[testing_index, 'meanPred'] = mean_prediction
    df.loc[testing_index, 'stdPred'] = std_prediction
    df.loc[training_index, 'training'] = True
    df['diff'] = df['value'] - df['meanPred']


@contextlib.contextmanager
def use_ax(ax, title: str = 'Title'):
    legend_position = 'upper left'
    bbox_to_anchor = (1, 1)
    try:
        yield ax
    finally:
        with contextlib.suppress(Exception):
            sns.move_legend(ax, legend_position, bbox_to_anchor=bbox_to_anchor)
        ax.set_title(title)


# %% ---- 2024-05-13 ------------------------
# Play ground
if __name__ == "__main__":
    # --------------------
    parser = argparse.ArgumentParser(
        'Gaussian process estimation for random time series')
    parser.add_argument('-g', '--generate', help='If generate new data flag',
                        dest='generate_data_flag', action='store_true')
    parser.add_argument('-s', '--show', help='If show the image',
                        dest='show_image_flag', action='store_true')

    option = parser.parse_args()

    # --------------------
    logger.info(f'Started with {option}')

    df = generate_random_series(generate_data_flag=option.generate_data_flag)
    fit(df)
    print(df)

    k = np.max(df['diff'].map(np.abs))
    hue_norm = matplotlib.colors.Normalize(-k, k)

    # --------------------
    sns.set_style('darkgrid')
    palette = 'RdBu'  # 'Spectral'
    palette = sns.diverging_palette(
        0, 180, s=80, l=50, center='light', as_cmap=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    with use_ax(axs[0, 0], 'Random data') as ax:
        sns.scatterplot(
            df, ax=ax, x='x', y='value', hue='dvalue', palette=palette, zorder=1)
        sns.lineplot(
            df, ax=ax, x='x', y='value', color='#0003', zorder=2)

    with use_ax(axs[0, 1], 'Pred') as ax:
        sns.lineplot(
            df, ax=ax, x='x', y='value', color='#333', size=1, legend=False, zorder=1)
        sns.scatterplot(
            df[df['training']], ax=ax, x='x', y='value', color='#03a', zorder=2)
        sns.lineplot(
            df, ax=ax, x='x', y='meanPred', color='#a00', size=1, legend=False, zorder=3)
        _df = df.copy()
        _df['upper'] = _df['meanPred'] + _df['stdPred']
        _df['lower'] = _df['meanPred'] - _df['stdPred']
        ax.fill_between(_df['x'].astype(np.float32),
                        _df['upper'].astype(np.float32),
                        _df['lower'].astype(np.float32),
                        alpha=0.3, color='#a00')
        # ax.invert_yaxis()

    with use_ax(axs[1, 0], 'Compare') as ax:
        sns.lineplot(df, ax=ax, x='x', y='meanPred', color='#a00')
        sns.lineplot(df, ax=ax, x='x', y='value', color='#03a')
        ax.legend(labels=['meanPred', None, 'value', None])

    with use_ax(axs[1, 1], 'Diff') as ax:
        sns.scatterplot(df, ax=ax, x='stdPred', y='diff',
                        hue='diff', palette=palette)

    fig.tight_layout()
    fig.savefig(root.joinpath('res/gaussian_process.jpg'))

    if option.show_image_flag:
        plt.show()

    logger.info('Finished')


# %% ---- 2024-05-13 ------------------------
# Pending


# %% ---- 2024-05-13 ------------------------
# Pending
