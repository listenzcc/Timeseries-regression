"""
File: regression.py
Author: Chuncheng Zhang
Date: 2024-05-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Explain the illusion that the random walk is predictable.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-05-10 ------------------------
# Requirements and constants
import argparse
import contextlib
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics

from rich import print
from util import logger, root
from util.options import RegressionOption
from util.generate_data import generate_random_series


# %% ---- 2024-05-10 ------------------------
# Function and class


def fit(df: pd.DataFrame,
        training_samples: int = RegressionOption.training_samples,
        total_length: int = RegressionOption.total_length,
        sample_length: int = RegressionOption.sample_length,
        predict_offset: int = 0
        ):
    """
    Fits a linear regression model to the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        training_samples (int): The number of training samples to use. Defaults to training_samples.
        total_length (int): The total length of the data. Defaults to total_length.
        sample_length (int): The length of each sample. Defaults to sample_length.

    Returns:
        None
    """

    # --------------------
    training_idx = list(range(training_samples))
    testing_idx = list(
        range(training_samples, total_length - sample_length - predict_offset))

    # --------------------
    x_training = np.stack([
        np.array(df.iloc[i:i+sample_length]['value']) for i in training_idx])

    y_training = np.stack([
        np.array(df.iloc[i+sample_length+predict_offset:i+sample_length+predict_offset+1]['value']) for i in training_idx]).squeeze()

    # --------------------
    x_testing = np.stack([
        np.array(df.iloc[i:i+sample_length]['value']) for i in testing_idx])

    y_testing = np.stack([
        np.array(df.iloc[i+sample_length:i+sample_length+1]['value']) for i in testing_idx]).squeeze()

    logger.debug(
        f'Segment data: training: {training_idx[0]}...{training_idx[-1]}, testing: {testing_idx[0]}...{testing_idx[-1]}')

    # --------------------
    regressor = linear_model.LinearRegression()
    regressor.fit(x_training, y_training)
    y_pred = regressor.predict(x_testing)
    mse = metrics.mean_squared_error(y_pred=y_pred, y_true=y_testing)
    logger.debug(f'MSE of {regressor} is {mse}')

    df['pred'] = None
    df.loc[[e+sample_length+predict_offset for e in testing_idx], 'pred'] = y_pred

    df['diff'] = df['value'] - df['pred']


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


# %% ---- 2024-05-10 ------------------------
# Play ground
if __name__ == "__main__":
    # --------------------
    parser = argparse.ArgumentParser('Native regression of random time series')
    parser.add_argument('-p', '--predict',
                        dest='predict_offset', default=0, type=int)
    parser.add_argument('-g', '--generate', help='If generate new data flag',
                        dest='generate_data_flag', action='store_true')
    parser.add_argument('-s', '--show', help='If show the image',
                        dest='show_image_flag', action='store_true')

    option = parser.parse_args()

    # --------------------
    logger.info(f'Start with {option}')

    df = generate_random_series(generate_data_flag=option.generate_data_flag)
    fit(df, predict_offset=option.predict_offset)
    print(df)

    k = np.max(df['diff'].map(np.abs))
    hue_norm = matplotlib.colors.Normalize(-k, k)

    # --------------------
    sns.set_style('darkgrid')
    palette = 'RdBu'  # 'Spectral'
    palette = sns.diverging_palette(
        0, 180, s=80, l=50, center='light', as_cmap=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    suptitle = f'Predict offset is {option.predict_offset}'
    fig.suptitle(suptitle)

    with use_ax(axs[0, 0], 'Random data') as ax:
        sns.scatterplot(
            df, ax=ax, x='x', y='value', hue='dvalue', palette=palette)

    with use_ax(axs[0, 1], 'Pred') as ax:
        sns.lineplot(
            df, ax=ax, x='x', y='value', color='#333', size=1, legend=False, zorder=1)
        sns.scatterplot(
            df, ax=ax, x='x', y='pred', hue="dvalue", palette=palette, zorder=2)
        sns.lineplot(
            df, ax=ax, x='x', y='pred', color='#a00', size=1, legend=False, zorder=3)
        ax.invert_yaxis()

    with use_ax(axs[1, 0], 'Diff: Value - prediction') as ax:
        sns.scatterplot(
            df, ax=ax, x='x', y='diff', hue="diff", palette=palette, hue_norm=hue_norm)

    with use_ax(axs[1, 1], 'Diff vs dvalue') as ax:
        sns.scatterplot(
            df, ax=ax, x='dvalue', y='diff', hue="diff", palette=palette, hue_norm=hue_norm)

    fig.tight_layout()
    fig.savefig(root.joinpath(f'res/{suptitle}.jpg'))

    if option.show_image_flag:
        plt.show()

    logger.info('Finished')


# %% ---- 2024-05-10 ------------------------
# Pending


# %% ---- 2024-05-10 ------------------------
# Pending
