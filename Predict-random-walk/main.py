"""
File: main.py
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
import contextlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics

from pathlib import Path
from rich import print
from loguru import logger

# --------------------
root = Path(__file__).parent
total_length = 1000
sample_length = total_length // 10
training_samples = total_length // 2

# %% ---- 2024-05-10 ------------------------
# Function and class


def generate_random_series(n: int = total_length):
    dvalue = np.random.randn(n)
    value = np.array([np.sum(dvalue[:i]) for i in range(n+1)])
    df = pd.DataFrame()
    df['x'] = range(n)
    df['dvalue'] = dvalue
    df['value'] = value[1:]
    return df


def fit(df: pd.DataFrame):
    # --------------------
    training_idx = [i for i in range(training_samples)]
    testing_idx = [i for i in range(
        training_samples, total_length-sample_length)]

    # --------------------
    x_training = np.stack([
        np.array(df.iloc[i:i+sample_length]['value']) for i in training_idx])

    y_training = np.stack([
        np.array(df.iloc[i+sample_length:i+sample_length+1]['value']) for i in training_idx]).squeeze()

    logger.debug(
        f'Prepared training data: {training_idx}, {(x_training.shape, y_training.shape)}')

    # --------------------
    x_testing = np.stack([
        np.array(df.iloc[i:i+sample_length]['value']) for i in testing_idx])

    y_testing = np.stack([
        np.array(df.iloc[i+sample_length:i+sample_length+1]['value']) for i in testing_idx]).squeeze()

    logger.debug(
        f'Prepared testing data: {testing_idx}, {(x_testing.shape, y_testing.shape)}')

    # --------------------
    regressor = linear_model.LinearRegression()
    regressor.fit(x_training, y_training)
    y_pred = regressor.predict(x_testing)
    mse = metrics.mean_squared_error(y_pred=y_pred, y_true=y_testing)
    logger.debug(f'MSE of {regressor} is {mse}')

    df['pred'] = None
    df.loc[[e+sample_length for e in testing_idx], 'pred'] = y_pred

    df['diff'] = df['value'] - df['pred']


@contextlib.contextmanager
def use_ax(ax, title: str = 'Title'):
    legend_position = 'upper left'
    bbox_to_anchor = (1, 1)
    try:
        yield ax
    finally:
        sns.move_legend(ax, legend_position, bbox_to_anchor=bbox_to_anchor)
        ax.set_title(title)


# %% ---- 2024-05-10 ------------------------
# Play ground
if __name__ == "__main__":
    logger.info('Start')
    df = generate_random_series()

    fit(df)
    print(df)

    # --------------------
    sns.set_style('darkgrid')
    palette = 'RdBu'  # 'Spectral'
    palette = sns.diverging_palette(
        0, 180, s=100, l=50, center='light', as_cmap=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    with use_ax(axes[0, 0], 'Random data') as ax:
        sns.scatterplot(
            df, ax=ax, x='x', y='value', hue='dvalue', palette=palette)

    with use_ax(axes[0, 1], 'Pred') as ax:
        sns.lineplot(
            df, ax=ax, x='x', y='value', color='#333', size=1, legend=False, zorder=1)
        sns.scatterplot(
            df, ax=ax, x='x', y='pred', hue="dvalue", palette=palette, zorder=2)

    with use_ax(axes[1, 0], 'Diff: Value - prediction') as ax:
        sns.scatterplot(
            df, ax=ax, x='x', y='diff', hue="diff", palette=palette)

    with use_ax(axes[1, 1], 'Diff vs dvalue') as ax:
        sns.scatterplot(
            df, ax=ax, x='dvalue', y='diff', hue="diff", palette=palette)

    fig.tight_layout()
    fig.savefig(root.joinpath('res.jpg'))
    plt.show()

    logger.info('Finished')


# %% ---- 2024-05-10 ------------------------
# Pending


# %% ---- 2024-05-10 ------------------------
# Pending
