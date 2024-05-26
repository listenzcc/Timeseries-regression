"""
File: prediction.py
Author: Chuncheng Zhang
Date: 2024-05-26
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


# %% ---- 2024-05-26 ------------------------
# Requirements and constants
import argparse
import matplotlib
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression

from IPython.display import display

from util import logger, root
from util.load_data import load_data


# %% ---- 2024-05-26 ------------------------
# Function and class
workers = [
    ('SVR_Linear', SVR(kernel='linear')),
    ('SVR_RBF', SVR(kernel='rbf')),
    ('BR', BayesianRidge()),
    ('Linear', LinearRegression()),
]


def fit_and_predict(
    df: pd.DataFrame,
    worker=workers[0],
    training_samples=150,
    total_length=252,
    sample_length=10,
    predict_offset=0
):
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
    regressor_name, regressor = worker
    regressor.fit(x_training, y_training)
    y_pred = regressor.predict(x_testing)
    mse = metrics.mean_squared_error(y_pred=y_pred, y_true=y_testing)
    logger.debug(f'MSE of {regressor} is {mse}')

    df.loc[:, 'pred'] = None
    df.loc[[e+sample_length+predict_offset for e in testing_idx], 'pred'] = y_pred

    df.loc[:, 'diff'] = df['value'] - df['pred']
    df.loc[:, 'name'] = regressor_name
    return df


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


def _after_load_data(df: pd.DataFrame):
    df = df[['Date', 'High']]
    df = df.loc[::-1]
    df.index = df.index[::-1]
    df.columns = ['Date', 'value']
    value = np.array(df['value'])
    dvalue = value[1:] - value[:-1]
    dvalue = np.concatenate([np.array([0.0]), dvalue])
    df.loc[:, 'dvalue'] = dvalue
    return df


def generate_hue_norm(df: pd.DataFrame):
    k = np.max(df['diff'].map(np.abs))
    return matplotlib.colors.Normalize(-k, k)


# %% ---- 2024-05-26 ------------------------
# Play ground

# sns.lineplot(df, x='Date', y='value')
# plt.show()

# %% ---- 2024-05-26 ------------------------
# Pending
if __name__ == "__main__":
    # --------------------
    parser = argparse.ArgumentParser('Prediction for the real stock')

    parser.add_argument('-p', '--predict',
                        dest='predict_offset', default=0, type=int)
    parser.add_argument('-s', '--show', help='If show the image',
                        dest='show_image_flag', action='store_true')

    option = parser.parse_args()
    logger.info(f'Start with {option}')

    # --------------------
    raw_df = load_data()
    parsed_df = _after_load_data(raw_df)
    logger.info(f'Data is loaded, {parsed_df}')

    # --------------------
    for worker in workers:
        df = fit_and_predict(
            parsed_df.copy(),
            worker=worker,
            predict_offset=option.predict_offset)
        display(df)

        # --------------------
        sns.set_style('darkgrid')
        palette = 'RdBu'  # 'Spectral'
        palette = sns.diverging_palette(
            0, 180, s=80, l=50, center='light', as_cmap=True)
        hue_norm = generate_hue_norm(df)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        suptitle = f'{worker[0]}, prediction offset is {option.predict_offset}'
        fig.suptitle(suptitle)

        with use_ax(axs[0, 0], 'Stock data') as ax:
            sns.scatterplot(
                df, ax=ax, x='Date', y='value', hue='dvalue', palette=palette, zorder=1)
            sns.lineplot(
                df, ax=ax, x='Date', y='value', color='#0003', zorder=2)

        with use_ax(axs[0, 1], 'Pred') as ax:
            sns.lineplot(
                df, ax=ax, x='Date', y='value', color='#333', size=1, legend=False, zorder=1)
            sns.scatterplot(
                df, ax=ax, x='Date', y='pred', hue="dvalue", palette=palette, zorder=2)
            sns.lineplot(
                df, ax=ax, x='Date', y='pred', color='#a00', size=1, legend=False, zorder=3)
            ax.invert_yaxis()

        with use_ax(axs[1, 0], 'Diff: Value - prediction') as ax:
            sns.scatterplot(
                df, ax=ax, x='Date', y='diff', hue="diff", palette=palette, hue_norm=hue_norm)

        with use_ax(axs[1, 1], 'Diff vs dvalue') as ax:
            sns.scatterplot(
                df, ax=ax, x='dvalue', y='diff', hue="diff", palette=palette, hue_norm=hue_norm)

        fig.tight_layout()
        fig.savefig(root.joinpath(f'res/{suptitle}.jpg'))

        if option.show_image_flag:
            plt.show()

    logger.info('Finished')

# %% ---- 2024-05-26 ------------------------
# Pending
