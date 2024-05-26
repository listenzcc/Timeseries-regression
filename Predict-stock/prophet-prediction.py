"""
File: prophet-prediction.py
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

from prophet import Prophet

from IPython.display import display

from util import logger, root
from util.load_data import load_data


# %% ---- 2024-05-26 ------------------------
# Function and class

def _after_load_data(df: pd.DataFrame):
    df = df[['Date', 'High']]
    df = df.loc[::-1]
    df.index = df.index[::-1]
    df.columns = ['ds', 'y']
    value = np.array(df['y'])
    dvalue = value[1:] - value[:-1]
    dvalue = np.concatenate([np.array([0.0]), dvalue])
    df.loc[:, 'dy'] = dvalue
    return df


def generate_hue_norm(df: pd.DataFrame):
    k = np.max(df['diff'].map(np.abs))
    return matplotlib.colors.Normalize(-k, k)


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


# %% ---- 2024-05-26 ------------------------
# Play ground
if __name__ == '__main__':
    # ----------------------------------------
    # ---- Parse input ----
    parser = argparse.ArgumentParser(
        'Prediction by prophet for the real stock')

    parser.add_argument('-s', '--show', help='If show the image',
                        dest='show_image_flag', action='store_true')

    parser.add_argument('-t', '--training-samples', help='Training with how many samples',
                        dest='training_samples', default=150)

    option = parser.parse_args()
    logger.info(f'Start with {option}')

    # ----------------------------------------
    # ---- Read raw data ----
    raw_df = load_data()
    parsed_df = _after_load_data(raw_df)
    logger.info(f'Data is loaded, {parsed_df}')

    # ----------------------------------------
    # ---- Training and predicting with prophet ----
    training_samples = option.training_samples

    training_df = parsed_df.iloc[:training_samples].copy()
    testing_df = parsed_df.iloc[training_samples:].copy()

    m = Prophet()
    m.fit(training_df)

    future = pd.DataFrame(testing_df['ds'])
    display(future)
    forecast = m.predict(future)
    forecast.index = testing_df.index
    display(forecast)
    testing_df['yhat'] = forecast['yhat']
    display(testing_df)

    # ----------------------------------------
    # ---- Convert the column names ----
    df = pd.concat([training_df, testing_df])
    df.columns = ['Date', 'value', 'dvalue', 'pred']
    df.loc[:, 'diff'] = df['value'] - df['pred']
    df.loc[:, 'name'] = 'Prophet'
    display(df)

    # ----------------------------------------
    # ---- Display and savefig ----
    sns.set_style('darkgrid')
    palette = 'RdBu'  # 'Spectral'
    palette = sns.diverging_palette(
        0, 180, s=80, l=50, center='light', as_cmap=True)
    hue_norm = generate_hue_norm(df)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    suptitle = 'Prophet prediction'
    fig.suptitle(suptitle)

    with use_ax(axs[0, 0], 'Stock data') as ax:
        sns.scatterplot(
            df, ax=ax, x='Date', y='value', hue='dvalue', palette=palette, zorder=1)
        sns.lineplot(
            df, ax=ax, x='Date', y='value', color='#0003', zorder=2)

    with use_ax(axs[0, 1], 'Pred') as ax:
        m.plot(forecast, ax=ax)
        sns.lineplot(
            df, ax=ax, x='Date', y='value', color='#333', size=1, legend=False, zorder=1)
        sns.scatterplot(
            df, ax=ax, x='Date', y='pred', hue="dvalue", palette=palette, zorder=2)
        sns.lineplot(
            df, ax=ax, x='Date', y='pred', color='#a00', size=1, legend=False, zorder=3)

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


# %% ---- 2024-05-26 ------------------------
# Pending
