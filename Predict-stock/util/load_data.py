"""
File: load_data.py
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
import pandas as pd

from pathlib import Path

from . import logger, root


# %% ---- 2024-05-26 ------------------------
# Function and class

def load_data():
    df = pd.read_csv(root.joinpath('data/HistoricalData_1644543186424.csv'))
    df['Date'] = df['Date'].map(lambda x: pd.to_datetime(x))

    logger.info(f'Loaded data with columns: {df.columns}')

    for col in df.columns:
        if col == 'Date':
            continue
        df[col] = df[col].map(lambda x: pd.to_numeric(
            x.replace('$', '') if isinstance(x, str) else x))
        logger.debug(f'Convert from str to int: {col}')

    logger.info(f'Loaded data with n_rows={len(df)}')

    return df

# %% ---- 2024-05-26 ------------------------
# Play ground


# %% ---- 2024-05-26 ------------------------
# Pending


# %% ---- 2024-05-26 ------------------------
# Pending
