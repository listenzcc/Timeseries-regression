"""
File: generate_data.py
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
import numpy as np
import pandas as pd

from . import logger, root
from .options import total_length

# %% ---- 2024-05-13 ------------------------
# Function and class


def generate_random_series(n: int = total_length, generate_data_flag: bool = False):
    """
    Generate a random time series.

    Args:
        n (int): The length of the time series. Defaults to total_length.
        generate_data_flag (bool): Flag indicating whether to generate new data. Defaults to False.

    Returns:
        pd.DataFrame: The generated time series data.

    Raises:
        None

    Examples:
        >>> generate_random_series(n=100, generate_data_flag=True)
        pd.DataFrame with 100 rows and 3 columns: 'x', 'dvalue', 'value'
    """

    data = root.joinpath('res/data.csv')
    data.parent.mkdir(exist_ok=True)

    if not generate_data_flag and data.is_file():
        logger.debug(f'Found data: {data}')
        return pd.read_csv(data, index_col=0)

    dvalue = np.random.randn(n)
    value = np.array([np.sum(dvalue[:i]) for i in range(n+1)])
    df = pd.DataFrame()
    df['x'] = range(n)
    df['dvalue'] = dvalue
    df['value'] = value[1:]
    df.to_csv(data)
    logger.debug(f'Wrote history: {data}')
    return df

# %% ---- 2024-05-13 ------------------------
# Play ground


# %% ---- 2024-05-13 ------------------------
# Pending


# %% ---- 2024-05-13 ------------------------
# Pending
