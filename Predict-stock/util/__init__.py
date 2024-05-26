"""
File: __init__.py
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
from pathlib import Path
from loguru import logger


# %% ---- 2024-05-26 ------------------------
# Function and class
root = Path(__file__).parent.parent
root.joinpath('res').mkdir(exist_ok=True)
logger.add('log/timeseries-regression.log', rotation='5 MB')


# %% ---- 2024-05-26 ------------------------
# Play ground


# %% ---- 2024-05-26 ------------------------
# Pending


# %% ---- 2024-05-26 ------------------------
# Pending
