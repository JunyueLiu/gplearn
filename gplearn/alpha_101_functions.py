import os
import json
import datetime
import math
import pandas as pd
import numpy as np

"""
detail definition see https://arxiv.org/pdf/1601.00991.pdf

"""

import numba


@numba.njit
def shift(arr, num, fill_value=np.nan):
    """
    https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    :param arr:
    :param num:
    :param fill_value:
    :return:
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def returns(close: np.array, n: int = 1) -> np.array:
    return np.diff(close, n, prepend=np.nan, axis=0) / shift(close, n)


def log_returns(close: np.array, n: int = 1) -> np.array:
    """
    returns = daily close-to-close returns
    :param n:
    :param close:
    :return:
    """
    return np.diff(np.log(close), n, prepend=np.nan, axis=0)


if __name__ == '__main__':
    num_obs = 100
    num_stocks = 3
    close = np.random.random((num_obs, num_stocks))
    returns(close)
    log_returns(close)
