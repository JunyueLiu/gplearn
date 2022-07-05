from typing import Union

import bottleneck as bn
import numpy as np


def var(x: np.ndarray, window) -> np.ndarray:
    return bn.move_var(x, window, axis=0)


def ternary_conditional_operator(cond: np.ndarray,
                                 ret1: Union[np.ndarray, float, int],
                                 ret2: Union[np.ndarray, float, int]) -> np.ndarray:
    return np.where(cond, ret1, ret2)


def log(x):
    return np.log(x)


def sign(x):
    return np.sign(x)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.add(x, y)


def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.divide(x, y)


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.multiply(x, y)
