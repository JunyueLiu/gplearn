from typing import Union

import bottleneck as bn
import numpy as np


def var(x: np.ndarray, window) -> np.ndarray:
    return bn.move_var(x, window, axis=0)


def ternary_conditional_operator(cond: np.ndarray,
                                 ret1: Union[np.ndarray, float, int],
                                 ret2: Union[np.ndarray, float, int]) -> np.ndarray:
    return np.where(cond, ret1, ret2)


def log(x: np.ndarray):
    return np.log(x)


def sign(x: np.ndarray):
    return np.sign(x)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.add(x, y)


def subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.subtract(x, y)


def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.divide(x, y)


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.multiply(x, y)


def negative(x: np.ndarray) -> np.ndarray:
    return -x


def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def _max(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(x, y)


def _min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.minimum(x, y)


def relu(x: np.ndarray):
    return np.where(x < 0, 0, x)



