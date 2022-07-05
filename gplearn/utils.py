"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers

import numpy as np
from joblib import cpu_count
import numba
import talib

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

@numba.njit
def nans(arr, dtype=np.float64) -> np.ndarray:
    a = np.empty_like(arr, dtype)
    a.fill(np.nan)
    return a


@numba.njit
def shift(arr, num, fill_value=np.nan):
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


def talib_corr(one_d_array_x: np.ndarray, one_d_array_y: np.ndarray, d: int):
    # try:
    #     return talib.CORREL(one_d_array_x, one_d_array_y, d)
    # except Exception as e:
    #     if 'inputs are all NaN' in e.args[0]:
    #         return nans(one_d_array_x)
    #     raise Exception(e)

    # new approach to use mask to deal with nan
    nan_mask = np.ones(one_d_array_x.shape)
    for j in range(d, len(one_d_array_x)):
        if np.isnan(one_d_array_x[j - d:j]).any() or np.isnan(one_d_array_y[j - d:j]).any():
            nan_mask[j] = np.nan

    one_d_array_x = np.nan_to_num(one_d_array_x)
    one_d_array_y = np.nan_to_num(one_d_array_y)
    corr = talib.CORREL(one_d_array_x, one_d_array_y, d)
    corr = corr * nan_mask
    return corr