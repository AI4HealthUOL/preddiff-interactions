__all__ = ['empirical_bootstrap']

import numpy as np
from sklearn.utils import resample
from multiprocessing import Pool
from functools import partial


def _eval(ids, input_tuple, score_fn):
    return score_fn(*[t[ids] for t in input_tuple])


def empirical_bootstrap(input_tuple, score_fn, ids=None, n_iterations=100, alpha=0.95, threads=None,
                        fast_evaluation=False):
    # evaluate prediction scores
    if isinstance(input_tuple, tuple) is False:
        input_tuple = (input_tuple,)
    score_point = score_fn(*input_tuple)

    if fast_evaluation is False:
        # calculate upper/lower bounds
        if ids is None:
            ids = []
            for _ in range(n_iterations):
                ids.append(resample(range(len(input_tuple[0])), n_samples=len(input_tuple[0])))
            ids = np.array(ids)

        pool = Pool(threads)
        fn = partial(_eval, input_tuple=input_tuple, score_fn=score_fn)
        results = pool.map(fn, ids)
        pool.close()
        pool.join()
        
        score_diff = np.array(results) - score_point

        score_low = score_point + np.percentile(score_diff, ((1.0-alpha)/2.0) * 100, axis=0)
        score_high = score_point + np.percentile(score_diff, (alpha+((1.0-alpha)/2.0)) * 100, axis=0)
    else:
        # set bounds to score
        score_low = score_point.copy()
        score_high = score_point.copy()
        ids = []

    return score_point, score_low, score_high, ids

