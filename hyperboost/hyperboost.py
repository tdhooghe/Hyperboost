from enum import Enum

import numpy as np
from smac.epm.util_funcs import get_types
from smac.facade.smac_hpo_facade import SMAC4HPO, SMAC4AC
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT

from hyperboost.acquistion_function import ScorePlusDistance
from hyperboost.lgbm import LightGBM


class Hyperboost(SMAC4AC):
    def __init__(self, scenario: Scenario, rng: np.random.RandomState = None, **kwargs):
        # Initialize HyperBoost's empirical performance model
        model = LightGBM

        model_kwargs = dict()
        model_kwargs['min_child_samples'] = 1
        model_kwargs['alpha'] = 0.9
        model_kwargs['num_leaves'] = 8
        model_kwargs['min_data_in_bin'] = 1
        model_kwargs['n_jobs'] = 4
        model_kwargs['n_estimators'] = 100
        kwargs['model_kwargs'] = model_kwargs

        # Pass parameters to SMAC4HPO
        super().__init__(scenario=scenario, rng=rng, model=model,
                         acquisition_function=ScorePlusDistance, **kwargs)
