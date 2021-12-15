import numpy as np

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from sklearn import datasets
from sklearn.model_selection import train_test_split
from hyperboost.hyperboost import Hyperboost


iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_val, y_train, y_val = train_test_split(X, y)

def train_random_forest(config):
    """
    Trains a random forest on the given hyperparameters, defined by config, and returns the accuracy
    on the validation data.

    Input:
        config (Configuration): Configuration object derived from ConfigurationSpace.

    Return:
        cost (float): Performance measure on the validation data.
    """
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # define the evaluation metric as return
    return 1 - model.score(X_val, y_val)


if __name__ == "__main__":
    # Define your hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformIntegerHyperparameter("depth", 2, 100))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })
    
    hyperboost = Hyperboost(scenario=scenario, tae_runner=train_random_forest)
    best_found_config = hyperboost.optimize()
    # smac = SMAC4HPO(scenario=scenario, tae_runner=train_random_forest)
    # best_found_config = smac.optimize()
    print(best_found_config)