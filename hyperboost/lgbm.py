import typing

import numpy as np
from lightgbm import LGBMRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from smac.configspace import ConfigurationSpace

from smac.epm.base_epm import AbstractEPM


class LightGBM(AbstractEPM):
    """Implementation of the Hyperboost EPM

    **Note:** The input dimensionality of Y for training and the output dimensions
    of all predictions (also called ``n_objectives``) depends on the concrete
    implementation of this abstract class.

    Attributes
    ----------
    instance_features : np.ndarray(I, K)
        Contains the K dimensional instance features
        of the I different instances
    types : list
        If set, contains a list with feature types (cat,const) of input vector
    """

    def __init__(
            self,
            configspace: ConfigurationSpace,
            types: np.ndarray,
            bounds: typing.List[typing.Tuple[float, float]],
            seed: int,
            min_child_samples: int = 1,
            num_leaves: int = 1,
            alpha: float = 0.9,
            min_data_in_bin: int = 1,
            n_jobs: int = -1,
            n_estimators: int = 100,
            instance_features: typing.Optional[np.ndarray] = None,
            pca_components: typing.Optional[int] = None,
    ) -> None:

        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components
        )

        self.lgbm = None  # type: LGBMRegressor
        self.min_child_samples = min_child_samples
        self.alpha = alpha
        self.num_leaves = num_leaves
        self.min_data_in_bin = min_data_in_bin
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.seed = seed

        self.kdtree = None  # A KDTree to be constructed for measuring distance

        self.types = np.asarray(types)

        self.inc = None  # The incumbent value
        self.selection = np.asarray(types) != 0  # Selection of hyperparameters that require one-hot-encoding
        self.contains_nominal = any(self.selection)  # Flag that checks if there are any nominal parameters
        self.categories = self.types[self.selection]  # Number of possible categories per nominal parameter

        self.max_distance = sum(np.maximum(i, 1) for i in types) ** 2  # Maximum L1 distance of two points in
        # hyperparameter space

        # self.pca_components_ = pca_components_
        # if pca_components_ is not None and pca_components_ > 0:
        #     self.pca_ = PCA(n_components=pca_components_)
        # else:
        #     self.pca_ = None

    def _train(self, X: np.ndarray, y: np.ndarray) -> 'LightGBM':
        # X_ = X
        # y_ = y
        # print(f'Shape X {X_.shape} and shape y {y_.shape}')
        # print(X_)

        self.X = X
        self.y = y.flatten()
        self.X_transformed = self.transform(X)
        self.inc = np.max(self.y)
        n_samples = self.X.shape[0]

        self.lgbm = LGBMRegressor(verbose=-1, min_child_samples=self.min_child_samples, objective="quantile",
                                  num_leaves=self.num_leaves, alpha=self.alpha, min_data_in_bin=self.min_data_in_bin,
                                  n_jobs=self.n_jobs, n_estimators=self.n_estimators, random_state=self.seed)

        self.lgbm.fit(self.X, self.y)
        # print(f'Flattened y is {y.flatten()} and shape is {self.y.shape}')
        # if self.pca_ is not None and self.X_transformed.shape[1] > self.pca_components_:
        #     self.X_transformed = self.pca_.fit_transform(self.X_transformed)

        self.kdtree = cKDTree(self.X_transformed)

    def _predict(self, X: np.ndarray,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov') \
            -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:

        loss = self.lgbm.predict(X)
        # print(f'Loss is {loss}')
        X_transformed = self.transform(X)

        # if self.pca_ is not None and X_transformed.shape[1] > self.pca_components_ and \
        #         self.X_transformed.shape[0] >= 2:
        #     X_transformed = self.pca_.transform(X_transformed)

        dist, ind = self.kdtree.query(X_transformed, k=1, p=2, workers=-1)
        # print(f'Distance is {dist}, ind is {ind}')
        # print(f'Reshaped distance is {dist.reshape(-1)}')

        scale = np.std(self.y)
        # print(f'Scale is {scale}')
        # print("var_y:", np.var(self.y), "var_x:", np.var(self.X))
        unscaled_dist = dist.reshape(-1) / self.max_distance
        # loss[unscaled_dist == 0] = 1
        dist = unscaled_dist * scale
        closeness = 1 - dist
        # print(f'closeness = {closeness}')
        return loss, closeness

    def transform(self, X):
        if not self.contains_nominal:
            return X

        result = []
        for i in X:
            # Split
            nominal = i[self.selection].astype(int)
            numerical = i[~self.selection]

            # Concatenate one-hot encoded together with numerical
            r = np.concatenate(
                [self.one_hot_vector(self.categories[index], indicator) for index, indicator in enumerate(nominal)])
            r = np.concatenate([numerical, r])

            result.append(r)

        return np.array(result)

    @staticmethod
    def one_hot_vector(length, indicator):
        result = np.zeros(length)
        result[indicator] = 1
        return result
