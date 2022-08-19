import pandas as pd
import numpy as np
import scipy
import os
from sklearn.datasets import load_boston
from typing import Tuple, Optional, Callable

# import utils
from pred_diff.datasets import utils

base_dir = os.path.dirname(__file__)


class SyntheticDataset:
    def __init__(self, function: Callable[[np.ndarray], np.ndarray], mean: Optional[np.ndarray] = None,
                 cov: Optional[np.ndarray] = None, noise: float = 0.):
        self.function = function
        self.noise = noise

        self.mean = np.array([0, 0, 0, 0]) if mean is None else np.array(mean)
        self.cov = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]) if cov is None else np.array(cov)
        assert np.alltrue(np.linalg.eigvals(self.cov) > 0), f'covariance matrix not valid: \n{self.cov}'
        # assert self.cov.shape == (4, 4)
        self.mvn = scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov)     # set-up multivariate normal

    def load_pd(self, n_samples: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        n_samples = 4000 if n_samples is None else n_samples
        x = self.mvn.rvs(n_samples)      # draw random samples
        y = self.function(x)
        # x += self.noise * np.random.randn(x.size).reshape(x.shape)
        return pd.DataFrame(x, columns=['0', '1', '2', '3']), pd.Series(y)


class SyntheticDynamaskData:
    def __init__(self, function: Callable[[np.ndarray], np.ndarray], mean: Optional[np.ndarray] = None,
                 cov: Optional[np.ndarray] = None):
        self.function = function

        self.mean = np.array([0, 0, 0, 0]) if mean is None else np.array(mean)
        self.cov = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]) if cov is None else np.array(cov)
        assert np.alltrue(np.linalg.eigvals(self.cov) > 0), f'covariance matrix not valid: \n{self.cov}'
        # assert self.cov.shape == (4, 4)
        self.mvn = scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov)     # set-up multivariate normal

    def load_pd(self, n_samples: int = None) -> [np.ndarray, np.ndarray]:
        n_samples = 4000 if n_samples is None else n_samples
        # x = self.mvn.rvs(n_samples)      # draw random samples
        rng = np.random.default_rng(seed=1)
        x = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=n_samples)
        y = self.function(x)
        # x += self.noise * np.random.randn(x.size).reshape(x.shape)
        return x, y


class BostonHousing:
    def load_pd(self) -> Tuple[pd.DataFrame, pd.Series]:
        boston = load_boston()
        boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
        y_pd = pd.Series(boston.target, name='target')
        return boston_pd, y_pd


class ConcreteCompression:
    def load_pd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        folder_name = 'concrete_data'
        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls", ]
        utils.download_data(urls, folder_name)

        path = f'{base_dir}/Data/{folder_name}/Concrete_Data.xls'
        data = pd.read_excel(path)
        assert pd.isna(data).sum().sum() == 0, "NAN values found in data"
        return data.iloc[:, :-1], data.iloc[:, -1]


class EnergyEfficiency:
    def load_pd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        folder_name = 'energy_efficiency'
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx", ]
        utils.download_data(urls, folder_name)

        path = f'{base_dir}/Data/{folder_name}/ENB2012_data.xlsx'
        data = pd.read_excel(path)
        assert pd.isna(data).sum().sum() == 0, "NAN values found in data"
        return data.iloc[:, :-2], data.iloc[:, -2:]


class WineQuality:
    def load_pd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        folder_name = 'winequality'
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/windequality-white.csv"]
        utils.download_data(urls, folder_name)

        path = f'{base_dir}/Data/{folder_name}/winequality-red.csv'
        data = pd.read_csv(path, sep=';')
        assert pd.isna(data).sum().sum() == 0, "NAN values found in data"
        return data.iloc[:, :-1], data.iloc[:, -1]


class BikeRental:
    def load_pd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        folder_name = 'bikerental'
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",]
        utils.download_data(urls, folder_name)
        file_path = f'{base_dir}/Data/{folder_name}/hour.csv'
        if os.path.exists(file_path) is False:
            from zipfile import ZipFile
            with ZipFile(f'{base_dir}/Data/{folder_name}/Bike-Sharing-Dataset.zip', 'r') as zip:
                print('unzip file')
                zip.extractall(f'{base_dir}/Data/{folder_name}')
        data = pd.read_csv(file_path, sep=',')
        data['dteday'] = data['dteday'].apply(lambda s: float(s[-2:]))
        assert pd.isna(data).sum().sum() == 0, "NAN values found in data"
        return data.iloc[:, :-3], data.iloc[:, -3]