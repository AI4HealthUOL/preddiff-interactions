import numpy as np

from .imputer_base import ImputerBase


class ColorHistogramImputer(ImputerBase):
    """
    samples a uniform color according to the rgb distribution in the image which is estimated via b^3 histogram,
    see Explain Black-box Image Classifications Using Superpixel-based Interpretation
    https://ieeexplore.ieee.org/abstract/document/8546302
    """
    def __init__(self, train_data: np.ndarray, n_bins=8, **kwargs):
        super().__init__()
        self.imputer_name = 'Histogram'
        self.b = n_bins         # number of histogram bins per rbg channel
        self.colormin = train_data.min()
        self.colormax = train_data.max()
        self.splits = np.linspace(self.colormin, self.colormax, self.b)
        # self.hist_splits = np.broadcast_to(self.splits[:, np.newaxis], shape=(self.b, 3))

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        """
        test_data: (1, 3, n_pixel, n_pixel), with 3 rgb channels
        """
        mask_impute = np.array(mask_impute, dtype=np.bool)

        # returns only imputation for a single mask
        assert (1, 3) == test_data.shape[:2], f'incorrect shape, test_data.shape =  {test_data.shape}'
        image = np.squeeze(test_data).transpose(1, 2, 0)
        image_flatten = image.reshape(-1, 3)
        H, edges = np.histogramdd(image_flatten, (self.splits, self.splits, self.splits))
        hist = H.flatten()
        rng = np.random.default_rng()
        colors_index = rng.choice(np.arange(hist.size), size=n_imputations, replace=True, p=hist / hist.sum())
        index_red, index_green, index_blue = np.unravel_index(colors_index, H.shape)

        imputations = np.array([test_data.copy() for _ in range(n_imputations)])
        for i in range(n_imputations):
            mask_red = np.logical_and(self.splits[index_red[i] + 1] > image_flatten[:, 0],
                           image_flatten[:, 0] > self.splits[index_red[i]])
            mask_green = np.logical_and(self.splits[index_green[i] + 1] > image_flatten[:, 1],
                           image_flatten[:, 1] > self.splits[index_green[i]])
            mask_blue = np.logical_and(self.splits[index_blue[i] + 1] > image_flatten[:, 2],
                           image_flatten[:, 2] > self.splits[index_blue[i]])
            mask = np.array([mask_red, mask_green, mask_blue]).all(axis=0)
            color = image_flatten[mask].mean(axis=0)
            color_broadcasted = np.broadcast_to(color[:, np.newaxis, np.newaxis], mask_impute.shape)
            imputations[i, 0, mask_impute] = color_broadcasted[mask_impute]
        return imputations, None

