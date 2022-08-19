import numpy as np

from .imputer_base import ImputerBase

from typing import Optional


class TrainSetImputer(ImputerBase):
    """
    imputer just inserts randomly sampled training samples
    """

    def __init__(self, train_data: np.ndarray, **kwargs):
        super().__init__()
        self.imputer_name = 'TrainSet'

        self.train_data = train_data.copy()
        kwargs["label_encode"] = False
        kwargs["standard_scale"] = False
        self.seg = None

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        # returns only imputation for a single mask
        np_test = np.array(test_data)
        n_samples = np_test.shape[0]
        rng = np.random.default_rng()
        if self.seg is None:
            # rng.choice(self.train_data, n_samples, replace=True)  # replace: multiple occurrences are allowed
            res = rng.choice(self.train_data, n_samples * n_imputations, replace=True).copy()
            imputations = res.reshape(n_imputations, n_samples, *mask_impute.shape)
        else:
            # DEBUG
            import matplotlib.pyplot as plt

            assert n_samples == 1, 'breaking feature dependence using segements within a feature mask only works for a ' \
                                   'single data instance'
            assert self.seg.shape == mask_impute.shape, 'inconsistent shape of segmentation and imputation mask'
            unique_segment_labels = np.unique(self.seg[mask_impute])         # no consistency check for seg vs. mask
            n_segments_imputed = unique_segment_labels.size         # number of individual segments

            if n_segments_imputed == 0:     # no segments is marked, no imputations, nevertheless return random images
                res = rng.choice(self.train_data, n_samples * n_imputations, replace=True).copy()
                imputations = res.reshape(n_imputations, n_samples, *mask_impute.shape)
                return imputations, None

            random_images = rng.choice(self.train_data, n_imputations * n_segments_imputed, replace=True).copy()
            random_images = random_images.reshape(n_imputations, n_segments_imputed, *mask_impute.shape)
            imputations = np.zeros((n_imputations, 1, *mask_impute.shape), dtype=test_data.dtype)      # shape: n_imputation, n_samples, *mask
            # imputations = res[:, 3:4]
            for i, seg_label in enumerate(unique_segment_labels):
                mask_seg = self.seg == seg_label
                assert mask_seg.shape == mask_impute.shape, 'segment shape does not fit to feature/imputation shape'
                imputations[:, 0, mask_seg] = random_images[:, i, mask_seg]

                # imputations[:, 0, :, mask_seg==False] = test_data[0, :, mask_seg==False][:, np.newaxis].copy()

        return imputations, None


class GaussianNoiseImputer(ImputerBase):
    """
    Adds gaussian white noise onto the test samples.
    """
    def __init__(self, train_data: np.ndarray, sigma: Optional[np.ndarray] = None, clip=True):
        """sigma: array containing the deviation for every feature dimension"""
        super().__init__()
        self.imputer_name = 'GaussianNoise'
        self.train_data = train_data.copy()
        self.clip = clip
        if self.clip is True:
            self.clip_max = self.train_data.max()
            self.clip_min = self.train_data.min()
        else:
            assert False, 'not implemented yet'

        self.sigma = self.train_data[:300].std(axis=0) if sigma is None else sigma        # variance for gaussian noise
        assert np.alltrue(train_data[0].shape == self.sigma.shape), 'incorrect shape for variance sigma'

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        imputations = np.array([test_data.copy() for _ in range(n_imputations)])
        noise = np.random.randn(imputations.size).reshape(imputations.shape)
        imputations = np.array(imputations + self.sigma * noise, dtype=imputations.dtype)
        imputations = np.clip(imputations, self.clip_min, self.clip_max)
        return imputations, None
