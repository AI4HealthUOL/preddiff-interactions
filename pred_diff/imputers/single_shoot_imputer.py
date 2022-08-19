import cv2
import numpy as np
import cv2 as cv

from typing import Tuple, Any

from .imputer_base import ImputerBase


class OpenCVInpainting(ImputerBase):
    """
    This imputer is implemented within the popular SHAP framework.
    See https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html for more details
    """
    def __init__(self, inpainting_algorithm='telea'):
        """
        inpainting_algorithm: choices: ['telea', 'navier-stokes']
        """
        super(OpenCVInpainting, self).__init__()
        self.algorithm = inpainting_algorithm
        if self.algorithm == 'telea':
            self.inpaint_cv = cv2.INPAINT_TELEA
        elif self.algorithm == 'navier-stokes':
            self.inpaint_cv = cv2.INPAINT_NS
        else:
            assert False, f'Incorrect keyword inpainting_algorithm: {inpainting_algorithm}'

        self.imputer_name = f'cv2_{self.algorithm}'

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations: int) -> Tuple[np.ndarray, Any]:
        """Only for image data. test_data.shape = (3, dim, dim)"""
        n_samples = test_data.shape[0]
        assert n_imputations == 1, 'only single shoot imputations allowed for this imputer'

        imputations = np.zeros((n_imputations, n_samples, *mask_impute.shape))
        for i in range(n_samples):
            img = np.transpose(test_data[i], (1, 2, 0))
            # rescale image to int value [0, 255]
            assert img.min() >= 0, 'please insert images in original format'
            if img.max() <= 1:       # value between 0 and 1
                rescale_factor = 255
            else:
                assert img.max() <= 255, 'invalid image format'
                rescale_factor = 1

            img *= rescale_factor
            mask = mask_impute[i]
            dst = cv.inpaint(src=img.astype(np.uint8), inpaintMask=mask.astype(np.uint8), inpaintRadius=3,
                             flags=self.inpaint_cv)
            imputations[i, 0] = np.transpose(dst, (2, 0, 1))/rescale_factor

        # import matplotlib.pyplot as plt
        # plt.imshow(img.astype(np.uint8))
        # plt.figure()
        # plt.imshow(dst)

        return imputations, None


class MeanImputer(ImputerBase):
    """
    Imputes with the mean value for each feature, for images this is usually a gray patch
    """
    def __init__(self, train_data: np.ndarray):
        super().__init__()
        self.imputer_name = 'MeanImputer'
        self.train_data = train_data.copy()
        self.mean_feature = self.train_data[:1000].mean(axis=0)

        assert np.alltrue(train_data[0].shape == self.mean_feature.shape), 'sanity check'

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        n_samples = test_data.shape[0]
        assert n_imputations == 1, 'only single shoot imputations allowed for this imputer'

        # imputations = np.zeros((n_imputations, n_samples, *mask_impute.shape))

        imputations = np.array([test_data.copy() for _ in range(n_imputations)])
        imputations[0, :, mask_impute] = self.mean_feature[np.newaxis][:, mask_impute].T

        return imputations, None