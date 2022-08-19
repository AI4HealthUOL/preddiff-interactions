import numpy as np
import pandas as pd
import scipy
from typing import List
from tqdm.auto import tqdm

from .imputers import imputer_base
from .tools.utils_bootstrap import empirical_bootstrap
from .tools import utils_preddiff as ut_preddiff
from functools import partial

global pbar


#####################################################################################################
# Main Class
#####################################################################################################
class PredDiff(object):
    def __init__(self, model, train_data: np.ndarray, imputer: imputer_base.ImputerBase, n_imputations=100,
                 regression=True, classifier_fn_call="predict_proba", regressor_fn_call="predict",
                 unified_integral=False, fast_evaluation=False, n_group=1):
        """
        Arguments:
        model: trained regressor/classifier
        train_data: array with training data, only used to retrieve information
        imputer: used to generate samples to evaluate marginalizing integral
        n_impuations: number of imputation samples to be used
        regression: regression or classification setting (classification return relevance for each class)
        classifier_fn_call: member function to be called to evaluate classifier object model
                            (sklearn default: predict_proba)- has to return prediction as numpy array
        regressor_fn_call: member function to be called to evaluate regressor object model
                            (sklearn default: predict)- has to return predictions as numpy array
        unified_integral: key word which significantly improves statistical accuracy, recycles imputations when
                          calculating the interaction relevance
        fast_evaluation: default: False, useful to speed-up evaluation, ignores bootstrapping and returns zero-errorbars
        n_group: groups multiple imputations into a single inference call to accelerates computations. However, the
                 test_data is simply copied and hence, storage demand increases linearly with n_group
        """
        self.model = model
        self.train_data = train_data
        self.n_samples_train = len(self.train_data)
        self.fast_evaluation = fast_evaluation

        self.mangage_cols = ut_preddiff.ManageCols(shape_mask=self.train_data[0].shape)
        self.imputer = imputer

        # customize for regression and classification
        self.regression = regression
        self.fn_call = regressor_fn_call if regression else classifier_fn_call
        # apply log_transform on m-values for classification
        self.log_transform = self.regression is False

        self.model_predict = getattr(self.model, self.fn_call)
        self.n_classes = self.model_predict(train_data[:2]).shape[1] if self.regression is False else None

        self.n_imputations = n_imputations
        self.unified_integral = unified_integral    # recycle imputations for interaction to unify all m-value integrals
        self.n_group = n_group          # group multiple imputations together in single fn_call
        self.pbar = None

        # sanity check
        if self.n_group > self.n_imputations:
            self.n_group = self.n_imputations

        assert (self.n_imputations % self.n_group == 0), \
            f'Speed up factor {self.n_group} does not match {self.n_imputations}\n' \
            f'{self.n_imputations % self.n_group} not zero\n' \
            f'Maybe try {np.gcd(self.n_imputations, int(self.n_imputations/2))}'

    def relevances(self, data_test: np.ndarray, list_masks: List[np.ndarray] = None, n_bootstrap=100) \
            -> List[pd.DataFrame]:
        """
        computes relevances from a given PredDiff object
        params:
        data_test: dataset for which the relevances are supposed to be calculated
        list_masks: a list of mask, a mask is a boolean array and shape equal to a single sample
                    for all masks the relevance is computed for all samples in data_test
        n_bootstrap: number of bootstrap samples for confidence intervals, set zero for ignoring and speed up
        """
        # print('Calculate PredDiff relevances')

        if list_masks is None:
            list_masks = self.mangage_cols.relevance_default_mask()
        if not (isinstance(list_masks[0], list)) and not (isinstance(list_masks[0], np.ndarray)):
            list_masks = [list_masks]

        # get predictions
        predictions_true, predictions_raw, shape_imputations = self._setup_predict(data_test)

        # print("Evaluating m-values for every element in list_masks...")
        m_list = []
        self.pbar = tqdm(total=len(list_masks), desc='Relevance: ')

        for mask_impute in list_masks:
            # get imputed predictions
            # if mask_impute.sum() > 0:       # there is something to impute
            if True:
                predictions, weights = self._calc_predictions(data=data_test, shape_imputations=shape_imputations,
                                                              mask_impute=mask_impute)
            else:
                shape = (shape_imputations[0], *predictions_true.shape)
                predictions = np.copy(np.broadcast_to(predictions_true, shape))
                weights = np.ones(shape)

            # evaluate and calculate m-values
            self.pbar.set_postfix_str(f'Evaluate m-values')
            eval_mean = partial(
                ut_preddiff.eval_mean_relevance, log_transform=self.log_transform,
                n_train_samples=self.n_samples_train, n_classes=self.n_classes)

            mean, low, high, _ = empirical_bootstrap(
                input_tuple=(predictions, weights),
                score_fn=eval_mean, n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)


            mean = predictions_true - mean
            lowtmp = predictions_true - high    # swap low and high here
            high = predictions_true - low
            low = lowtmp

            m_df = pd.DataFrame([{"mean": m, "low": l, "high": h, "pred": p}
                                 for m, l, h, p in zip(mean, low, high, predictions_raw)])
            m_list.append(m_df)
            self.pbar.set_postfix_str(f'finished column')
            self.pbar.update(1)

        self.pbar.close()
        return m_list

    def interactions(self, data_test: np.ndarray, list_interaction_masks: List[List[np.ndarray]] = None,
                     n_bootstrap=100, individual_contributions=False) -> List[pd.DataFrame]:
        """
        computes interactions from a given PredDiff object
        params:
        data_test: dataset for which the interactions are supposed to be calculated
        list_masks: a list of [mask1, mask2], a mask is a boolean array and shape equal to a single sample
                    for all mask1<->mask2 the interaction is computed for all samples in data_test
        n_bootstrap: number of bootstrap samples for confidence intervals
        individual_contributions: also return individual components whose sum makes up the interaction relevance
        """
        print('Calculate PredDiff interactions.')
        if list_interaction_masks is None:
            list_interaction_masks = self.mangage_cols.interaction_default_mask()

        predictions_true, predictions_raw, shape_imputations = self._setup_predict(data_test)

        print("Evaluating m-values for every element in list_interaction_masks...")
        self.pbar = tqdm(total=len(list_interaction_masks), desc='Interaction: ')
        m_list = []

        assert len(list_interaction_masks[0]) == 2, 'list_interaction_masks has wrong format'
        for mask1, mask2 in list_interaction_masks:
            mask_impute = np.bitwise_or(mask1, mask2)
            assert mask1.sum()+mask2.sum() == mask_impute.sum(), 'both mask1 and mask2 need to be mutually exclusive'

            # select where to generate on-the-fly imputations
            if self.unified_integral is True:
                imputed_raw, weights = self.imputer.impute(test_data=data_test.copy(), mask_impute=mask_impute,
                                                  n_imputations=self.n_imputations)
            else:
                imputed_raw = None
                weights = None

            # get imputed predictions
            predictions01, weights = self._calc_predictions(data=data_test, shape_imputations=shape_imputations,
                                                            mask_impute=mask_impute, imputations_raw=imputed_raw,
                                                            weights=weights,
                                                            break_feature_dependence=True, mask_shuffle=mask2)

            predictions0, weights = self._calc_predictions(data=data_test, shape_imputations=shape_imputations,
                                                           mask_impute=mask_impute, mask_preddiff=mask1,
                                                           imputations_raw=imputed_raw, weights=weights)

            predictions1, weights = self._calc_predictions(data=data_test, shape_imputations=shape_imputations,
                                                           mask_impute=mask_impute, mask_preddiff=mask2,
                                                           imputations_raw=imputed_raw, weights=weights)

            # evaluate m-values
            self.pbar.set_postfix_str(f'evaluate m-values')
            input_tuple = (predictions01, predictions0, predictions1, weights)

            # rwa joint (interaction) effects
            eval_raw_interaction = partial(ut_preddiff.eval_mean_interaction,
                                       log_transform=self.log_transform, n_train_samples=self.n_samples_train,
                                       n_classes=self.n_classes)

            mean_raw_joined, low_joined, high_joined, _ = empirical_bootstrap(
                input_tuple=input_tuple, score_fn=eval_raw_interaction, n_iterations=n_bootstrap,
                fast_evaluation=self.fast_evaluation)

            mean_raw_joined -= predictions_true
            low_joined -= predictions_true
            high_joined -= predictions_true

            if individual_contributions is True:
                eval_mean = partial(ut_preddiff.eval_mean_relevance, weights=weights, log_transform=self.log_transform,
                                    n_train_samples=self.n_samples_train, n_classes=self.n_classes)

                mean01_overall, low01_overall, high01_overall, _ = empirical_bootstrap(input_tuple=predictions01, score_fn=eval_mean, n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)
                mean0_main, low0_main, high0_main, _ = empirical_bootstrap(input_tuple=predictions0, score_fn=eval_mean, n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)
                mean1_main, low1_main, high1_main, _ = empirical_bootstrap(input_tuple=predictions1, score_fn=eval_mean, n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)

                mean01_overall = predictions_true - mean01_overall
                low01_overall = predictions_true - high01_overall
                high01_overall = predictions_true - low01_overall

                mean0_main = predictions_true - mean0_main
                low0_main = predictions_true - high0_main
                high0_main = predictions_true - low0_main

                mean1_main = predictions_true - mean1_main
                low1_main = predictions_true - high1_main
                high1_main = predictions_true - low1_main

                # shielded main effects for 1/2
                eval_shielded = partial(ut_preddiff.eval_mean_shielded,
                                        log_transform=self.log_transform, n_train_samples=self.n_samples_train,
                                        n_classes=self.n_classes)

                mean0_shielded, low0_shielded, high0_shielded, _ = empirical_bootstrap(input_tuple=(predictions01, predictions1, weights), score_fn=eval_shielded,
                                                     n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)
                mean1_shielded, low1_shielded, high1_shielded, _ = empirical_bootstrap(input_tuple=(predictions01, predictions0, weights), score_fn=eval_shielded,
                                                     n_iterations=n_bootstrap, fast_evaluation=self.fast_evaluation)

                res = [{'mean': m, 'high': h, 'low': l, 'mean01': m01, 'high01': h01, 'low01': l01, 'mean0': m0,'high0': h0, 'low0': l0, 'mean1': m1,'high1': h1, 'low1': l1, 'mean0shielded': m0s, 'high0shielded': h0s, 'low0shielded': l0s, 'mean1shielded': m1s,'high1shielded': h1s, 'low1shielded': l1s}
                       for m, h, l, m01, h01, l01, m0, h0, l0, m1, h1, l1, m0s, h0s, l0s, m1s, h1s, l1s in zip(mean_raw_joined, high_joined, low_joined, mean01_overall, high01_overall, low01_overall, mean0_main, high0_main, low0_main, mean1_main, high1_main, low1_main, mean0_shielded, high0_shielded, low0_shielded, mean1_shielded, high1_shielded, low1_shielded)]
            else:
                res = [{'mean': m, 'high': h, 'low': l} for m, h, l in zip(mean_raw_joined, high_joined, low_joined)]

            m_df = pd.DataFrame(res)
            m_list.append(m_df)
            self.pbar.update(1)
        self.pbar.close()
        return m_list

    def _setup_predict(self, data: np.ndarray):
        n_samples = data.shape[0]

        predictions = self.model_predict(data)
        predictions_raw = predictions.copy()

        # prellocate memory for all imputations
        if self.regression is False:
            assert predictions.shape[1] == self.n_classes, 'something went wrong during prediction'
            shape_imputations = (self.n_imputations, n_samples, self.n_classes)
            predictions = ut_preddiff.log_laplace(predictions, self.n_samples_train, self.n_classes)
        else:
            shape_imputations = (self.n_imputations, n_samples)
        return predictions, predictions_raw, shape_imputations

    def _calc_predictions(self, data: np.ndarray, shape_imputations, mask_impute: np.ndarray,
                          mask_preddiff: np.ndarray = None, imputations_raw: np.ndarray = None,
                          weights: np.ndarray = None, break_feature_dependence=False, mask_shuffle: np.ndarray = None):
        # SETUP
        predictions = np.zeros(shape_imputations)

        n_samples = len(data)
        # imputation and preddiff mask may differ for interaction
        mask_preddiff = mask_impute if mask_preddiff is None else mask_preddiff

        # PREPARE IMPUTATIONS
        # generate new imputations for every integral
        if self.unified_integral is False or imputations_raw is None:
            self.pbar.set_postfix_str(f'generate imputations')
            imputations_raw, weights = self.imputer.impute(test_data=data.copy(), mask_impute=mask_impute,
                                                           n_imputations=self.n_imputations)

        # only for interaction
        # shuffle second feature to explicitly to enforce a factorizing imputer distribution
        if break_feature_dependence is True:
            shuffle_imputations = np.arange(shape_imputations[0])
            np.random.shuffle(shuffle_imputations)
            imputations_raw_new = imputations_raw.copy()
            # store shuffled feature 2 as independent imputations
            imputations_raw_new[:, :, mask_shuffle] = imputations_raw[shuffle_imputations][:, :, mask_shuffle]

        # extract relevant features
        imputations = imputations_raw[:, :, mask_preddiff]

        # PREDICT
        # prepare data for imputations
        test_data_imputed = np.array([data.copy() for _ in range(self.n_group)])

        # loop all batches and predict imputed samples
        for i in range(self.n_imputations)[::self.n_group]:
            self.pbar.set_postfix_str(f'imputations [{i}, {i+self.n_group}]')
            test_data_imputed[:, :, mask_preddiff] = imputations[i:i+self.n_group]

            self.pbar.set_postfix_str(f'inference [{i}, {i+self.n_group}]')
            shape = (self.n_group, n_samples) if self.n_classes is None else (self.n_group, n_samples, self.n_classes)
            predictions[i:i + self.n_group] = \
                self.model_predict(test_data_imputed.reshape((self.n_group*n_samples, *mask_preddiff.shape))).reshape(shape)

        return predictions, weights
