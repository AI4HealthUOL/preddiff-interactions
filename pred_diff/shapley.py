import numpy as np
import pandas as pd
from typing import List, Tuple

import tqdm

from . import preddiff
from .imputers import imputer_base


class ShapleyExplainer(preddiff.PredDiff):
    def __init__(self, model, train_data: np.ndarray, imputer: imputer_base.ImputerBase, n_coalitions: int,
                 n_imputations=1, regression=True, classifier_fn_call="predict_proba", regressor_fn_call="predict",
                 n_group=1):
        """
        According to
        https://christophm.github.io/interpretable-ml-book/shapley.html
        Arguments:
        model: trained regressor/classifier
        train_data: array with training data, only used to retrieve information
        imputer: used to generate samples to evaluate marginalizing integral
        n_coalitions: number of coalitions formed when Shapley values are computed
        n_impuations: number of imputation samples to be used
        regression: regression or classification setting (classification return relevance for each class)
        classifier_fn_call: member function to be called to evaluate classifier object model
                            (sklearn default: predict_proba)- has to return prediction as numpy array
        regressor_fn_call: member function to be called to evaluate regressor object model
                            (sklearn default: predict)- has to return predictions as numpy array
        unified_integral: key word which significantly improves statistical accuracy, recycles imputations when
                          calculating the interaction relevance
        n_group: groups multiple imputations into a single inference call to accelerates computations. However, the
                 test_data is simply copied and hence, storage demand increases linearly with n_group
        """
        super().__init__(
            model=model, train_data=train_data, imputer=imputer, n_imputations=n_imputations,
            regression=regression, classifier_fn_call=classifier_fn_call, regressor_fn_call=regressor_fn_call,
            fast_evaluation=True, n_group=n_group)
        self.n_coalitions = n_coalitions

    def shapley_values(self, data_test: np.ndarray, list_masks: List[np.ndarray], base_feature_mask: np.ndarray) \
            -> List[pd.DataFrame]:
        """
        base_feature_mask: array of int with shape equal to a single data instance, e.g. [1, 1, 2, 3, 3]
        """
        assert base_feature_mask.dtype == np.int, 'base_feature_mask need to consist of integers'
        assert base_feature_mask.shape == list_masks[0].shape, 'incorrect shape of base_feature_mask'

        shapley_list = []
        for mask_i in tqdm.tqdm(list_masks, desc='shapley values'):
            list_masks_S, list_masks_S_i = self._create_random_coalitions(mask_i, base_feature_mask)
            marginalized_S_i = self.relevances(data_test=data_test, list_masks=list_masks_S_i)    # marginalized S u i
            marginalized_S = self.relevances(data_test=data_test, list_masks=list_masks_S)

            # calculate shapley values
            f_prediction_true = np.array(marginalized_S_i[0]['pred'])
            # calculate marginalized prediction f(Sbar)
            f_Sibar_u_empty = - np.array([m['mean'] for m in marginalized_S_i]) + f_prediction_true[np.newaxis]
            # prediction including target feature i
            f_Sibar_u_i = - np.array([m['mean'] for m in marginalized_S]) + f_prediction_true
            np_shapley = f_Sibar_u_i - f_Sibar_u_empty       # check shape when multiple data instances are passed
            mean = np_shapley.mean(axis=0)
            # std = np_shapley.std(axis=0)/np.sqrt(self.n_coalitions)
            pd_shapley = pd.DataFrame({'mean': mean, 'low': mean, 'high': mean, 'pred': f_prediction_true})
            shapley_list.append(pd_shapley)
        return shapley_list

    def shapley_interaction_index(self, data_test: np.ndarray, list_interaction_masks: List[List[np.ndarray]],
                                  base_feature_mask: np.ndarray) \
            -> List[pd.DataFrame]:
        """
        Calculates the Shapley Interaction Index based on subsampling random coalitions S.
        This correspond to our shielded effect up to a sign and factor of 0.5
        base_feature_mask: array of int with shape equal to a single data instance, e.g. [1, 1, 2, 3, 3]
        """
        assert base_feature_mask.dtype == np.int, 'base_feature_mask need to consist of integers'
        assert base_feature_mask.shape == list_interaction_masks[0][0].shape, 'incorrect shape of base_feature_mask'

        shapley_list = []
        for mask_i, mask_j in tqdm.tqdm(list_interaction_masks, desc='shapley interaction index'):
            # get mask for all terms in interaction measure
            # delta_ij = f(S u ij) - f(S u i) - f(S u j) + f(S)
            list_masks_S, list_masks_S_i, list_masks_S_j, list_masks_S_ij = \
                self._create_random_coalitions_interaction(mask_i, mask_j, base_feature_mask)

            # masked features are marginalized, this is opposite to the shapley notation m_S = f(Sbar)
            m_S = self.relevances(data_test=data_test, list_masks=list_masks_S)
            m_S_i = self.relevances(data_test=data_test, list_masks=list_masks_S_i)
            m_S_j = self.relevances(data_test=data_test, list_masks=list_masks_S_j)
            m_S_ij = self.relevances(data_test=data_test, list_masks=list_masks_S_ij)

            # get mean prediction given the different input feature sets
            f_prediction_true = np.array(m_S_i[0]['pred'])
            f_Sijbar_u_ij = - np.array([m['mean'] for m in m_S]) + f_prediction_true[np.newaxis]
            f_Sijbar_u_j = - np.array([m['mean'] for m in m_S_i]) + f_prediction_true[np.newaxis]
            f_Sijbar_u_i = - np.array([m['mean'] for m in m_S_j]) + f_prediction_true[np.newaxis]
            f_Sijbar_u_empty = - np.array([m['mean'] for m in m_S_ij]) + f_prediction_true[np.newaxis]

            # calculate interaction index
            # global sign since we use centered m-values mbar = f(X) - f(S)
            np_shapley_interaction_index = f_Sijbar_u_ij - f_Sijbar_u_i - f_Sijbar_u_j + f_Sijbar_u_empty  # check shape when multiple data instances are passed
            mean_interaction_index = 0.5 * np_shapley_interaction_index.mean(axis=0)
            # std = np_shapley.std(axis=0)/np.sqrt(self.n_coalitions)

            shapley_i = (f_Sijbar_u_i - f_Sijbar_u_empty).mean(axis=0)
            shapley_j = (f_Sijbar_u_j - f_Sijbar_u_empty).mean(axis=0)

            shapley_i = (f_Sijbar_u_ij - f_Sijbar_u_j).mean(axis=0)
            shapley_j = (f_Sijbar_u_ij - f_Sijbar_u_i).mean(axis=0)

            pd_shapley = pd.DataFrame({'mean_interaction': mean_interaction_index, 'low_interaction': mean_interaction_index, 'high_interaction': mean_interaction_index,
                                       'pred': f_prediction_true, 'mean': mean_interaction_index,
                                       'shapley_i': shapley_i, 'shapley_j': shapley_j})
            shapley_list.append(pd_shapley)
        return shapley_list

    def shap_interaction_matrix(self, data_test: np.ndarray, base_feature_mask: np.ndarray) -> np.ndarray:
        """returns the interaction matrix between all pairwise features, as proposed in TreeSHAP paper"""
        base_features = np.unique(base_feature_mask)
        n_features = len(base_features)
        n_samples = data_test.shape[0]

        list_masks = []
        for i in base_features:
            assert i == int(i), 'base_feature_mask does contain non-integer values'     # i int-valued
            # mask = np.zeros(base_feature_mask.shape, dtype=np.bool)
            # mask[i] = True
            mask = base_feature_mask == i
            list_masks.append(mask)
        shapley_list = self.shapley_values(data_test=data_test, list_masks=list_masks,
                                           base_feature_mask=base_feature_mask)
        shapley_values = np.array([m['mean'] for m in shapley_list])

        mask_interactions = self._create_interaction_mask(base_feature_mask=base_feature_mask)
        pd_shapley_interaction_list = self.shapley_interaction_index(data_test=data_test, list_interaction_masks=mask_interactions,
                                                                     base_feature_mask=base_feature_mask)
        interaction_matrix = np.zeros(shape=(n_samples, n_features, n_features))

        for index, (mask_i, mask_j) in enumerate(mask_interactions):
            i_feature = int(np.unique(base_feature_mask[mask_i]))       # convert mask into feature index
            j_feature = int(np.unique(base_feature_mask[mask_j]))  # convert mask into feature index

            pd_interaction = pd_shapley_interaction_list[index]
            shapley_interaction = np.array(pd_interaction['mean_interaction'])
            interaction_matrix[:, i_feature, j_feature] = shapley_interaction
            interaction_matrix[:, j_feature, i_feature] = shapley_interaction

            # interaction_matrix[:, i_feature, i_feature] -= shapley_interaction
            # interaction_matrix[:, j_feature, j_feature] -= shapley_interaction

        for i_reference in range(n_features):
            assert interaction_matrix[:, i_reference, i_reference].sum() == 0
            interaction_matrix[:, i_reference, i_reference] = shapley_values[i_reference] - \
                                                              interaction_matrix[:, i_reference].sum(axis=-1)

        return interaction_matrix

    def _create_random_coalitions(self, mask_i: np.ndarray, base_feature_mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        randomly shuffles features and creates corresponding total mask which including/excluding target mask.
        Returns a list with self.n_coalition repetitions/masks
        """
        # consistency check for mask
        base_features = np.unique(base_feature_mask)
        try:
            target_feature = int(np.unique(base_feature_mask[mask_i]))
            mask_temp = base_feature_mask == target_feature
            assert np.alltrue(mask_temp == mask_i), 'mask does not cover the complete base_feature'
        except TypeError:
            assert False, 'mask refers to more than one base_feature'

        list_masks_S = []       # contains masks refering to random coalitions
        list_masks_S_i = []        # random coalitions including target feature
        rng = np.random.default_rng()
        for i in range(self.n_coalitions):
            # create a random ordering of all features
            random_ordering = rng.choice(base_features, base_features.size, replace=False)
            index_target_feature = np.argmax(random_ordering == target_feature)

            # a random feature coalition (S)
            coalition = random_ordering[np.arange(random_ordering.size)>index_target_feature]
            mask_coalition = np.zeros(mask_i.shape, dtype=np.bool)
            for feature in coalition:
                assert np.alltrue(mask_coalition[base_feature_mask == feature] == False), 'double counting'
                mask_coalition[base_feature_mask == feature] = True

            list_masks_S.append(mask_coalition)
            list_masks_S_i.append(mask_coalition + mask_i)
        return list_masks_S, list_masks_S_i

    def _create_random_coalitions_interaction(self, mask_i: np.ndarray, mask_j: np.ndarray, base_feature_mask: np.ndarray) \
            -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        randomly shuffles features and creates corresponding total mask which including/excluding target mask.
        Returns a list with self.n_coalition repetitions/masks
        list_masks_S: only random coalition masked
        list_masks_S_i: first reference feature also masked
        list_masks_S_j: second reference feature additionally masked
        list_masks_S_ij: S plus both features masked
        """
        # consistency check for mask
        base_features = np.unique(base_feature_mask)
        try:
            target_feature_i = int(np.unique(base_feature_mask[mask_i]))
            mask_temp = base_feature_mask == target_feature_i
            assert np.alltrue(mask_temp == mask_i), 'mask does not cover the complete base_feature'

            target_feature_j = int(np.unique(base_feature_mask[mask_j]))
            mask_temp = base_feature_mask == target_feature_j
            assert np.alltrue(mask_temp == mask_j), 'mask does not cover the complete base_feature'
        except TypeError:
            assert False, 'mask refers to more than one base_feature'

        list_masks_S, list_masks_S_i, list_masks_S_j, list_masks_S_ij = [], [], [], []

        rng = np.random.default_rng()
        for i in range(self.n_coalitions):
            # create a random ordering of all features
            random_ordering = rng.choice(base_features, base_features.size, replace=False)
            index_target_feature = np.argmax(random_ordering == target_feature_i)

            # a random feature coalition (S)
            coalition_temp = random_ordering[np.arange(random_ordering.size) > index_target_feature]
            coalition = coalition_temp[coalition_temp != target_feature_j]

            mask_coalition = np.zeros(mask_i.shape, dtype=np.bool)
            for feature in coalition:
                assert np.alltrue(mask_coalition[base_feature_mask == feature] == False), 'double counting'
                assert feature != target_feature_i
                assert feature != target_feature_j
                mask_coalition[base_feature_mask == feature] = True
            list_masks_S.append(mask_coalition)
            list_masks_S_i.append(mask_coalition + mask_i)
            list_masks_S_j.append(mask_coalition + mask_j)
            list_masks_S_ij.append(mask_coalition + mask_i + mask_j)
        return list_masks_S, list_masks_S_i, list_masks_S_j, list_masks_S_ij

    def _create_interaction_mask(self, base_feature_mask: np.ndarray):
        """Creates interaction mask between all pairs of features [[mask1, mask2], [mask1, maks3], ...]"""
        # consistency check for mask
        base_features = np.unique(base_feature_mask)

        list_masks_interaction = []  # contains masks refering to random coalitions

        rng = np.random.default_rng()
        for i_reference_feature in range(len(base_features)):
            # create a random ordering of all features
            mask_i = base_feature_mask == i_reference_feature
            for j_target in range(i_reference_feature + 1, len(base_features)):
                mask_j = base_feature_mask == j_target
                list_masks_interaction.append([mask_i, mask_j])
        return list_masks_interaction