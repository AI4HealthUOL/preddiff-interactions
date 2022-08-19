import numpy as np
import pandas as pd
import scipy.stats as st


from typing import List, Union, Tuple, Any


from sklearn.experimental import enable_iterative_imputer  # noqa

from ..tools.utils_bootstrap import empirical_bootstrap

# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder


##########################################################################################
# EVALUATION
##########################################################################################
def _eval_mean(predictions, weights=None, log_transform=False, n_train_samples=10000, n_classes = 10):
    if(weights is None):
        mean = predictions.mean(axis=1)
    else:
        mean = np.sum(predictions*weights, axis=1)/np.sum(weights, axis=1)
    return mean


def evaluate_imputer(df, x, imputer_name=None, n_bootstrap=100, verbose=False):
    """
    evaluates imputers based on metrics from https://stefvanbuuren.name/fimd/sec-evaluation.html and some basic statistical tests
    Parameters:
    df: original dataframe
    x: result of applying imputer to df
    """
    res = []
    for xi in x:
        cols = [a for a in xi.columns if a not in ['id','imputation_id','sampling_prob']]
        for c in cols:
            imputations = np.stack(list(xi.groupby('id')[c].apply(lambda y: np.array(y))), axis=0)  # n_samples,n_imputations
            if("sampling_prob" in xi.columns):
                sampling_probs =np.stack(list(xi.groupby('id')["sampling_prob"].apply(lambda y: np.array(y))),axis=0)
            else:
                sampling_probs = None
            ground_truth = np.array(df[c])
            
            mean_pred, bounds_low, bounds_high, _ = empirical_bootstrap(imputations if sampling_probs is None else (imputations,sampling_probs), _eval_mean, n_iterations=n_bootstrap)
            
            raw_bias = np.abs(mean_pred- ground_truth)
            percentage_bias = np.mean(np.abs(raw_bias/(ground_truth+1e-8)))
            raw_bias = np.mean(raw_bias)
            if(verbose):
                print("\n\nimputed cols:",cols," var:",c,"dtype:",df[c].dtype)
                print("for reference: mean",np.mean(ground_truth),"std",np.std(ground_truth))
                print("mean raw bias:",np.mean(raw_bias))
                print("mean percentage bias:",np.mean(percentage_bias))
            
            #print(bounds_low[:3],bounds_high[:3],mean_pred[:3],ground_truth[:3])
            coverage_rate = np.mean(np.logical_and(bounds_low <= ground_truth, ground_truth <= bounds_high))
            
            average_width = np.mean(bounds_high-bounds_low)
            
            rmse = np.sqrt(np.mean(np.power(ground_truth-mean_pred, 2)))
            try:
                mwu = st.mannwhitneyu(ground_truth, mean_pred).pvalue   # low p value: reject H0 that both are sampled from the same distribution
            except:
                mwu = np.nan #all identical
            try:
                wc = st.wilcoxon(ground_truth,mean_pred).pvalue
            except:
                wc = np.nan     # Wilcoxon pvalue could not be calculated due to constant predictions
            
            if(verbose):
                print("coverage rate:",coverage_rate)
                print("average width:",average_width)
                print("rmse:",rmse)
                print("Mann Whitney U pvalue:", mwu)
                print("Wilcoxon pvalue:",wc)
            
            res.append({"imputed_cols":cols,"var":c,"dtype":df[c].dtype,"gt_mean":np.mean(ground_truth),"gt_std":np.std(ground_truth),"pred_mean":np.mean(mean_pred),"pred_std":np.std(mean_pred),"raw_bias":raw_bias,"percentage_bias":percentage_bias,"coverage_rate":coverage_rate,"average_width":average_width,"rmse":rmse,"mann-whitney_u_p":mwu,"wilcoxon_p":wc})
        
    df=pd.DataFrame(res)
    if(imputer_name is not None):
        df["imputer"]=imputer_name
    return df


#########################################################################################
# Imputer Baseclass
#########################################################################################
class ImputerBase(object):
    """
    Imputer Base class using boolean masks.
    """
    def __init__(self):
        self.imputer_name = ''

    def impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        generates imputations for data_test with respect to the mask_impute
        test_data: (n_samples, shape_mask),
        mask_impute: a boolean array defining which features are imputed, shape equal to a single instance
        n_imputations: number of imputations
        ---
        RETURN
         imputations: an array with all imputations, (n_imputations, n_samples, shape_mask)
         weights: (n_imputations, n_samples), probability associated with every imputations, one if unspecified
        """
        # print(f'Imputing dataset with n = {len(df_test)} samples and {n_imputations} imputations')
        assert mask_impute.shape == test_data.shape[1:], 'shapes of mask_impute and test_data do not match'
        test_data = test_data.copy()

        imputations, weights = self._impute(test_data=test_data, mask_impute=mask_impute, n_imputations=n_imputations)

        # to be replace if imputer provides weights,
        if weights is None:
            weights = np.ones(shape=imputations.shape[:2])

        assert mask_impute.shape == imputations.shape[2:], 'imputation changes data shape'

        return imputations, weights

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations: int) -> Tuple[np.ndarray, Any]:
        """
        internal imputation routine to be implemented by each specific imputer
        imputations: array
        weights: return None if not applicable for imputer else np.ndarray
        """
        pass


class ImputerBaseTabular(ImputerBase):
    def __init__(self, df_train: pd.DataFrame, **kwargs):
        super().__init__()
        self.train_data = df_train.to_numpy() if isinstance(df_train, np.ndarray) is False else df_train
        self.kwargs = kwargs
        self.df_train = df_train
        self.columns = df_train.columns

        self.exclude_cols = self.kwargs["exclude_cols"] if "exclude_cols" in self.kwargs.keys() else []
        self.nocat_cols = self.kwargs["nocat_cols"] if "nocat_cols" in self.kwargs.keys() else []
        # label-encode by default
        label_encode = self.kwargs["label_encode"] if "label_encode" in self.kwargs.keys() else True
        standard_scale = self.kwargs["standard_scale"] if "standard_scale" in self.kwargs.keys() else False
        standard_scale_all = self.kwargs["standard_scale_all"] if "standard_scale_all" in self.kwargs.keys() else False
        self.categorical_columns = [x for x in self.df_train.columns[np.where(np.logical_and(self.df_train.dtypes != np.float64,self.df_train.dtypes != np.float32))[0]] if not(x in self.exclude_cols)]
        self.numerical_columns = [x for x in self.df_train.columns[np.where(np.logical_or(self.df_train.dtypes == np.float64,self.df_train.dtypes == np.float32))[0]] if not(x in self.exclude_cols)]

        self.custom_postprocessing_fn = self.kwargs["custom_postprocessing_fn"] if "custom_postprocessing_fn" in kwargs.keys() else None

        self.oe = {}
        for c in self.categorical_columns if label_encode else []:
            oe = OrdinalEncoder()
            self.df_train[c] = oe.fit_transform(self.df_train[[c]].values)
            self.oe[c] = oe
        self.df_train_min = self.df_train.min()
        self.df_train_max = self.df_train.max()
        self.ss = {}
        for c in self.categorical_columns+self.numerical_columns if standard_scale_all else (self.numerical_columns if standard_scale else []):
            ss = RobustScaler()
            self.df_train[c] = ss.fit_transform(self.df_train[[c]].values)
            self.ss[c] = ss

        self.imputer = None

    def impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        generates imputations for data_test with respect to the mask_impute
        test_data: (n_samples, shape_mask),
        mask_impute: a boolean array defining which features are imputed
        n_imputations: number of imputations
        ---
        RETURN
         imputations: an array with all imputations, (n_imputations, n_samples, shape_mask)
         weights: (n_imputations, n_samples), probability associated with every imputations, one if unspecified
        """
        # print(f'Imputing dataset with n = {len(df_test)} samples and {n_imputations} imputations')
        quickfix_mnist = False
        if test_data.ndim > 2:
            quickfix_mnist = True
            test_data = test_data.reshape((test_data.shape[0], -1))
        df_test = pd.DataFrame(test_data, columns=self.columns)

        # assert mask_impute.ndim == 1, 'only 1-d mask allowed for tabular imputers'
        impute_cols = self.columns[mask_impute.reshape(-1)]         # convert to List[labels], old format

        df_test = self.preprocess(df=df_test.copy())

        list_df_imputations, weights = self._impute(df_test=df_test, impute_cols=impute_cols, n_imputations=n_imputations)

        if quickfix_mnist is False:
            list_df_imputations = self.postprocess(list_df=list_df_imputations)
            imputations = np.array([l.values for l in list_df_imputations])
        else:
            imputations = list_df_imputations

        # to be replace if imputer provides weights,
        if weights is None:
            weights = np.ones(shape=imputations.shape[:2])
        assert mask_impute.shape == imputations.shape[2:], 'imputation changes data shape'

        return imputations, weights

    def _impute(self, df_test: pd.DataFrame, impute_cols: List[Any], n_imputations: int) \
            -> Tuple[List[pd.DataFrame], Any]:
        """
        internal imputation routine to be implemented by each specific imputer
        imputations: array
        weights: return None if not applicable for imputer
        """
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        '''routine to be applied in derived classes '''
        for c in self.oe.keys():
            df[c] = self.oe[c].transform(df[[c]].values)
        for c in self.ss.keys():
            df[c] = self.ss[c].transform(df[[c]].values)
        return df

    def postprocess(self, list_df: List[pd.DataFrame]) -> List[pd.DataFrame]:
        for df in list_df:
            # round and truncate
            for c in self.ss.keys():
                if c in df.columns:
                    df[c] = self.ss[c].inverse_transform(df[[c]].values)

            for c in self.categorical_columns:
                if c in df.columns:
                    df[c] = df[c].astype(float).round(0).astype(int)
                    df[c] = df[c].clip(lower=self.df_train_min[c], upper=self.df_train_max[c])
            for c in self.oe.keys():
                if c in df.columns:
                    df[c] = self.oe[c].inverse_transform(df[[c]].values)

            # custom postprocessing
            # cols=[x for x in df.columns if x not in ["id","imputation_id","sampling_prob"]]
            # if(self.custom_postprocessing_fn is not None):
            #     df[cols] = self.custom_postprocessing_fn(df[cols])
        return list_df
