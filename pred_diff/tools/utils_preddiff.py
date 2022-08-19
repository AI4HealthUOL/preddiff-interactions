import numpy as np


#####################################################################################################
# AUX FUNCTIONS
#####################################################################################################
def laplace_smoothing(p, N, d, alpha=1):
    p_laplace = (p*N+alpha)/(N+d)
    return p_laplace


def log_laplace(p, N, d, alpha=1):
    return np.log2(laplace_smoothing(p, N, d, alpha=alpha))


def eval_mean_relevance(predictions, weights=None, log_transform=False, n_train_samples=10000, n_classes = 10):
    if weights is None:
        mean = predictions.mean(axis=0)
    else:
        if log_transform is True:
            weights = weights[:, :, np.newaxis]
        mean = np.sum(predictions*weights, axis=0)/np.sum(weights, axis=0)
    if log_transform:
        mean = log_laplace(mean, n_train_samples, n_classes)
    return mean


def eval_mean_interaction(predictions01, predictions0, predictions1, weights=None, log_transform=False,
                          n_train_samples=10000, n_classes=10):
    if weights is None:
        mean01 = predictions01.mean(axis=0)
        mean0 = predictions0.mean(axis=0)
        mean1 = predictions1.mean(axis=0)

    else:
        if log_transform is True:
            weights = weights[:, :, np.newaxis]
        mean01 = np.sum(predictions01*weights, axis=0)/np.sum(weights, axis=0)
        mean0 = np.sum(predictions0*weights, axis=0)/np.sum(weights, axis=0)
        mean1 = np.sum(predictions1*weights, axis=0)/np.sum(weights, axis=0)
    if log_transform is True:
        mean01 = log_laplace(mean01, n_train_samples, n_classes)
        mean0 = log_laplace(mean0, n_train_samples, n_classes)
        mean1 = log_laplace(mean1, n_train_samples, n_classes)
    return -(mean01 - mean0 - mean1)        # returns raw joint (interaction) effect


def eval_mean_shielded(predictions01, predictions0, weights=None, log_transform=False,
                       n_train_samples=10000, n_classes=10):
    if weights is None:
        mean01 = predictions01.mean(axis=0)
        mean0 = predictions0.mean(axis=0)

    else:
        if log_transform is True:
            weights = weights[:, :, np.newaxis]
        mean01 = np.sum(predictions01*weights, axis=0)/np.sum(weights, axis=0)
        mean0 = np.sum(predictions0*weights, axis=0)/np.sum(weights, axis=0)
        
    if log_transform is True:
        mean01 = log_laplace(mean01, n_train_samples, n_classes)
        mean0 = log_laplace(mean0, n_train_samples, n_classes)
        
    return -mean01 + mean0


class ManageCols:
    def __init__(self, columns=None, shape_mask=None):
        """
        This class is meant to be the interface between the special PredDiff columns interface and other data modalities
        such as two-dimensional image data.
        """
        self.columns = columns
        self.shape_mask = shape_mask

    def relevance_default_cols(self):
        """
        generates a list containing all features
        """
        impute_cols = [[x] for x in self.columns]
        return impute_cols

    def relevance_default_mask(self):
        list_mask = []
        for i in range(np.zeros(self.shape_mask).size):
            mask = np.zeros(self.shape_mask, dtype=np.bool).flatten()
            mask[i] = True
            list_mask.append(mask)
        return list_mask

    def interaction_default_cols(self):
        """
        generates a list of all feature combinations
        """
        interaction_cols = []
        for i, k1 in enumerate(self.columns):
            for k2 in self.columns[i + 1:]:
                interaction_cols.append([[k1], [k2]])
        return interaction_cols

    def interaction_default_mask(self):
        list_mask_relevance = self.relevance_default_mask()
        pass
        list_mask_interaction = []
        for i, mask1 in enumerate(list_mask_relevance):
            for mask2 in list_mask_relevance[i+1:]:
                list_mask_interaction.append([mask1, mask2])
        return list_mask_interaction