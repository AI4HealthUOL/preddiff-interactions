import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from treeinterpreter import treeinterpreter as ti
import shap                 # conda install -c conda-forge shap
import copy

from ..datasets import utils
from .. import preddiff, shapley
from ..imputers import general_imputers, tabular_imputers, imputer_base


def add_text(text: str, ax: plt.Axes, loc: str):
    anchored_text = plt.matplotlib.offsetbox.AnchoredText(text,
                                                          loc=loc, borderpad=0.03)
    anchored_text.patch.set_boxstyle("round, pad=-0.2, rounding_size=0.1")
    anchored_text.patch.set_alpha(0.8)
    ax.add_artist(anchored_text)


def plot_performance(reg, x, y):
    plt.figure('performance')
    plt.plot(y, reg.predict(x), '.', label=reg.__module__[-10:])
    diag = np.linspace(y.min(), y.max(), 10)
    plt.plot(diag, diag)
    plt.legend()

alpha_errorbar = 0.7

def _scatter(c: pd.DataFrame, x_df: pd.DataFrame, method='', error_bars=None):
    # errorbars: List[[low, high]], list of high/low errorbars for all features in columns
    plt.figure('Scatter - ' + method)
    columns = np.ceil(np.sqrt(len(c.keys())))
    rows = np.ceil(len(c.keys())/columns)

    for n, key in enumerate(c.keys()):
        ax = plt.subplot(int(rows), int(columns), int(n+1))
        if error_bars is None:
            plt.scatter(x_df[key], c[key], marker='.', s=12)
        if error_bars is not None:
            plt.errorbar(x_df[key], c[key], yerr=error_bars[n], marker='', linestyle='', capsize=0, capthick=0.6,
                         elinewidth=1, alpha=alpha_errorbar)
            plt.scatter(x_df[key], c[key], marker='.', s=12)

        character = chr(97+n)      # ASCII character
        add_text(text=f"$x^{character}$", ax=ax, loc='lower right')
        # add_text(text=r"${\bar{m}}^{\,\,\mathrm{res}}$", ax=ax, loc='upper left')
        if method == 'SHAPTree' or method == 'custom shapley':
            add_text(text=r"${\phi}$", ax=ax, loc='upper left')
        else:
            add_text(text=r"${\bar{m}}$", ax=ax, loc='upper left')

        ax.grid(True)

        plt.tight_layout(pad=0.1)


def shap_interaction(shap_interaction_values: np.ndarray, x: np.ndarray, y: np.ndarray,
                     title='', axis_symmetric=False):
    shap_interaction_ab = shap_interaction_values[:, 1, 0] + shap_interaction_values[:, 0, 1]

    fig = plt.figure(f'SHAP interaction - {title}')
    vmax = 0.8 * np.abs(shap_interaction_ab).max()
    xmax = 1.05*np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05*xmax
    if axis_symmetric is True:
        xmin = -xmax

    ax = plt.subplot(2, 1, 2)
    ax.set_title('(c) shapley interaction index', fontsize=11)
    im = ax.scatter(x, y, c=shap_interaction_ab, cmap='coolwarm', vmax=vmax,
                    vmin=-vmax, alpha=0.9, marker='o', s=10)
    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text='$x^b$', ax=ax, loc='upper left')
    plt.axis([xmin, xmax, xmin, xmax])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel(r"joined effect", rotation='vertical')
    cbar.ax.set_xlabel(r"  $\phi_{a, b}$")
    cbar.ax.xaxis.set_label_position('top')

    ax = plt.subplot(2, 2, 1)
    ax.scatter(x, shap_interaction_values[:, 0, 0], marker='.', s=15)
    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text=r"$\phi_{a, a}$", ax=ax, loc='upper left')

    ax = plt.subplot(2, 2, 2)
    ax.scatter(y, shap_interaction_values[:, 1, 1], marker='.', s=15)
    add_text(text='$x^b$', ax=ax, loc='lower right')
    add_text(text=r"$\phi_{b, b}$", ax=ax, loc='upper left')

    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)
    # plt.savefig("shielded_effects.pdf")



def shielded_effects(explainer: preddiff.PredDiff, data_test: np.ndarray, x: np.ndarray, y: np.ndarray,
                     title='', axis_symmetric=False, error_bars=False):
    mask_interaction = [[np.array([True, False, False, False]), np.array([False, True, False, False])]]
    m_list = explainer.interactions(data_test=data_test, list_interaction_masks=mask_interaction, individual_contributions=True)
    m_values = m_list[0]

    shielded_joined_effect = - m_values['mean']
    shielded_main_x = m_values['mean0shielded']     # m_values['mean0'] - shielded_joined_effect
    shielded_main_y = m_values['mean1shielded']     # m_values['mean1'] - shielded_joined_effect

    fig = plt.figure(f'shielded effects - {title}')
    vmax = 0.8 * np.abs(shielded_joined_effect).max()
    xmax = 1.05*np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05*xmax
    if axis_symmetric is True:
        xmin = -xmax

    ax = plt.subplot(2, 1, 2)
    ax.set_title('(c) shielded joint effect', fontsize=11)
    im = ax.scatter(x, y, c=shielded_joined_effect, cmap='coolwarm', vmax=vmax,
                    vmin=-vmax, alpha=0.9, marker='o', s=10)
    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text='$x^b$', ax=ax, loc='upper left')
    plt.axis([xmin, xmax, xmin, xmax])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel(r"joined effect", rotation='vertical')
    cbar.ax.set_xlabel(r"  $\,\,{\bar{m}}^{f}_{\backslash \, ab}$")
    cbar.ax.xaxis.set_label_position('top')

    ax = plt.subplot(2, 2, 1)
    if error_bars is False:
        ax.scatter(x, shielded_main_x, marker='.', s=15)
    else:
        #error_shielded_x = np.sqrt(np.power(m_values['high'] - m_values['low'],2) + np.power(m_values['high0'] - m_values['low0'],2))
        error_shielded_x_high = m_values['high0shielded'] - shielded_main_x
        error_shielded_x_low = shielded_main_x - m_values['low0shielded']
        error_shielded_x = [error_shielded_x_low, error_shielded_x_high]

        ax.errorbar(x, shielded_main_x, error_shielded_x, marker='', linestyle='', capsize=0, capthick=0.6,
                    elinewidth=1, alpha=alpha_errorbar)
        ax.scatter(x, shielded_main_x, marker='.', s=10)
    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text=r"${\bar{m}}_{a \backslash \,b}$", ax=ax, loc='upper left')

    ax = plt.subplot(2, 2, 2)
    if error_bars is False:
        ax.scatter(y, shielded_main_y, marker='.', s=10)
    else:
        error_shielded_y_high = m_values['high1shielded'] - shielded_main_y
        error_shielded_y_low = shielded_main_y - m_values['low1shielded']
        error_shielded_y = [error_shielded_y_low, error_shielded_y_high]

        ax.errorbar(y, shielded_main_y, error_shielded_y, marker='', linestyle='', capsize=0, capthick=0.6,
                    elinewidth=1, alpha=alpha_errorbar)
        ax.scatter(y, shielded_main_y, marker='.', s=12)
    add_text(text='$x^b$', ax=ax, loc='lower right')
    add_text(text=r"${\bar{m}}_{b \backslash \, a}$", ax=ax, loc='upper left')

    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)
    # plt.savefig("shielded_effects.pdf")


def scatter_2d_heatmap(x: np.ndarray, y: np.ndarray, relevance: np.ndarray, title='', axis_symmetric=False):
    fig = plt.figure(f'heatmap - {title}')

    vmax = 0.8 * np.abs(relevance).max()
    xmax = 1.05*np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05*xmax
    if axis_symmetric is True:
        xmin = -xmax

    im = plt.scatter(x, y, c=relevance, cmap='coolwarm', vmax=vmax,
               vmin=-vmax, marker='o', alpha=0.9)
    ax = plt.gca()
    add_text(text='$x_a$', ax=ax, loc='lower right')
    add_text(text='$x_b$', ax=ax, loc='upper left')

    plt.axis([xmin, xmax, xmin, xmax])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"interaction  -  ${\bar{m}}^{\,\,\mathrm{int}}$", rotation='vertical')


    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)


def scatter_m_plots(explainer: preddiff.PredDiff, df_test: pd.DataFrame, n_imputations,
                    error_bars=False):
    # explainer = preddiff.PredDiff(model=reg, train_data=df_train.to_numpy(), imputer=imputer, fast_evaluation=True)
    m_list = explainer.relevances(data_test=df_test.to_numpy())

    values = np.array([m["mean"] for m in m_list])
    error_low = np.array([m["mean"] - m['low'] for m in m_list])
    error_high = np.array([m['high'] - m['mean'] for m in m_list])

    c = pd.DataFrame(values.T, columns=df_test.columns)
    error_bars = [[low, high] for low, high in zip(error_low, error_high)] if error_bars is True else None
    _scatter(c=c, x_df=df_test, method=f'PredDiff, n_imputations = {n_imputations}',
             error_bars=error_bars)
    # plt.savefig("scatter_m_plots.pdf")


def scatter_contributions(reg: RandomForestRegressor, x_df: pd.DataFrame, method='Sabaas',
                          x_train: pd.DataFrame = None):
    if method == 'Sabaas':
        prediction, bias, contributions = ti.predict(reg, x_df)
    elif method == 'TreeExplainer':
        explainer = shap.TreeExplainer(reg)
        contributions = explainer.shap_values(x_df)
    elif method == 'DeepExplainer':
        assert x_train is not None, 'please insert background data x_train'
        # select a set of background examples to take an expectation over
        mask = np.unique(np.random.randint(0, x_train.shape[0], size=100))
        background = x_train.iloc[mask]
        explainer = shap.DeepExplainer(reg, background)
        contributions = explainer.shap_values(x_df)
    elif method == 'KernelExplainer':
        assert x_train is not None, 'please insert background data x_train'
        # select a set of background examples to take an expectation over
        mask = np.unique(np.random.randint(0, x_train.shape[0], size=100))
        background = x_train.iloc[mask]
        ex = shap.KernelExplainer(reg.predict, background)
        contributions = ex.shap_values(x_df)
    else:
        assert 0 == 1, f'method not implemented: {method}'

    c = pd.DataFrame(contributions, columns=x_df.columns)

    _scatter(c=c, x_df=x_df, method=method)


def plot_n_dependence(x_test: pd.DataFrame, n_imputations, explainer: preddiff.PredDiff):
    n_plot = 100
    data = np.array(x_test.iloc[:n_plot])

    relevance_col = [['1']]
    mask_relevance = np.array([False, True, False, False], dtype=np.bool)
    m_list = explainer.relevances(data_test=data, list_masks=mask_relevance)

    m_values = m_list[0]
    values = np.squeeze(m_values['mean'])
    error_low = values - np.squeeze( m_values['low'])
    error_high = np.squeeze(m_values['high']) - values
    error = [error_low, error_high]

    figsize = plt.rcParams['figure.figsize'].copy()
    figsize[1] = 0.55*figsize[1]
    title = f'Ndepedence_{n_imputations}'
    fig = plt.figure(title, figsize=figsize)

    ax = fig.add_subplot(1, 2, 1)
    ax.errorbar(np.squeeze(data[:, mask_relevance]), np.squeeze(values), yerr=error, marker='.', linestyle='', capsize=1.5, capthick=0.5)
    add_text(text='$x_b$', ax=ax, loc='lower right')
    add_text(text=r'$\bar{m}$', ax=ax, loc='upper left')
    ax.set_yticks([-7, 0, 7])

    data = np.array(x_test.iloc[:2*n_plot])
    list_interaction_masks = [[np.array([True, False, False, False], dtype=np.bool),
                               np.array([False, True, False, False], dtype=np.bool)]]
    m_int = explainer.interactions(data_test=data.copy(), list_interaction_masks=list_interaction_masks)
    interaction = m_int[0]['mean']

    ax = fig.add_subplot(1, 2, 2)
    axis_symmetric = True
    x = data[:, list_interaction_masks[0][0]]
    y = data[:, list_interaction_masks[0][1]]
    vmax = 0.8 * np.abs(interaction).max()
    xmax = 1.05 * np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05 * xmax
    if axis_symmetric is True:
        xmin = -xmax
    im = ax.scatter(x, y, c=interaction, cmap='coolwarm', vmax=vmax,
                vmin=-vmax, marker='o', alpha=0.8)
    add_text(text='$x_a$', ax=ax, loc='lower right')
    add_text(text='$x_b$', ax=ax, loc='upper left')
    ax.axis([xmin, xmax, xmin, xmax])

    fig.tight_layout(pad=0.1)


def visualize_3d(interaction: np.ndarray, x_df: pd.DataFrame, interaction_cols, title=''):
    ic = interaction_cols[0]

    fig = plt.figure(f'3d-plot - {ic} - ' + title)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'{ic}')
    vmax = np.abs(interaction).max()
    ax.scatter(x_df[ic[0][0]], x_df[ic[1][0]], interaction, c=interaction, cmap='coolwarm', vmax=vmax,
               vmin=-vmax, marker='.')
    plt.xlabel(ic[0][0])
    plt.ylabel(ic[1][0])


def partial_dependence_plot(reg, data: pd.DataFrame):
    plt.figure('Partial Dependence Plot')
    columns = np.ceil(np.sqrt(len(data.keys())))
    rows = np.ceil(len(data.keys())/columns)

    predict = reg.predict(data)
    max, min = predict.max(), predict.min()
    for n, col_key in enumerate(data.keys()):
        feature = np.linspace(data[col_key].min(), data[col_key].max(), 30)
        mean, err = [], []
        for f in feature:
            arr = data.copy()
            arr[col_key] = f
            prediction = reg.predict(arr)
            m, e = utils.jackknife(np.array(prediction))
            mean.append(m)
            err.append(e)
        plt.subplot(int(rows), int(columns), n + 1)
        plt.ylim([np.min(mean)-1, np.max(mean)*1.1])
        plt.errorbar(feature, mean, err)
        sns.rugplot(data[col_key])
    plt.tight_layout()


def ice_plot(reg, data: pd.DataFrame):
    plt.figure('ICE Plot')
    n_evaluate = 20
    columns = np.ceil(np.sqrt(len(data.keys())))
    rows = np.ceil(len(data.keys())/columns)
    n_samples = len(data.index)
    predict = reg.predict(data)
    max, min = predict.max(), predict.min()
    d = max-min

    for n_key, col_key in enumerate(data.keys()):
        plt.subplot(rows, columns, n_key + 1)
        plt.ylim([-d/2, d/2])
        sns.rugplot(data[col_key])
        plt.xlabel(col_key[:15])

        feature = np.linspace(data[col_key].min(), data[col_key].max(), n_evaluate)
        for i in range(n_samples):
            sample = data.iloc[i].to_numpy()
            arr = np.repeat(sample[np.newaxis], n_evaluate, axis=0)
            arr[:, n_key] = feature

            prediction = reg.predict(arr)
            plt.plot(feature, prediction - prediction[0], color='gray', alpha=0.5, linewidth=1)
    plt.tight_layout()


def gp_feature_generator(data_train: pd.DataFrame, data_test: pd.DataFrame):
    plt.figure('GP feature generator')
    columns = np.ceil(np.sqrt(len(data_train.keys())))
    rows = np.ceil(len(data_train.keys())/columns)
    for n, col_key in enumerate(data_train.keys()):
        reg = GaussianProcessRegressor(1.0 * Matern() + 1.0 * WhiteKernel())
        col_train = data_train.drop(columns=col_key)
        col_test = data_test.drop(columns=col_key)

        # use labels
        # col_train = pd.concat([col_train, y_train], axis=1)
        # col_test = pd.concat([col_test, y_test], axis=1)

        reg.fit(col_train, data_train[col_key])
        print(f"key = {col_key}\n"
              f"train score = {reg.score(col_train, data_train[col_key]):.3F}\n"
              f"test score  = {reg.score(col_test, data_test[col_key]):.3F}\n")

        plt.subplot(int(rows), int(columns), n + 1)
        plt.plot(data_test[col_key], reg.predict(col_test), '.')
        plt.xlabel(col_key)
        plt.ylabel('predicted')
        plt.ylim(plt.xlim())
    plt.tight_layout()


def shapley_interaction_index(shapley_interaction: np.ndarray, shielded_shapley_x:np.ndarray,
                              shielded_shapley_y: np.ndarray, x: np.ndarray, y: np.ndarray,
                              title='', axis_symmetric=False, error_bars=False):


    fig = plt.figure(f'custom shapley interaction - {title}')
    vmax = 0.8 * np.abs(shapley_interaction).max()
    xmax = 1.05*np.abs([x, y]).max()
    xmin = np.array([x, y]).min() - 0.05*xmax
    if axis_symmetric is True:
        xmin = -xmax

    ax = plt.subplot(2, 1, 2)
    ax.set_title('(c) shapley interaction index', fontsize=11)
    im = ax.scatter(x, y, c=shapley_interaction, cmap='coolwarm', vmax=vmax,
                    vmin=-vmax, alpha=0.9, marker='o', s=10)
    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text='$x^b$', ax=ax, loc='upper left')
    plt.axis([xmin, xmax, xmin, xmax])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel(r"joined effect", rotation='vertical')
    cbar.ax.set_xlabel(r"  $\,\,{\bar{m}}^{f}_{\backslash \, ab}$")
    cbar.ax.xaxis.set_label_position('top')

    ax = plt.subplot(2, 2, 1)
    if error_bars is False:
        ax.scatter(x, shielded_shapley_x, marker='.', s=15)
    else:
        raise ValueError('errorbars are not implemented')

    add_text(text='$x^a$', ax=ax, loc='lower right')
    add_text(text=r"${\bar{m}}_{a \backslash \,b}$", ax=ax, loc='upper left')

    ax = plt.subplot(2, 2, 2)
    if error_bars is False:
        ax.scatter(y, shielded_shapley_y, marker='.', s=10)
    else:
        raise ValueError('errorbars are not implemented')

    add_text(text='$x^b$', ax=ax, loc='lower right')
    add_text(text=r"${\bar{m}}_{b \backslash \, a}$", ax=ax, loc='upper left')

    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)
    # plt.savefig("shielded_effects.pdf")
