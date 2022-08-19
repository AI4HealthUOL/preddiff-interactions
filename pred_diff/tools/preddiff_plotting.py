import numpy as np
import pandas as pd
from scipy.stats import iqr
import matplotlib.pyplot as plt

import shap


#################################################################################################
# SHAP CONVERSION FOR PLOTTING
def preddiff_list_to_shap_explanation(m_list, df, classifier_component=1):
    values = np.array([m["mean"] for m in m_list])

    if len(values.shape) > 2:   # classification
        values = values[:, :, classifier_component]
    values = values.T.astype(np.float64)
    base_values = 0*values
    data = np.array(df)
    feature_names = df.columns
    return shap.Explanation(values=values, base_values=base_values, data=data, display_data=data, feature_names=feature_names)


def preddiff_list_to_m_shap_classification(m_list):
    temp_means = [temp['mean'] for temp in m_list]
    c0 = [np.array([c[0] for c in feature]) for feature in temp_means]
    c1 = [np.array([c[1] for c in feature]) for feature in temp_means]
    m_values = [np.array(c0).T, np.array(c1).T]
    return m_values

#################################################################################################
# AUX FUNCTIONS FOR SUMMARIZATION
#################################################################################################
def plot_global_preddiff_stats(m_stats, col="meanabs", min_value=0, max_entries=5, title=None, filename=None):
    '''plots result of calculate_global_preddiff_stats as barplot'''
    m_stats_local = m_stats.sort_values(col,ascending=True)
    if(min_value>0):
        ids = np.where(np.array(m_stats_local[col])>min_value)[0]
    else:
        ids = range(len(m_stats_local[col]))
    if(max_entries>0):
        ids = ids[-min(max_entries,len(m_stats_local)):]
    y_pos = np.arange(len(ids))

    fig = plt.figure()
    ax = plt.subplot()
    if(title is None):
        plt.title("Global m-value stats (based on "+col+")")
    else:
        pass
        ax.set_title(title, size=6)
    plt.barh(y_pos, np.array(m_stats_local[col])[ids])
    if("col" in m_stats_local.columns):
        plt.yticks(y_pos, np.array(m_stats_local["col"])[ids], rotation=26)
    plt.xscale('log')
    if(col=='meanabs'):
        # plt.xlabel('mean(|m-value|)')
        plt.xlabel(r'mean$(|\,\bar m^{\,f}\,|)$', labelpad=0.1)
        # add_text(r'mean$(|\,\bar m^{\,f}\,|)$', ax=ax, loc='lower left')

    if(filename is not None):
        fig.tight_layout(pad=0.1)
        fig.savefig(filename, bbox_inches='tight')
    plt.show()


def calculate_global_prediff_stats_clas(m_list,y,cols=None,sortby="meanabs"):
    '''analogue of calculate_global_prediff_stats for classification (returns a df for every class)'''
    res=[]
    for c in range(len(y[0])):
        m_listc=[]
        for m in m_list:
            m_tmp = m.iloc[np.where(y==c)[0]].copy()#select just target class
            m_tmp["mean"]=m_tmp["mean"].apply(lambda x:x[c])#select score for this particular class
            m_tmp["conf95"]=m_tmp.apply(lambda x: x["high"][c]- x["low"][c],axis=1)
            m_listc.append(m_tmp)
        res.append(calculate_global_m_value_stats(m_listc,cols,sortby))
    return res

def calculate_global_preddiff_stats(m_list,cols=None,sortby="meanabs"):
    '''calculates global preddiff stats for each impute_col (operates on output of preddiff.relevances())'''
    res = []
    for i,m in enumerate(m_list):
        tmp={"id":i, "median": m["mean"].median(), "iqr": iqr(m["mean"]), "mean": m["mean"].mean(), "std": m["mean"].std(), "min":m["mean"].min(), "max":m["mean"].max(), "low": m["low"].mean(), "high": m["high"].mean(), "conf95": m.apply(lambda x: x["high"]-x["low"],axis=1).mean(), "absmean": abs(m["mean"].mean()), "meanabs": m["mean"].abs().mean()}
        if(cols is not None):
            tmp["col"]=cols[i]
        res.append(tmp)
    return pd.DataFrame(res).sort_values(sortby,ascending=False)


