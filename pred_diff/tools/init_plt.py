import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use('default')


def calculate_fig_size_in_inches(fig_width_pt: float) -> Tuple[float, float]:
    """

    :param
    fig_width_pt: use "\showthe\columnwidth" within a figure in your tex and search for it in the .log
    :return:
    tuple with fig_width, fig_height in inches using the golden mean as ratio
    """
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    # fig_size = [fig_width, fig_height]
    return fig_width, fig_height


def update_figsize(fig_width_pt: float):
    fig_size = calculate_fig_size_in_inches(fig_width_pt)
    plt.rcParams.update({'figure.figsize': fig_size})


def update_NHANES():
    plt.style.use('seaborn-paper')

    params = {
        # 'text.usetex': True,
        # 'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        # "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        # "font.sans-serif": [],
        "font.monospace": [],
        'axes.labelsize': 4,
        'font.size': 6,
        'legend.fontsize': 6,
        'legend.handlelength': 1.5,
        'legend.borderpad': 0.4,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4,
        'xtick.major.size': 1.,
        'ytick.major.size': 1.,
        'xtick.minor.size': 0.5,
        'ytick.minor.size': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # 'figure.figsize': fig_size,
        # 'savefig.directory': 'home/bluecher/Dokumente/Git',
        'savefig.format': 'pdf',
        'pgf.texsystem': 'pdflatex',
        'pgf.rcfonts': False,
        'lines.linewidth': 1,
        'lines.markersize': 1.0,
        'axes.linewidth': 0.4,
        'axes.grid': True,
        'grid.color': '#D3D3D3',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
        'savefig.pad_inches': 0.01,
        # 'text.latex.preamble': ['\usepackage{palatino}']
        # 'text.latex.preamble': [],
        # 'axes.prop_cycle': plt.cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
        # 'axes.prop_cycle': plt.cycler('color', ['#9400d3', '#009e73', '#56b4e9', '#e69f00', '#e51e10', '#f0e442', '#0072b2'])
        'axes.prop_cycle': plt.cycler('color', ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
    }
    plt.rcParams.update(params)


def update_rcParams(fig_width_pt, half_size_image=False):
    fig_size = calculate_fig_size_in_inches(fig_width_pt)
    plt.style.use('seaborn-paper')

    params = {
        # 'text.usetex': True,
        # 'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        # "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        # "font.sans-serif": [],
        "font.monospace": [],
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 8,
        'legend.handlelength': 1.5,
        'legend.borderpad': 0.4,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.figsize': fig_size,
        # 'savefig.directory': 'home/bluecher/Dokumente/Git',
        'savefig.format': 'pdf',
        'pgf.texsystem': 'pdflatex',
        'pgf.rcfonts': False,
        'lines.linewidth': 1,
        'lines.markersize': 4.0,
        'axes.linewidth': 0.4,
        'axes.grid': False,
        'grid.color': '#D3D3D3',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'savefig.pad_inches': 0.01,
        # 'text.latex.preamble': ['\usepackage{palatino}']
        # 'text.latex.preamble': [],
        # 'axes.prop_cycle': plt.cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
        # 'axes.prop_cycle': plt.cycler('color', ['#9400d3', '#009e73', '#56b4e9', '#e69f00', '#e51e10', '#f0e442', '#0072b2'])
        'axes.prop_cycle': plt.cycler('color', ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
    }
    plt.rcParams.update(params)
    if half_size_image is True:
        fig_size = calculate_fig_size_in_inches(fig_width_pt/2)
        plt.rcParams.update({
            'axes.labelsize': 8,
            'font.size': 12,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.figsize': fig_size,
        })
