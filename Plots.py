import os


import matplotlib.pyplot as plt
import seaborn as sns




def save_plot(plot: plt, path:os.path):


def scatter_plot(ax, data1, data2, param_dict: dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.scatter(data1, data2, **param_dict)
    return out


def add_text(ax, loc_x, loc_y, text: str):
    out = ax.text(loc_x, loc_y, text)
    return out



def plot_scatter(pred: pd.Series, test: pd.Series):
    fig, ax = plt.subplots()
    sns.set_theme()
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(test.name)
    x = convert_to_np(pred)
    y = convert_to_np(test)
    ax.plot(x, y)



def plot_heatmap(ax, data: pd.DataFrame, column: str, param_dict: dict):
    out = sns.heatmap(data, column, **param_dict)
