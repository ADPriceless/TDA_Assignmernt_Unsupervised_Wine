import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from datassist import explore


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            explore.find_outliers(X, cut_off=self.threshold).index
        )


def plot_decomposed_features_3d(df_decomposed: pd.DataFrame, labels: pd.Series, fig_labels: str, fig_size: int = 15):
    if df_decomposed.shape[1] != 3:
        raise ValueError('df_decomposed should have 3 columns')
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    handles = ax.scatter(df_decomposed[:, 0], df_decomposed[:, 1], df_decomposed[:, 2], c=labels, alpha=0.5, cmap='inferno')
    # plot legend
    ax.legend(*handles.legend_elements(), title=labels.name, loc='best')
    ax.set_xlabel(f'{fig_labels} 1')
    ax.set_ylabel(f'{fig_labels} 2')
    ax.set_zlabel(f'{fig_labels} 3')
    plt.show()


def plot_features_3d(df: pd.DataFrame, labels: list[str], category: str, fig_size = 10):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    handles = ax.scatter(
        df.loc[:, labels[0]],
        df.loc[:, labels[1]],
        df.loc[:, labels[2]],
        c=df[category],
        alpha=0.5,
        cmap='inferno'
    )
    # plot legend
    ax.legend(*handles.legend_elements(), title=category, loc='best')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.show()
