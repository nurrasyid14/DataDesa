#fuzzycmeans.py

from skfuzzy import cmeans
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import argparse
import os
import sys

class FuzzyCMeans:
    '''Fuzzy C-Means Clustering Implementation
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The input data to cluster.
        n_clusters : int, optional
        The number of clusters to form. Default is 3.
        m : float, optional
        The fuzziness parameter. Default is 2.
        error : float, optional
        The stopping criterion. Default is 0.005.
        maxiter : int, optional
        The maximum number of iterations. Default is 1000.
        '''
    def __init__(self, data, n_clusters=3, m=2, error=0.005, maxiter=1000):
        self.data = data
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.centers = None
        self.u = None

    def fit(self):
        '''Fit the Fuzzy C-Means model to the data.'''
        cntr, u, _, _, _, _, _ = cmeans(
            self.data.T, 
            c=self.n_clusters, 
            m=self.m, 
            error=self.error, 
            maxiter=self.maxiter, 
            init=None
        )
        self.centers = cntr
        self.u = u

    def predict(self):
        '''Predict the closest cluster each sample in data belongs to.'''
        if self.u is None:
            raise ValueError("Model has not been fitted yet.")
        return np.argmax(self.u, axis=0)

    def plot(self, output_file='fuzzy_cmeans_plot.html'):
        '''Plot the clustering results and save to an HTML file.'''
        if self.centers is None or self.u is None:
            raise ValueError("Model has not been fitted yet.")
        
        df = pd.DataFrame(self.data.T, columns=['X', 'Y'])
        df['Cluster'] = self.predict()

        fig = px.scatter(df, x='X', y='Y', color='Cluster', title='Fuzzy C-Means Clustering')
        
        for i, center in enumerate(self.centers):
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(size=15, symbol='x', color='black'),
                name=f'Center {i}'
            ))

        pio.write_html(fig, file=output_file, auto_open=True)