# clustering.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.figure_factory as ff


class Clustering:
    def __init__(self, data):
        """
        data: pandas DataFrame or NumPy array
        """
        self.data = data

    # ------------------------------------------
    # KMEANS
    # ------------------------------------------
    def kmeans_clustering(self, X, n_clusters=3, random_state=42):
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model

    # ------------------------------------------
    # DBSCAN
    # ------------------------------------------
    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print("Warning: DBSCAN produced fewer than 2 clusters.")

        return labels, model

    # ------------------------------------------
    # AGGLOMERATIVE CLUSTERING
    # ------------------------------------------
    def agglomerative_clustering(self, X, n_clusters=3, linkage_method="ward"):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        return labels, model

    # ------------------------------------------
    # HIERARCHICAL LINKAGE MATRIX
    # ------------------------------------------
    def hierarchical_clustering(self, X, method="ward"):
        Z = linkage(X, method=method)
        return Z

    # ------------------------------------------
    # PLOTLY DENDROGRAM
    # ------------------------------------------
    def plot_dendrogram(self, Z, labels=None):
        """
        Z: linkage matrix
        labels: optional sample labels
        """
        fig = ff.create_dendrogram(Z, labels=labels, orientation="top")
        fig.update_layout(
            width=900,
            height=500,
            title="Hierarchical Clustering Dendrogram",
            xaxis_title="Samples",
            yaxis_title="Distance",
        )
        return fig

    # ------------------------------------------
    # OPTIONAL SCATTER PLOT FOR CLUSTERS (2D / PCA)
    # ------------------------------------------
    def plot_clusters(self, X, labels, title="Cluster Visualization"):
        X = np.array(X)

        # if >2 features → apply PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            df = {
                "PC1": X_2d[:, 0],
                "PC2": X_2d[:, 1],
                "Cluster": labels.astype(str),
            }
            fig = px.scatter(df, x="PC1", y="PC2", color="Cluster",
                             title=title,
                             symbol="Cluster")
            return fig

        # if 2 features → plot directly
        elif X.shape[1] == 2:
            df = {
                "X1": X[:, 0],
                "X2": X[:, 1],
                "Cluster": labels.astype(str),
            }
            fig = px.scatter(df, x="X1", y="X2", color="Cluster",
                             title=title,
                             symbol="Cluster")
            return fig

        else:
            raise ValueError("Data must have at least 2 dimensions to visualize clusters.")
