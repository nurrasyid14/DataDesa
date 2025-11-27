import pandas as pd
import numpy as np

from .preprocessor import Cleaner
from .clusterer import Clustering
from .fuzzycmeans import FuzzyCMeans


class Pipeline:

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.cleaned_df = None
        self.numeric_df = None

        self.cluster_labels = None
        self.cluster_model = None

        self.fuzzy_model = None
        self.fuzzy_labels = None

    # ------------------------------------------------------------
    # 1. PREPROCESSING 
    # ------------------------------------------------------------
    def preprocess(self):
        df = self.raw_df.copy()

        # A. Remove non-numeric column "Keterangan"
        if "Keterangan" in df.columns:
            df = df.drop(columns=["Keterangan"])

        # B. Check required numeric columns exist
        numeric_cols = ["IKS_2024", "IKE_2024", "IKL_2024", "NILAI_IDM_2024"]
        for col in numeric_cols:
            if col not in df.columns:
                raise KeyError(f"ERROR: Missing required column: {col}")

        # C. Last 4 rows → set those numeric columns to zero
        df.loc[df.tail(4).index, numeric_cols] = 0

        # D. Encode STATUS_IDM_2024
        mapping = {
            "sangat tertinggal": -1,
            "tertinggal": 0,
            "berkembang": 1,
            "maju": 2
        }

        df["STATUS_IDM_2024"] = (
            df["STATUS_IDM_2024"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(mapping)
            .fillna(0)  # <- important!
        )

        # Save clean df
        self.cleaned_df = df

        # Build numeric matrix safely
        cols = numeric_cols + ["STATUS_IDM_2024"]

        self.numeric_df = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        return self.cleaned_df

    # ------------------------------------------------------------
    # 2. CLUSTERING METHODS
    # ------------------------------------------------------------
    def kmeans(self, n_clusters=4, random_state=42):
        cluster = Clustering(self.numeric_df)
        labels, model = cluster.kmeans_clustering(
            self.numeric_df.values,
            n_clusters=n_clusters,
            random_state=random_state
        )
        self.cluster_labels = labels
        self.cluster_model = model
        return labels

    def dbscan(self, eps=0.5, min_samples=5):
        cluster = Clustering(self.numeric_df)
        labels, model = cluster.dbscan_clustering(
            self.numeric_df.values,
            eps=eps,
            min_samples=min_samples
        )
        self.cluster_labels = labels
        self.cluster_model = model
        return labels

    def agglomerative(self, n_clusters=4, linkage="ward"):
        cluster = Clustering(self.numeric_df)
        labels, model = cluster.agglomerative_clustering(
            self.numeric_df.values,
            n_clusters=n_clusters,
            linkage_method=linkage
        )
        self.cluster_labels = labels
        self.cluster_model = model
        return labels

    def hierarchical(self, method="ward"):
        cluster = Clustering(self.numeric_df)
        return cluster.hierarchical_clustering(
            self.numeric_df.values,
            method=method
        )

    def fuzzy_cmeans(self, n_clusters=4, m=2.0, error=0.005, maxiter=1000):
        model = FuzzyCMeans(
            data=self.numeric_df.values,
            n_clusters=n_clusters,
            m=m,
            error=error,
            maxiter=maxiter
        )
        model.fit()
        labels = model.predict()

        self.fuzzy_model = model
        self.fuzzy_labels = labels
        return labels

    # ------------------------------------------------------------
    # 3. ATTACH LABELS BACK
    # ------------------------------------------------------------
    def attach(self, labels, name="Cluster"):
        df = self.cleaned_df.copy()
        df[name] = labels
        return df

    # ------------------------------------------------------------
    # 4. STATIC ENTRYPOINT  ✔ FIXED
    # ------------------------------------------------------------
    @staticmethod
    def run(df, method="kmeans", **kwargs):
        pipe = Pipeline(df)
        pipe.preprocess()

        if method == "kmeans":
            labels = pipe.kmeans(**kwargs)
            return pipe.attach(labels, "KMeans")

        elif method == "dbscan":
            labels = pipe.dbscan(**kwargs)
            return pipe.attach(labels, "DBSCAN")

        elif method == "agglomerative":
            labels = pipe.agglomerative(**kwargs)
            return pipe.attach(labels, "Agglomerative")

        elif method == "fuzzy":
            labels = pipe.fuzzy_cmeans(**kwargs)
            return pipe.attach(labels, "Fuzzy")

        else:
            raise ValueError(f"Unknown method: {method}")
