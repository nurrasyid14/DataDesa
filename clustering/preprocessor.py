# preprocessor.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Union, List


class Cleaner:
    """Class for cleaning datasets: missing values, duplicates, outliers, and encoding."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.initial_shape = self.data.shape
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.outliers: Optional[pd.DataFrame] = None

    def inspect(self) -> None:
        """Basic inspection of the dataset."""
        print("Initial Data Shape:", self.initial_shape)
        print("\nData Types:\n", self.data.dtypes)
        print("\nFirst 5 Rows:\n", self.data.head())
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nDuplicate Rows:", self.data.duplicated().sum())
        print("\nStatistical Summary:\n", self.data.describe(include="all"))

    def handle_missing_values(self, strategy: str = "mean", fill_value: Optional[Union[str, int, float]] = None) -> pd.DataFrame:
        """Fill missing values using mean, median, mode, or constant."""
        if strategy == "mean":
            self.cleaned_data = self.data.fillna(self.data.mean(numeric_only=True))
        elif strategy == "median":
            self.cleaned_data = self.data.fillna(self.data.median(numeric_only=True))
        elif strategy == "mode":
            self.cleaned_data = self.data.fillna(self.data.mode().iloc[0])
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("Must provide fill_value when using 'constant' strategy.")
            self.cleaned_data = self.data.fillna(fill_value)
        else:
            raise ValueError("Invalid strategy. Choose 'mean', 'median', 'mode', 'constant'.")
        return self.cleaned_data

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        df = self.cleaned_data if self.cleaned_data is not None else self.data
        before = len(df)
        self.cleaned_data = df.drop_duplicates()
        print(f"Removed {before - len(self.cleaned_data)} duplicate rows.")
        return self.cleaned_data

    def detect_outliers(self, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers using z-score or IQR method."""
        df = self.cleaned_data if self.cleaned_data is not None else self.data
        num_cols = df.select_dtypes(include="number").columns

        if method == "zscore":
            from scipy.stats import zscore
            z_scores = df[num_cols].apply(zscore)
            mask = (z_scores.abs() > threshold).any(axis=1)

        elif method == "iqr":
            Q1, Q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

        else:
            raise ValueError("Invalid method. Use 'zscore' or 'iqr'.")

        self.outliers = df[mask]
        print(f"Detected {len(self.outliers)} outliers using {method}.")
        return self.outliers

    def encode_categorical(self, method: str = "onehot") -> pd.DataFrame:
        """Encode categorical variables using one-hot or label encoding."""
        df = self.cleaned_data if self.cleaned_data is not None else self.data
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if method == "onehot":
            self.cleaned_data = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        elif method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
            self.cleaned_data = df
        else:
            raise ValueError("Invalid method. Use 'onehot' or 'label'.")
        return self.cleaned_data


class ETL:
    """Extract, Transform, Load pipeline for CSV datasets."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None

    def extract(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.filepath)
        return self.data

    def transform(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Run extract() before transform().")
        self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
        self.data = pd.get_dummies(self.data, drop_first=True)
        return self.data

    def load(self, output_filepath: str) -> None:
        if self.data is None:
            raise ValueError("No data to load. Run transform() first.")
        self.data.to_csv(output_filepath, index=False)
        print(f"Data saved to {output_filepath}.")

    def visualize(self):
        """Plot histograms and scatter matrix using Plotly."""
        if self.data is None:
            raise ValueError("Run transform() before visualize().")

        figs = {}

        # Histogram (interactive)
        for col in self.data.select_dtypes(include="number").columns:
            figs[f"hist_{col}"] = px.histogram(self.data, x=col, nbins=30)

        # Scatter matrix
        figs["scatter_matrix"] = px.scatter_matrix(self.data)

        return figs


class EDA:
    """Exploratory Data Analysis utilities."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _check_column(self, col: str) -> None:
        if col not in self.data.columns:
            raise ValueError(f"Column '{col}' not found.")

    def summary_statistics(self) -> pd.DataFrame:
        return self.data.describe(include="all")

    def missing_values(self) -> pd.Series:
        return self.data.isnull().sum()

    def correlation_matrix(self):
        """Plot correlation matrix using Plotly Heatmap."""
        numeric_df = self.data.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError("No numeric columns available for correlation matrix.")

        corr = numeric_df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                reversescale=True
            )
        )
        fig.update_layout(title="Correlation Matrix")

        return fig

    def plot(self, kind: str, x: Optional[str] = None, y: Optional[str] = None):
        """Generic plotting function using Plotly."""
        if kind == "hist":
            fig = px.histogram(self.data, x=x)
        elif kind == "box":
            fig = px.box(self.data, y=x)
        elif kind == "scatter":
            fig = px.scatter(self.data, x=x, y=y)
        elif kind == "count":
            fig = px.histogram(self.data, x=x)
        elif kind == "bar":
            fig = px.bar(self.data, x=x, y=y)
        elif kind == "violin":
            fig = px.violin(self.data, x=x, y=y, box=True)
        elif kind == "line":
            fig = px.line(self.data, x=x, y=y)
        else:
            raise ValueError("Invalid plot kind.")
        
        return fig

    def data_types(self) -> pd.Series:
        return self.data.dtypes
    
    def detect_outliers_iqr(self, col: str) -> pd.DataFrame:
        """Detect outliers in a specific column using the IQR method."""
        self._check_column(col)
        Q1 = self.data[col].quantile(0.25)
        Q3 = self.data[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))
        outliers = self.data[mask]
        print(f"Detected {len(outliers)} outliers in column '{col}' using IQR method.")
        return outliers
