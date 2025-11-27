# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

from clustering.pipeline import Pipeline
from clustering.clusterer import Clustering
from clustering.fuzzycmeans import FuzzyCMeans

st.set_page_config(page_title="Indeks Desa Membangun Indonesia", layout="wide")

# -----------------------------------------
# CONFIGS
# -----------------------------------------
STATUS_MAP = {
    -1: "sangat tertinggal",
     0: "tertinggal",
     1: "berkembang",
     2: "maju"
}

LEGENDS = {
    "IKL": "Indeks Ketahanan Lingkungan (IKL)",
    "IKS": "Indeks Ketahanan Sosial (IKS)",
    "IKE": "Indeks Ketahanan Ekonomi (IKE)"
}

DATA_PATH = "indeks-desa-membangun-tahun-2024-hasil-pemutakhiran.xlsx"
WARNING_STR = "Data hanya memuat Desa, kelurahan tidak diikutsertakan."


@st.cache_data
def load_data(path):
    return pd.read_excel(path)


# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("Indeks Desa Membangun Indonesia")
st.caption(WARNING_STR)

with st.expander("Legend - Index definitions"):
    for key, value in LEGENDS.items():
        st.write(f"**{key}** : {value}")


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("Controls")

method_choice = st.sidebar.selectbox(
    "Clustering method:", 
    ["kmeans", "dbscan", "agglomerative", "fuzzy", "all"]
)

n_clusters = st.sidebar.slider("Clusters", 2, 8, 4)

dbscan_eps = st.sidebar.number_input("DBSCAN eps", 0.1, 10.0, 0.5)
dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 20, 5)

run_button = st.sidebar.button("Run pipeline")


# ----------------------------------------------------
# INITIALIZE PIPELINE
# ----------------------------------------------------
pipe = Pipeline(df_raw)


# ====================================================
# MAIN EXECUTION
# ====================================================
if not run_button:
    st.info("Click **Run pipeline** to start.")
    st.stop()


# ----------------------------------------------------
# PREPROCESS
# ----------------------------------------------------
with st.spinner("Preprocessing dataset..."):
    cleaned = pipe.preprocess()

st.success("Done preprocessing.")

st.subheader("Dataset Preview")
st.dataframe(cleaned.head(50))


# ----------------------------------------------------
# DISTRIBUTION PIE CHART
# ----------------------------------------------------
st.subheader("STATUS_IDM_2024 Distribution")
status_series = cleaned["STATUS_IDM_2024"].map(STATUS_MAP).fillna("unknown")
status_counts = status_series.value_counts()

fig_pie = px.pie(
    names=status_counts.index,
    values=status_counts.values,
    title="Distribusi STATUS_IDM_2024",
    hole=0.35
)
st.plotly_chart(fig_pie, use_container_width=True)


# ----------------------------------------------------
# TOP 20 BAR CHART
# ----------------------------------------------------
st.subheader("Top 20 Desa NILAI_IDM_2024")
if "NILAI_IDM_2024" in cleaned.columns:
    top20 = cleaned.nlargest(20, "NILAI_IDM_2024")
    st.dataframe(top20.reset_index(drop=True))

    fig_bar = px.bar(
        top20,
        x="NAMA_DESA",
        y="NILAI_IDM_2024",
        hover_data=["NAMA_KABUPATEN", "NAMA_KECAMATAN"],
        title="Top 20 Desa"
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)


# ----------------------------------------------------
# CLUSTERING SECTION
# ----------------------------------------------------
st.subheader("Clustering Results")

def plot_clusters(X, labels, title):
    X = np.array(X)
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X2 = PCA(n_components=2).fit_transform(X)
        df = pd.DataFrame({"PC1": X2[:,0], "PC2": X2[:,1], "Cluster": labels.astype(str)})
        return px.scatter(df, x="PC1", y="PC2", color="Cluster", title=title)
    else:
        df = pd.DataFrame({
            "X1": X[:,0],
            "X2": X[:,1] if X.shape[1] > 1 else X[:,0],
            "Cluster": labels.astype(str)
        })
        return px.scatter(df, x="X1", y="X2", color="Cluster", title=title)


methods = (
    ["kmeans", "dbscan", "agglomerative", "fuzzy"]
    if method_choice == "all"
    else [method_choice]
)

# Store results separately
results = {}

for method in methods:
    st.markdown(f"### Method: **{method.upper()}**")

    if method == "kmeans":
        labels = pipe.kmeans(n_clusters=n_clusters)
        results["kmeans"] = labels
        st.plotly_chart(plot_clusters(pipe.numeric_df.values, labels, f"KMeans k={n_clusters}"), use_container_width=True)
        st.dataframe(pipe.attach(labels, "KMeans").head(10))

    elif method == "dbscan":
        labels = pipe.dbscan(eps=dbscan_eps, min_samples=dbscan_min_samples)
        results["dbscan"] = labels
        st.plotly_chart(plot_clusters(pipe.numeric_df.values, labels, "DBSCAN"), use_container_width=True)
        st.dataframe(pipe.attach(labels, "DBSCAN").head(10))

    elif method == "agglomerative":
        labels = pipe.agglomerative(n_clusters=n_clusters, linkage="ward")
        results["agglomerative"] = labels
        st.plotly_chart(plot_clusters(pipe.numeric_df.values, labels, "Agglomerative"), use_container_width=True)

        # dendrogram
        Z = pipe.hierarchical(method="ward")
        fig_dend = ff.create_dendrogram(Z, orientation="top")
        fig_dend.update_layout(height=500)
        st.plotly_chart(fig_dend, use_container_width=True)

    elif method == "fuzzy":
        labels = pipe.fuzzy_cmeans(n_clusters=n_clusters, m=2.0)
        results["fuzzy"] = labels

        st.dataframe(pd.DataFrame({"Label": labels}).head(10))

        # membership matrix
        try:
            u = pipe.fuzzy_model.u.T
            cols = [f"p_cluster_{i}" for i in range(u.shape[1])]
            st.dataframe(pd.DataFrame(u, columns=cols).head(20))
        except:
            st.warning("Could not read fuzzy membership matrix.")

        st.plotly_chart(plot_clusters(pipe.numeric_df.values, labels, "Fuzzy C-Means"), use_container_width=True)


st.success("Clustering completed successfully (including ALL mode).")

