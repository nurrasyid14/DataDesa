# -------------------------------------------------------------
# app.py  —  Cleaned Version (KMeans + Fuzzy C-Means only)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from clustering.pipeline import Pipeline
from clustering.clusterer import Clustering
from clustering.fuzzycmeans import FuzzyCMeans


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(page_title="Indeks Desa Membangun Indonesia", layout="wide")

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


# ======================================================================
# LOAD DATA
# ======================================================================
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()


# ======================================================================
# HEADER
# ======================================================================
st.title("Indeks Desa Membangun Indonesia")
st.caption(WARNING_STR)

with st.expander("Legend - Index Definitions"):
    for key, value in LEGENDS.items():
        st.write(f"**{key}** : {value}")


# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.header("Controls")

method_choice = st.sidebar.selectbox(
    "Clustering Method:",
    ["kmeans", "fuzzy", "all"]
)

n_clusters = st.sidebar.slider("Clusters", 2, 8, 4)

run_button = st.sidebar.button("Run")


# Initialize Pipeline
pipe = Pipeline(df_raw)


# ======================================================================
# EXECUTION
# ======================================================================
if not run_button:
    st.info("Click **Run** to begin processing.")
    st.stop()

# PREPROCESS
with st.spinner("Cleaning dataset..."):
    cleaned = pipe.preprocess()

st.success("Preprocessing Complete.")

st.subheader("Dataset Preview")
st.dataframe(cleaned.head(50))


# ======================================================================
# PIE CHART — NATIONAL STATUS DISTRIBUTION
# ======================================================================
st.subheader("Distribusi Status IDM (Nasional)")

status_series = cleaned["STATUS_IDM_2024"].map(STATUS_MAP)
status_counts = status_series.value_counts()

fig_status = px.pie(
    names=status_counts.index,
    values=status_counts.values,
    title="Distribusi STATUS_IDM_2024 (NASIONAL)",
    hole=0.35
)
st.plotly_chart(fig_status, width="stretch")


# ======================================================================
# PIE CHARTS — PER PROVINCE FOR EACH STATUS
# ======================================================================
st.subheader("Distribusi Status per Provinsi")

if "NAMA_PROVINSI" in cleaned.columns:

    for code, label in STATUS_MAP.items():
        st.markdown(f"### **Status: {label.upper()}**")

        sub = cleaned[cleaned["STATUS_IDM_2024"] == code]

        if len(sub) == 0:
            st.info("No data available for this status.")
            continue

        prov_counts = sub["NAMA_PROVINSI"].value_counts()

        fig_prov = px.pie(
            names=prov_counts.index,
            values=prov_counts.values,
            title=f"Sebaran Provinsi — {label}",
            hole=0.3
        )
        st.plotly_chart(fig_prov, width="stretch")


# ======================================================================
# TOP 20 DESA
# ======================================================================
st.subheader("Top 20 Desa Berdasarkan NILAI_IDM_2024")

if "NILAI_IDM_2024" in cleaned.columns:
    top20 = cleaned.nlargest(20, "NILAI_IDM_2024")

    st.dataframe(top20.reset_index(drop=True))

    fig_top20 = px.bar(
        top20,
        x="NAMA_DESA",
        y="NILAI_IDM_2024",
        title="Top 20 Desa",
        hover_data=["NAMA_KABUPATEN", "NAMA_KECAMATAN"]
    )
    fig_top20.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_top20, width="stretch")


# ======================================================================
# CLUSTERING
# ======================================================================
st.subheader("Clustering Results")


# Helper plotting function
def plot_clusters(X, labels, title):
    X = np.array(X)
    from sklearn.decomposition import PCA

    X2 = PCA(n_components=2).fit_transform(X)

    df = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "Cluster": labels.astype(str)
    })

    return px.scatter(df, x="PC1", y="PC2", color="Cluster", title=title)


methods = (
    ["kmeans", "fuzzy"]
    if method_choice == "all"
    else [method_choice]
)


# RUN CLUSTERING
for method in methods:

    st.markdown(f"## Method: **{method.upper()}**")

    if method == "kmeans":
        labels = pipe.kmeans(n_clusters=n_clusters)
        clustered_df = pipe.attach(labels, "KMeans")

        fig = plot_clusters(pipe.numeric_df.values, labels, "KMeans Clusters")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(clustered_df.head(20))

        # Subcluster pie per STATUS
        st.markdown("### Subcluster Distribution per STATUS")
        for code, status_label in STATUS_MAP.items():
            sub = clustered_df[clustered_df["STATUS_IDM_2024"] == code]
            if len(sub) == 0:
                continue
            pie = px.pie(
                sub,
                names="KMeans",
                title=f"KMeans — {status_label}",
                hole=0.25
            )
            st.plotly_chart(pie, width="stretch")


    elif method == "fuzzy":
        labels = pipe.fuzzy_cmeans(n_clusters=n_clusters, m=2.0)
        clustered_df = pipe.attach(labels, "Fuzzy")

        fig = plot_clusters(pipe.numeric_df.values, labels, "Fuzzy C-Means")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(clustered_df.head(20))

        # Membership matrix
        try:
            u = pipe.fuzzy_model.u.T
            st.subheader("Fuzzy Membership Matrix (first 20 rows)")
            cols = [f"p_cluster_{i}" for i in range(u.shape[1])]
            st.dataframe(pd.DataFrame(u, columns=cols).head(20))
        except:
            st.warning("Membership matrix unavailable.")

        # Subcluster pie per STATUS
        st.markdown("### Subcluster Distribution per STATUS")
        for code, status_label in STATUS_MAP.items():
            sub = clustered_df[clustered_df["STATUS_IDM_2024"] == code]
            if len(sub) == 0:
                continue
            pie = px.pie(
                sub,
                names="Fuzzy",
                title=f"Fuzzy C-Means — {status_label}",
                hole=0.25
            )
            st.plotly_chart(pie, width="stretch")


st.success("Finished Clustering.")
