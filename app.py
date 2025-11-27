# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# Import your Pipeline and other modules
from clustering.pipeline import Pipeline
from clustering import Clustering
from clustering.fuzzycmeans import FuzzyCMeans

st.set_page_config(page_title="Indeks Desa Membangun Indonesia", layout="wide")

# ---------------------------
# Constants / Mappings
# ---------------------------
STATUS_MAP = {
    -1: "sangat tertinggal",
     0: "tertinggal",
     1: "berkembang",
     2: "maju"
}

# Legends text to show on UI
LEGENDS = {
    "IKL": "Indeks Ketahanan Lingkungan (IKL)",
    "IKS": "Indeks Ketahanan Sosial (IKS)",
    "IKE": "Indeks Ketahanan Ekonomi (IKE)"
}

WARNING_STR = "Data hanya memuat Desa, kelurahan tidak diikutsertakan."

# ---------------------------
# Load dataset (user said dataset exists locally)
# replace 'data/idm_dataset.csv' with your actual path
# ---------------------------
DATA_PATH = "indeks-desa-membangun-tahun-2024-hasil-pemutakhiran.xlsx"  # EDIT THIS PATH if needed

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df

# Try to load dataset
try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Please place your dataset there or update DATA_PATH in app.py.")
    st.stop()

# ---------------------------
# App header
# ---------------------------
st.title("Indeks Desa Membangun Indonesia")
st.caption(WARNING_STR)

with st.expander("Legend - Index definitions"):
    st.write(f"- **IKL** : {LEGENDS['IKL']}")
    st.write(f"- **IKS** : {LEGENDS['IKS']}")
    st.write(f"- **IKE** : {LEGENDS['IKE']}")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
method_choice = st.sidebar.selectbox("Clustering method (single-run):", ["kmeans", "dbscan", "agglomerative", "fuzzy", "all"])
n_clusters = st.sidebar.slider("Number of clusters (for KMeans/Agglomerative/Fuzzy)", min_value=2, max_value=8, value=4)
dbscan_eps = st.sidebar.number_input("DBSCAN eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=1, max_value=20, value=5)
run_button = st.sidebar.button("Run pipeline")

# ---------------------------
# Preprocess + pipeline object
# ---------------------------
pipe = Pipeline(df_raw)

if run_button:
    with st.spinner("Preprocessing dataset..."):
        cleaned = pipe.preprocess()  # internal rules applied (drop Keterangan, last-4->0, encode status)
    
    st.success("Preprocessing completed.")
    
    # Prepare numeric df used internally (your Pipeline has self.numeric_df)
    numeric_df = pipe.numeric_df.copy()
    
    # Show small overview
    st.subheader("Dataset preview (cleaned)")
    st.dataframe(cleaned.head(50))

    # --- Category pie chart (STATUS_IDM_2024) ---
    st.subheader("STATUS_IDM_2024 distribution (original text labels)")
    # Map numeric status back to textual labels (handle nan)
    status_series = cleaned["STATUS_IDM_2024"].fillna(-999).map(STATUS_MAP).fillna("unknown")
    status_counts = status_series.value_counts().reindex(["sangat tertinggal","tertinggal","berkembang","maju","unknown"]).dropna()
    fig_pie = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        title="Distribusi STATUS_IDM_2024",
        hole=0.35
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Top 20 villages by NILAI_IDM_2024 ---
    st.subheader("Top 20 Desa berdasarkan NILAI_IDM_2024")
    if "NILAI_IDM_2024" in cleaned.columns:
        top20 = cleaned.sort_values("NILAI_IDM_2024", ascending=False).head(20)
        st.dataframe(top20[[
            "KODE_PROV", "NAMA_PROVINSI", "KODE_KAB", "NAMA_KABUPATEN",
            "KODE_KEC","NAMA_KECAMATAN","KODE_DESA","NAMA_DESA",
            "IKS_2024","IKE_2024","IKL_2024","NILAI_IDM_2024","STATUS_IDM_2024"
        ]].reset_index(drop=True))
        fig_bar = px.bar(top20, x="NAMA_DESA", y="NILAI_IDM_2024",
                         hover_data=["NAMA_KABUPATEN","NAMA_KECAMATAN"],
                         title="Top 20 Desa - NILAI_IDM_2024")
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Kolom NILAI_IDM_2024 tidak ditemukan di dataset.")

    # ---------------------------
    # Run requested clustering(s)
    # ---------------------------
    st.subheader("Clustering Results & Visualizations")
    # Helper to create a 2D scatter with clusters (PCA if needed)
    def scatter_clusters(X, labels, title):
        X = np.array(X)
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
            df_plot = pd.DataFrame({"PC1": X2[:,0], "PC2": X2[:,1], "Cluster": labels.astype(str)})
            fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", title=title, hover_data=None)
        elif X.shape[1] == 2:
            df_plot = pd.DataFrame({"X1": X[:,0], "X2": X[:,1], "Cluster": labels.astype(str)})
            fig = px.scatter(df_plot, x="X1", y="X2", color="Cluster", title=title)
        else:
            # fallback: show first two numeric cols
            cols = numeric_df.columns.tolist()
            df_plot = pd.DataFrame({"X1": X[:,0], "X2": X[:,0], "Cluster": labels.astype(str)})
            fig = px.scatter(df_plot, x="X1", y="X2", color="Cluster", title=title)
        return fig

    # Run all if requested
    methods_to_run = ["kmeans","dbscan","agglomerative","fuzzy"] if method_choice == "all" else [method_choice]

    for method in methods_to_run:
        st.markdown(f"### Method: **{method.upper()}**")
        if method == "kmeans":
            labels = pipe.kmeans(n_clusters=n_clusters)
            fig = scatter_clusters(pipe.numeric_df.values, labels, f"KMeans (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)
            # attach & preview
            df_out = pipe.attach(labels, name="KMeans")
            st.write("Sample labeled rows (KMeans):")
            st.dataframe(df_out.head(10))

        elif method == "dbscan":
            labels = pipe.dbscan(eps=dbscan_eps, min_samples=dbscan_min_samples)
            fig = scatter_clusters(pipe.numeric_df.values, labels, f"DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})")
            st.plotly_chart(fig, use_container_width=True)
            df_out = pipe.attach(labels, name="DBSCAN")
            st.write("Unique DBSCAN labels (note -1 => noise):")
            st.write(sorted(list(set(labels))))
            st.dataframe(df_out.head(10))

        elif method == "agglomerative":
            labels = pipe.agglomerative(n_clusters=n_clusters, linkage="ward")
            fig = scatter_clusters(pipe.numeric_df.values, labels, f"Agglomerative (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)
            df_out = pipe.attach(labels, name="Agglomerative")
            st.dataframe(df_out.head(10))

            # dendrogram
            Z = pipe.hierarchical(method="ward")
            # use plotly's dendrogram
            try:
                fig_dend = ff.create_dendrogram(Z, orientation='top')
                fig_dend.update_layout(width=1000, height=500, title="Hierarchical Dendrogram")
                st.plotly_chart(fig_dend, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create dendrogram: {e}")

        elif method == "fuzzy":
            fuzzy_labels = pipe.fuzzy_cmeans(n_clusters=n_clusters, m=2.0)
            # fuzzy model is pipe.fuzzy_model, membership matrix at pipe.fuzzy_model.u
            st.write("Fuzzy C-Means: hard labels (argmax of membership):")
            st.dataframe(pd.DataFrame({"Label": fuzzy_labels}).head(10))

            # Show membership probabilities for first 20 samples
            try:
                u = pipe.fuzzy_model.u  # shape: (n_clusters, n_samples)
                # transpose to (n_samples, n_clusters)
                u_t = u.T
                df_u = pd.DataFrame(u_t, columns=[f"prob_cluster_{i}" for i in range(u_t.shape[1])])
                st.subheader("Example fuzzy membership probabilities (first 20 samples)")
                st.dataframe(df_u.head(20))
            except Exception as e:
                st.warning(f"Could not retrieve fuzzy membership matrix: {e}")

            # scatter using fuzzy hard labels
            fig = scatter_clusters(pipe.numeric_df.values, fuzzy_labels, f"FuzzyCMeans (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Additional infographics / summary metrics
    # ---------------------------
    st.subheader("Summary Metrics by Cluster (KMeans result if available)")
    if pipe.cluster_labels is not None:
        # Use last assigned cluster_labels (likely from last clustering method run)
        labels = pipe.cluster_labels
        df_summary = pipe.cleaned_df.copy()
        df_summary["Cluster"] = labels
        group = df_summary.groupby("Cluster").agg({
            "IKS_2024":"mean","IKE_2024":"mean","IKL_2024":"mean","NILAI_IDM_2024":"mean","STATUS_IDM_2024":"count"
        }).rename(columns={"STATUS_IDM_2024":"count"}).reset_index()
        st.dataframe(group)

        # Radar-like (spider) charts can be complex; show bar averages instead
        fig_avg = px.bar(group.melt(id_vars="Cluster", value_vars=["IKS_2024","IKE_2024","IKL_2024","NILAI_IDM_2024"]),
                         x="variable", y="value", color="Cluster", barmode="group",
                         title="Average Index per Cluster")
        st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.info("Run a clustering method to see cluster summary metrics here.")

    st.success("All visualizations generated.")
else:
    st.info("Click **Run pipeline** in the sidebar to preprocess, cluster, and produce infographics.")
