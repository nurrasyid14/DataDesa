"""
Enhanced Streamlit app for IDM clustering (KMeans + Fuzzy C-Means)
- Safe for large datasets (76k rows)
- Adds cluster profiling, radar, province dominance, parallel coordinates,
  cluster centroids, PCA scatter, membership matrix, and optional choropleth
- Uses sampling for heavy visualizations to keep memory/CPU usage bounded
- Progress indicators + graceful fallbacks

Drop-in replacement for your existing app. Edit DATA_PATH / columns to match.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from clustering.pipeline import Pipeline
from clustering.clusterer import Clustering
from clustering.fuzzycmeans import FuzzyCMeans

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Indeks Desa Membangun — Enhanced", layout="wide")

DATA_PATH = "indeks-desa-membangun-tahun-2024-hasil-pemutakhiran.xlsx"
# column names used by the app (adjust if your file differs)
COL_STATUS = "STATUS_IDM_2024"
COL_IDM = "NILAI_IDM_2024"
COL_IKL = "IKL_2024"
COL_IKS = "IKS_2024"
COL_IKE = "IKE_2024"
COL_PROV = "NAMA_PROVINSI"
COL_DESA = "NAMA_DESA"

# thresholds
SAMPLE_FOR_PLOTTING = 5000        # number of points for heavy interactive plots
MAX_FCM_ROWS = 50000              # only run fuzzy cmeans if rows below this

# -------------------- HELPERS & CACHES --------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

@st.cache_data
def sample_df(df: pd.DataFrame, n: int):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def safe_sample_for_plot(df: pd.DataFrame):
    return sample_df(df, min(SAMPLE_FOR_PLOTTING, len(df)))


# -------------------- LOAD --------------------
try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

# quick column sanity
for c in [COL_STATUS, COL_IDM, COL_IKL, COL_IKS, COL_IKE, COL_PROV, COL_DESA]:
    if c not in df_raw.columns:
        st.warning(f"Warning: column '{c}' not found in dataset. Some features will be disabled.")

# initialize pipeline
pipe = Pipeline(df_raw)

# -------------------- UI: controls --------------------
st.title("Indeks Desa Membangun — Enhanced Clustering & Insights")
st.caption("Optimized for large datasets: sampling is used for heavy charts to preserve responsiveness.")

with st.sidebar:
    st.header("Controls")
    method_choice = st.selectbox("Clustering Method:", ["kmeans", "fuzzy", "both"], index=2)
    n_clusters = st.slider("Clusters", 2, 8, 4)
    run_button = st.button("Run Clustering")
    show_choropleth = st.checkbox("Enable choropleth (requires GeoJSON/TopoJSON upload)")
    geo_file = st.file_uploader("Upload provinces GeoJSON/TopoJSON (optional)", type=["json", "geojson", "topojson"])

if not run_button:
    st.info("Click **Run Clustering** to begin. Optionally upload GeoJSON for choropleth maps.")
    st.stop()

# -------------------- PREPROCESS --------------------
with st.spinner("Cleaning & preparing data..."):
    cleaned = pipe.preprocess()

st.success("Preprocessing complete.")

# quick dataframe preview
st.subheader("Dataset Preview")
st.dataframe(cleaned.head(50))

# numeric dataframe used for clustering
numeric_df = pipe.numeric_df.copy()
X = numeric_df.values
n_rows = len(numeric_df)

st.markdown(f"**Observations (numeric rows):** {n_rows}")

# -------------------- UTILS: plotting --------------------

def pca_scatter(X_in, labels, title="PCA Projection"):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_in)
    dfp = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1], "cluster": labels.astype(str)})
    fig = px.scatter(dfp, x="PC1", y="PC2", color="cluster", title=title, opacity=0.7)
    return fig


def radar_chart(df_profile, method_name="Method"):
    # df_profile: dataframe with cluster index and columns IKL, IKS, IKE, IDM
    agg = df_profile.set_index("cluster")[ [COL_IKL, COL_IKS, COL_IKE, COL_IDM] ]
    fig = go.Figure()
    for cluster in agg.index:
        fig.add_trace(go.Scatterpolar(r=agg.loc[cluster].values, theta=agg.columns, fill='toself', name=f"Cluster {cluster}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"{method_name} — Cluster Radar")
    return fig


# -------------------- CLUSTERING EXECUTION --------------------

results = {}  # store outputs per method

if method_choice in ("kmeans", "both"):
    with st.spinner("Running KMeans..."):
        labels_k = pipe.kmeans(n_clusters=n_clusters)
        clustered_k = pipe.attach(labels_k, "KMeans")
    results['kmeans'] = {'labels': labels_k, 'df': clustered_k}
    st.success("KMeans finished.")

if method_choice in ("fuzzy", "both"):
    if n_rows > MAX_FCM_ROWS:
        st.warning(f"Fuzzy C-Means skipped: dataset too large ({n_rows} rows). Max allowed: {MAX_FCM_ROWS}.")
        results['fuzzy'] = None
    else:
        with st.spinner("Running Fuzzy C-Means..."):
            labels_f = pipe.fuzzy_cmeans(n_clusters=n_clusters, m=2.0)
            clustered_f = pipe.attach(labels_f, "Fuzzy")
        results['fuzzy'] = {'labels': labels_f, 'df': clustered_f}
        st.success("Fuzzy C-Means finished.")

# -------------------- VISUALIZATIONS & INSIGHTS --------------------

st.header("Clustering Insights")

for method, out in results.items():
    if out is None:
        st.markdown(f"## Method: **{method.upper()}** — skipped")
        continue

    labels = out['labels']
    clustered_df = out['df']

    st.markdown(f"## Method: **{method.upper()}**")

    # PCA scatter (sample for plotting if large)
    try:
        sample_for_plot = safe_sample_for_plot(numeric_df)
        sample_idx = sample_for_plot.index
        # get labels for sample (labels are aligned with numeric_df)
        sample_labels = np.array(labels)[sample_idx]
        fig = pca_scatter(sample_for_plot.values, sample_labels, title=f"{method.upper()} — PCA (sample)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"PCA scatter failed for {method}: {e}")

    # Cluster profiling: means, counts, mins, max
    st.markdown("### Cluster Profiling")
    profiling_cols = [c for c in [COL_IKL, COL_IKS, COL_IKE, COL_IDM] if c in clustered_df.columns]
    clustered_df = clustered_df.copy()
    clustered_df['cluster'] = np.array(labels)

    profile = (
        clustered_df
        .groupby('cluster')[profiling_cols]
        .agg(['count', 'mean', 'median', 'min', 'max', 'std'])
    )
    # flatten columns
    profile.columns = ["_".join(col).strip() for col in profile.columns.values]
    st.dataframe(profile)

    # Radar chart (use cluster means)
    try:
        mean_df = clustered_df.groupby('cluster')[profiling_cols].mean().reset_index()
        mean_df.rename(columns={'cluster': 'cluster'}, inplace=True)
        st.plotly_chart(radar_chart(mean_df, method_name=method.upper()), use_container_width=True)
    except Exception as e:
        st.warning(f"Radar chart failed for {method}: {e}")

    # Province dominance in each cluster (top provinces per cluster)
    st.markdown("### Province dominance per Cluster (Top 5 provinces by count)")
    prov_table = (
        clustered_df
        .groupby(['cluster', COL_PROV])
        .size()
        .reset_index(name='count')
        .sort_values(['cluster', 'count'], ascending=[True, False])
    )
    # show top 5 per cluster
    top_prov = prov_table.groupby('cluster').head(5).reset_index(drop=True)
    st.dataframe(top_prov)

    # stacked bar: cluster distribution by province (sampled for speed)
    try:
        sample_df_for_bar = safe_sample_for_plot(clustered_df)
        fig = px.bar(sample_df_for_bar, x=COL_PROV, color='cluster', title=f"{method.upper()} — Province distribution (sample)")
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Province bar chart failed for {method}: {e}")

    # Parallel coordinates for cluster signature (sample to SAMPLE_FOR_PLOTTING)
    try:
        pc_sample = safe_sample_for_plot(clustered_df)
        dims = [c for c in profiling_cols]
        fig = px.parallel_coordinates(pc_sample, dimensions=dims, color='cluster', title=f"{method.upper()} — Parallel Coordinates (sample)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Parallel coordinates failed for {method}: {e}")

    # Show cluster centroids / representative record
    st.markdown("### Cluster centroids / representative profiles")
    try:
        centroids = clustered_df.groupby('cluster')[profiling_cols].mean().reset_index()
        st.table(centroids)
    except Exception as e:
        st.warning(f"Centroid table failed for {method}: {e}")

    # Show top 10 records per cluster (small table)
    st.markdown("### Sample records per Cluster")
    try:
        for clus in sorted(clustered_df['cluster'].unique()):
            st.write(f"Cluster {clus} — sample 5 records")
            st.dataframe(clustered_df[clustered_df['cluster'] == clus][ [COL_DESA, COL_PROV, COL_IDM] + profiling_cols ].head(5))
    except Exception as e:
        st.warning(f"Sample records preview failed for {method}: {e}")

    # For fuzzy: show membership matrix (first N rows)
    if method == 'fuzzy':
        st.markdown("### Fuzzy membership (first 20 rows)")
        try:
            u = pipe.fuzzy_model.u.T  # shape: (n_samples, n_clusters)
            cols = [f"p_cluster_{i}" for i in range(u.shape[1])]
            st.dataframe(pd.DataFrame(u, columns=cols).head(20))
        except Exception as e:
            st.warning(f"Membership matrix unavailable: {e}")

# -------------------- OPTIONAL: Choropleth --------------------
if show_choropleth and geo_file is not None:
    st.header("Choropleth map (Province-level)")
    try:
        import json
        gj = json.load(geo_file)

        # require a mapping: province name in geo -> your province names
        # Here we aggregate by province & cluster from KMeans (if exists)
        if 'kmeans' in results and results['kmeans'] is not None:
            kdf = results['kmeans']['df'].copy()
            agg = kdf.groupby(COL_PROV)['cluster'].agg(lambda s: s.value_counts().idxmax()).reset_index()
            agg.columns = [COL_PROV, 'dominant_cluster']
            fig = px.choropleth(agg, geojson=gj, locations=COL_PROV, featureidkey='properties.name', color='dominant_cluster', title='Dominant cluster per province (KMeans)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No KMeans result to map. Run KMeans and upload GeoJSON to enable.')
    except Exception as e:
        st.warning(f"Choropleth failed: {e}")

st.success("Finished enhanced analysis.")
