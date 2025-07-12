
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def load_data():
    df = pd.read_csv("multi_cloud_service_composition.csv")
    providers = sorted(df["Cloud_Provider"].dropna().unique().tolist())
    types = sorted(df["Service_Type"].dropna().unique().tolist())
    return df, providers, types

def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def model_exists(filename):
    return os.path.exists(filename)

def preprocess(df):
    features = ["CPU_Utilization (%)", "Memory_Usage (MB)", "Network_Bandwidth (Mbps)", "Service_Latency (ms)"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # PCA for 2D plotting
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled[features])
    df_scaled["PCA1"] = pca_result[:, 0]
    df_scaled["PCA2"] = pca_result[:, 1]
    return df_scaled

def run_dbscan(df_scaled, eps=0.5, min_samples=3):
    model_path = "dbscan_model.pkl"
    features = df_scaled[["PCA1", "PCA2"]]

    if model_exists(model_path):
        model = load_model(model_path)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(features)
        save_model(model, model_path)

    df_scaled["Cluster"] = model.fit_predict(features)
    return df_scaled

def run_lda(df_scaled):
    if "Optimal_Service_Placement" not in df_scaled.columns:
        return df_scaled

    X = df_scaled[["CPU_Utilization (%)", "Memory_Usage (MB)", "Network_Bandwidth (Mbps)", "Service_Latency (ms)"]]
    y = df_scaled["Optimal_Service_Placement"]
    model_path = "lda_model.pkl"

    if model_exists(model_path):
        lda = load_model(model_path)
        try:
            X_lda = lda.transform(X)
        except Exception:
            X_lda = [[0] for _ in range(len(X))]
    else:
        lda = LDA(n_components=1)
        try:
            X_lda = lda.fit_transform(X, y)
            save_model(lda, model_path)
        except Exception:
            X_lda = [[0] for _ in range(len(X))]

    df_scaled["LDA_1D"] = X_lda
    return df_scaled

def filter_services(df, provider, service_type, min_cpu, min_memory, min_bandwidth, use_dbscan=False, use_lda=False, optimize=False, eps=0.5, min_samples=3):
    df_filtered = df.copy()

    if provider:
        df_filtered = df_filtered[df_filtered["Cloud_Provider"] == provider]
    if service_type:
        df_filtered = df_filtered[df_filtered["Service_Type"] == service_type]
    df_filtered = df_filtered[df_filtered["CPU_Utilization (%)"] >= min_cpu]
    df_filtered = df_filtered[df_filtered["Memory_Usage (MB)"] >= min_memory]
    df_filtered = df_filtered[df_filtered["Network_Bandwidth (Mbps)"] >= min_bandwidth]

    df_scaled = preprocess(df_filtered.copy())

    if use_dbscan:
        df_clustered = run_dbscan(df_scaled, eps=eps, min_samples=min_samples)
        df_filtered["Cluster"] = df_clustered["Cluster"]

    if use_lda:
        df_lda = run_lda(df_scaled)
        df_filtered["LDA_1D"] = df_lda["LDA_1D"]

    if optimize:
        return df_filtered.sort_values(by="QoS_Score", ascending=False).head(5).to_dict(orient="records")

    return df_filtered.head(10).to_dict(orient="records")

def get_cluster_plot_data(df, eps=0.5, min_samples=3):
    df_scaled = preprocess(df.copy())
    df_clustered = run_dbscan(df_scaled, eps=eps, min_samples=min_samples)
    return df_clustered[["Service_ID", "PCA1", "PCA2", "Cluster"]]
