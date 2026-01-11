import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hybrid Fake Account Detection",
    layout="wide"
)

st.title("🔍 Hybrid Fake Account Detection System")
st.write(
    """
    This demo evaluates **one social media account at a time** using a **hybrid ML system**:
    - **DBSCAN** for high-confidence coordinated behavior
    - **RandomForest** for general classification
    """
)

# -----------------------------
# Load artifacts (cached)
# -----------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    le = joblib.load("models/label_encoder.pkl")
    nn = joblib.load("models/nn_index.pkl")

    db_labels_train = np.load("models/db_labels_train.npy")

    with open("models/db_cluster_mapping.json") as f:
        db_cluster_mapping = json.load(f)

    with open("models/feature_order.json") as f:
        feature_order = json.load(f)

    return rf, scaler, pca, le, nn, db_labels_train, db_cluster_mapping, feature_order


rf, scaler, pca, le, nn, db_labels_train, db_cluster_mapping, feature_order = load_models()

# -----------------------------
# Load test users
# -----------------------------
@st.cache_data
def load_test_data():
    return pd.read_csv("test_buckets.csv")


df = load_test_data()

# -----------------------------
# Sidebar: bucket & user select
# -----------------------------
st.sidebar.header("🎯 Select User")

bucket = st.sidebar.selectbox(
    "Select account bucket (true label)",
    ["genuine", "fake_follower", "social_spambot"]
)

bucket_df = df[df["account_type"] == bucket].reset_index(drop=True)

user_idx = st.sidebar.selectbox(
    "Select user index",
    bucket_df.index
)

selected_user = bucket_df.loc[user_idx]

# -----------------------------
# Show user details
# -----------------------------
st.subheader("👤 Selected User (Input Features)")
st.dataframe(selected_user.drop("account_type"))

# -----------------------------
# Hybrid inference function
# -----------------------------
def hybrid_predict(user_row):
    # prepare feature vector
    x = user_row[feature_order].values.reshape(1, -1)

    # RF prediction
    rf_proba = rf.predict_proba(x)[0]
    rf_pred = le.inverse_transform([rf_proba.argmax()])[0]
    rf_conf = rf_proba.max()

    # DBSCAN logic
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    neighbors = nn.radius_neighbors(
        x_pca,
        return_distance=False
    )[0]

    if len(neighbors) >= 10:  # min_samples
        neighbor_clusters = db_labels_train[neighbors]
        neighbor_clusters = neighbor_clusters[neighbor_clusters != -1]

        if len(neighbor_clusters) > 0:
            cluster_id = int(pd.Series(neighbor_clusters).mode()[0])

            if str(cluster_id) in db_cluster_mapping:
                cluster_info = db_cluster_mapping[str(cluster_id)]
                return {
                    "prediction": cluster_info["label"],
                    "confidence": cluster_info["purity"],
                    "source": "DBSCAN",
                    "cluster_id": cluster_id
                }

    # fallback to RF
    return {
        "prediction": rf_pred,
        "confidence": rf_conf,
        "source": "RandomForest",
        "cluster_id": None
    }

# -----------------------------
# Evaluate button
# -----------------------------
if st.button("🚀 Evaluate Account"):
    result = hybrid_predict(selected_user)

    st.subheader("📊 Model Output")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Label", result["prediction"])
    col2.metric("Confidence", f"{result['confidence']:.2f}")
    col3.metric("Decision Source", result["source"])

    st.subheader("✅ True Label (for comparison)")
    st.write(f"**{selected_user['account_type']}**")

    if result["cluster_id"] is not None:
        st.info(
            f"Account matched **DBSCAN cluster {result['cluster_id']}** "
            f"(high-confidence coordinated behavior)."
        )
    else:
        st.info("Account classified using RandomForest (general behavior).")
