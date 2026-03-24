import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2563eb; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load(os.path.join("models", "kmeans.pkl"))
        scaler = joblib.load(os.path.join("models", "rfm_scaler.pkl"))
        return kmeans, scaler
    except:
        return None, None

kmeans, scaler = load_models()

st.title("👥 Customer Segmentation Module")
st.caption("Agentic Pattern Recognition | RFM Clustering")

if kmeans and scaler:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Optimal Clusters (k)", int(kmeans.n_clusters))
    with m2:
        st.metric("Model Inertia", f"{kmeans.inertia_:,.2f}")
    with m3:
        st.metric("Scaling Engine", "RobustScaler")

    st.divider()

    col_input, col_result = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Simulated Customer (RFM)")
        recency = st.number_input("Recency (Days since last purchase)", min_value=1, value=30)
        frequency = st.number_input("Frequency (Total orders)", min_value=1, value=5)
        monetary = st.number_input("Monetary (Total spent ₹)", min_value=1.0, value=1200.0)
        
        predict_btn = st.button("🔍 Execute Segmentation Analysis")

    with col_result:
        if predict_btn:
            raw_input = np.array([[recency, frequency, monetary]])
            log_input = np.log1p(raw_input)
            scaled_input = scaler.transform(log_input)
            
            cluster_id = kmeans.predict(scaled_input)[0]
            
            # --- THE FIX: COMPOSITE SCORING ---
            centers = kmeans.cluster_centers_
            # Score = Monetary + Frequency - Recency (Higher is better)
            composite_scores = centers[:, 2] + centers[:, 1] - centers[:, 0]
            
            sorted_clusters = np.argsort(composite_scores)
            
            label_map = {
                sorted_clusters[3]: ("Core Actives", "Enroll in VIP Loyalty Tier.", "#10b981"),
                sorted_clusters[2]: ("Consistent Contributors", "Send Volume-Based Upsell.", "#3b82f6"),
                sorted_clusters[1]: ("Lapsing High-Potential", "Trigger High-Value Win-Back Discount.", "#f59e0b"),
                sorted_clusters[0]: ("Dormant / Low-Yield", "Move to Low-Cost Email Cadence.", "#ef4444")
            }
            
            segment_name, recommendation, color = label_map[cluster_id]
            
            st.subheader("Agent Output")
            st.markdown(f"""
                <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 20px;">
                    <h3 style="color: {color}; margin-top: 0;">{segment_name}</h3>
                    <p style="font-size: 1.2em; margin-bottom: 0;"><b>Agent Recommendation:</b> {recommendation}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Feature Space Topography")
            
            fig = go.Figure()
            
            for cid in range(kmeans.n_clusters):
                name, _, c_color = label_map[cid]
                fig.add_trace(go.Scatter3d(
                    x=[centers[cid, 0]], y=[centers[cid, 1]], z=[centers[cid, 2]],
                    mode='markers+text',
                    marker=dict(size=15, color=c_color, opacity=0.6),
                    name=f"Center: {name}",
                    text=[name],
                    textposition="top center"
                ))
            
            fig.add_trace(go.Scatter3d(
                x=[scaled_input[0][0]], y=[scaled_input[0][1]], z=[scaled_input[0][2]],
                mode='markers',
                marker=dict(size=10, color='white', symbol='diamond'),
                name="Current Customer"
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="Recency (Scaled)",
                    yaxis_title="Frequency (Scaled)",
                    zaxis_title="Monetary (Scaled)",
                    bgcolor="#0e1117"
                ),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Intelligence Core missing. Please ensure `kmeans.pkl` and `rfm_scaler.pkl` are in the `models/` folder.")

st.sidebar.divider()
st.sidebar.info("Dynamic Unsupervised Learning Active")
st.sidebar.markdown("**SIT Pune | Group 18**")