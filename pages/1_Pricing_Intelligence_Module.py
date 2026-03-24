import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Price Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2563eb; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ── Embedded label map — works from pkl alone, no session state needed ─────────
EMBEDDED_LABEL_MAP = {0: 'decrease', 1: 'discount', 2: 'hold', 3: 'increase'}

# ── Base price adjustment per action ──────────────────────────────────────────
BASE_ADJ = {
    'increase': +0.10,
    'decrease': -0.08,
    'discount': -0.15,
    'hold'    :  0.00,
}

ACTION_COLORS = {
    'increase': '#10b981',
    'decrease': '#ef4444',
    'discount': '#f59e0b',
    'hold'    : '#6b7280',
}

ACTION_ICONS = {
    'increase': '📈',
    'decrease': '📉',
    'discount': '🏷️',
    'hold'    : '⏸️',
}

ACTION_DESCRIPTION = {
    'increase': 'Proven demand with headroom — capture margin.',
    'decrease': 'Known quality issue or significantly overpriced — reduce to stay competitive.',
    'discount': 'Good product, low market exposure — discount to build volume.',
    'hold'    : 'Insufficient signal or mid-range rating — maintain current price.',
}

@st.cache_resource
def load_pricing_assets():
    for base in [os.path.join("models"), ""]:
        try:
            model  = joblib.load(os.path.join(base, "pricing_model.pkl"))
            scaler = joblib.load(os.path.join(base, "pricing_scaler.pkl"))
            return model, scaler
        except Exception:
            continue
    return None, None

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏷️ Price Intelligence Module")
st.caption("Agentic Supply Chain | XGBoost Dynamic Pricing — 4-Action Engine")

pricing_model, scaler = load_pricing_assets()

col_params, col_metrics = st.columns([1, 2], gap="large")

with col_params:
    st.subheader("Product Parameters")

    rating = st.slider(
        "Product Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1,
        help="Customer satisfaction score (1–5)"
    )
    rating_count = st.number_input(
        "Rating Count (Volume)", min_value=0, value=120, step=10,
        help="Number of customer reviews — proxy for market exposure"
    )
    current_price = st.number_input(
        "Current Price (£)", min_value=0.01, value=45.0, step=0.5,
        help="Your current selling price"
    )
    competitor_price = st.number_input(
        "Competitor Price (£)", min_value=0.01, value=50.0, step=0.5,
        help="Competitor's price for the same or equivalent product"
    )

    # Derived ratio — show it live for transparency
    price_comp_ratio = round(current_price / (competitor_price + 0.01), 3)
    ratio_color = "#10b981" if price_comp_ratio <= 1.0 else "#ef4444"
    st.markdown(
        f"<small style='color:{ratio_color}'>📊 Price Ratio (Our / Competitor): <b>{price_comp_ratio:.3f}</b> "
        f"{'— we are cheaper ✅' if price_comp_ratio < 1.0 else '— we are pricier ⚠️' if price_comp_ratio > 1.0 else '— at parity'}</small>",
        unsafe_allow_html=True
    )

    st.divider()
    predict_btn = st.button("🧠 Generate Pricing Strategy")

# ── Inference ─────────────────────────────────────────────────────────────────
with col_metrics:
    if predict_btn:
        if pricing_model and scaler:
            with st.spinner("Intelligence Core computing pricing strategy..."):

                input_features = np.array([[rating, rating_count, price_comp_ratio]])
                scaled_input   = scaler.transform(input_features)

                action_id   = int(pricing_model.predict(scaled_input)[0])
                probas      = pricing_model.predict_proba(scaled_input)[0]
                confidence  = float(probas.max())
                action      = EMBEDDED_LABEL_MAP.get(action_id, "hold")

                adj_pct         = round(BASE_ADJ[action] * confidence, 4)
                recommended_price = round(current_price * (1 + adj_pct), 2)
                recommended_price = max(recommended_price, current_price * 0.70)
                recommended_price = min(recommended_price, competitor_price * 1.20)
                recommended_price = round(recommended_price, 2)

                price_delta = recommended_price - current_price
                color       = ACTION_COLORS[action]
                icon        = ACTION_ICONS[action]

            # ── Agent output card ─────────────────────────────────────────
            st.subheader("Agent Output")
            st.markdown(f"""
                <div style="background-color:#1f2937; padding:28px; border-radius:10px;
                            border-left:8px solid {color}; margin-bottom:20px;">
                    <h2 style="color:{color}; margin-top:0; text-transform:uppercase;
                               font-size:2.2em; letter-spacing:2px;">
                        {icon} {action}
                    </h2>
                    <p style="font-size:1.1em; margin-bottom:0; color:#d1d5db;">
                        <b>Agent Recommendation:</b> {ACTION_DESCRIPTION[action]}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # ── Price recommendation metrics ──────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Current Price", f"£{current_price:.2f}")
            with m2:
                st.metric("Competitor Price", f"£{competitor_price:.2f}")
            with m3:
                st.metric(
                    "Recommended Price", f"£{recommended_price:.2f}",
                    delta=f"£{price_delta:+.2f}"
                )
            with m4:
                st.metric("Model Confidence", f"{confidence:.1%}")

            st.divider() 
            # ── Model internals expander ──────────────────────────────────
            with st.expander("📊 View Model Internals"):
                st.markdown(f"""
**Model Architecture:** XGBoost Classifier (4-class: decrease / discount / hold / increase)  
**Feature Scaling:** RobustScaler  
**Input Features:** Rating ({rating}) · Rating Count ({rating_count}) · Price Ratio ({price_comp_ratio})  
**Scaled Input:** `{np.round(scaled_input[0], 4).tolist()}`  
**Raw Predicted Class ID:** `{action_id}` → `{action}`  
**Adjustment:** `{adj_pct * 100:+.2f}%` (Base {BASE_ADJ[action]*100:+.0f}% × Confidence {confidence:.1%})  
**Guardrails:** Min £{current_price * 0.70:.2f} (70% current) · Max £{competitor_price * 1.20:.2f} (120% competitor)
                """)

        else:
            st.error(
                "Intelligence Core missing. Ensure `pricing_model.pkl` and `pricing_scaler.pkl` "
                "are in the `models/` directory, and that `xgboost` is installed."
            )
    else:
        # Placeholder before prediction
        st.info("👈 Set product parameters and click **Generate Pricing Strategy** to run the agent.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.info("Price Intelligence Active")
st.sidebar.markdown("**Actions:** INCREASE · DECREASE · DISCOUNT · HOLD")
st.sidebar.markdown("**SIT Pune | Group 18**")