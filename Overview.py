import streamlit as st;

st.title("Retail Agentic AI System")

st.markdown("""
A data-driven decision support system designed to optimize retail operations through intelligent pricing, customer segmentation, and demand forecasting.

---

### Problem Context

Retail decision-making involves multiple interdependent factors such as pricing strategy, customer behavior, and demand variability.  
Traditional approaches often rely on static rules or manual judgment, leading to inefficiencies and missed optimization opportunities.

This system addresses these challenges by combining machine learning models with business logic to generate actionable insights.

---

### System Capabilities

**Pricing Intelligence**
- Dynamic price recommendations based on market conditions  
- Competitive positioning analysis  
- Margin-aware decision logic  

**Customer Segmentation**
- RFM-based clustering of customers  
- Identification of behavioral segments  
- Strategy recommendations for each segment  

**Demand Forecasting**
- Time-series demand prediction using Prophet  
- Detection of seasonal trends and demand spikes  
- Inventory planning support  

---

### System Workflow

Input Data → Machine Learning Models → Business Rules → Actionable Recommendations

---

### Technology Stack

Python · Scikit-learn · XGBoost · Prophet · Streamlit · Plotly

---

Use the sidebar to navigate between modules.
""")