import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Smart Grid Dashboard", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open("smart_grid_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Real-time Analysis", "Data Visualization", "Raw Data"])

# --- Page 1: Real-time Analysis (Your Original Logic) ---
if page == "Real-time Analysis":
    st.title("⚡ Real-time Stability Prediction")
    st.write("Adjust grid parameters to see live stability predictions.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Reaction Times")
            tau1 = st.number_input("tau1 (Supplier)", 0.0, 10.0, 0.5)
            tau2 = st.number_input("tau2 (C1)", 0.0, 10.0, 6.7)
            tau3 = st.number_input("tau3 (C2)", 0.0, 10.0, 3.4)
            tau4 = st.number_input("tau4 (C3)", 0.0, 10.0, 8.5)
        with col2:
            st.markdown("### Power Balance")
            p1 = st.number_input("p1 (Produced)", -5.0, 5.0, 3.5)
            p2 = st.number_input("p2 (Cons 1)", -5.0, 5.0, -1.2)
            p3 = st.number_input("p3 (Cons 2)", -5.0, 5.0, -1.5)
            p4 = st.number_input("p4 (Cons 3)", -5.0, 5.0, -0.8)
        with col3:
            st.markdown("### Elasticity")
            g1 = st.number_input("g1 coefficient", 0.0, 1.0, 0.1)
            g2 = st.number_input("g2 coefficient", 0.0, 1.0, 0.5)
            g3 = st.number_input("g3 coefficient", 0.0, 1.0, 0.3)
            g4 = st.number_input("g4 coefficient", 0.0, 1.0, 0.8)
        
        submit = st.form_submit_button("Analyze Stability")

    if submit:
        if model:
            features = np.array([[tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]])
            prediction = model.predict(features)
            prob = model.predict_proba(features)
            
            if prediction[0] == 0:
                st.error(f"⚠️ **Grid is Unstable** ({np.max(prob)*100:.1f}% confidence)")
            else:
                st.success(f"✅ **Grid is Stable** ({np.max(prob)*100:.1f}% confidence)")
        else:
            st.warning("Model not loaded. Check 'smart_grid_model.pkl'.")

# --- Page 2: Data Visualization ---
elif page == "Data Visualization":
    st.title("📊 Grid Parameter Visualization")
    
    # Sample data for visualization (Replace with your actual training DF if available)
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Reaction Time', 'Power Load', 'Price Elasticity']
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Stability Factor Distribution")
        fig = px.histogram(chart_data, x="Reaction Time", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig, use_container_width=True)
        
    with col_b:
        st.subheader("Power vs Elasticity Correlation")
        fig2 = px.scatter(chart_data, x="Power Load", y="Price Elasticity")
        st.plotly_chart(fig2, use_container_width=True)

# --- Page 3: Raw Data ---
elif page == "Raw Data":
    st.title("📋 Raw System Logs")
    st.write("This table shows the recent input logs and captured grid telemetry.")
    
    # Creating a dummy dataframe to represent what raw data would look like
    raw_df = pd.DataFrame({
        'Timestamp': pd.date_range("2023-01-01", periods=10, freq="H"),
        'Node_ID': [f"Node_{i}" for i in range(10)],
        'Voltage_KV': np.random.uniform(220, 240, 10),
        'Frequency_Hz': np.random.uniform(49.5, 50.5, 10),
        'Status': np.random.choice(['Stable', 'Unstable'], 10)
    })
    
    st.dataframe(raw_df, use_container_width=True)
    st.download_button("Download CSV", raw_df.to_csv().encode('utf-8'), "grid_data.csv", "text/csv")

st.sidebar.info("Model: Random Forest / Bagging Classifier")