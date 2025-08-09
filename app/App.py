# Frontend code for the Streamlit app
import streamlit as st


st.set_page_config(
    page_title="Bikes Availability Predictor",
    page_icon="🚲",
)

st.title("Predict Bikes Availability in Washington DC")

st.write("""
    This app predicts the availability of bikes in Washington DC using a machine learning 
    model trained on historical weather data and bike usage patterns.
""")
