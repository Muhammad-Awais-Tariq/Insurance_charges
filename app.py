import pandas as pd
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰", layout="centered")

st.title("Insurance Charge Predictor")
st.caption("Estimate annual medical insurance charges from personal & lifestyle details")

@st.cache_resource
def load_model():
    return joblib.load("tree_pipeline.joblib")

model = load_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=30.66, step=0.1, format="%.2f")
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

    predict_clicked = st.form_submit_button("Predict Charges", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    prediction = model.predict(input_df)[0]

    if prediction < 8000:
        badge = "🟢 Below Average"
    elif prediction < 20000:
        badge = "🟡 Moderate"
    else:
        badge = "🔴 High"

    st.divider()
    st.metric("Estimated Annual Charge", f"${prediction:,.2f}", badge )


    with st.expander("See input summary"):
        st.dataframe(input_df, width=True, hide_index=True)