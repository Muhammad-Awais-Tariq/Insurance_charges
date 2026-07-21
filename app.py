import pandas as pd
import streamlit as st
import joblib
import numpy as np

model = joblib.load("tree_pipeline.joblib")

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("bmi", min_value=0.0, value=30.66)
children = st.number_input("Childern", min_value=0, value=0)
smoker = st.selectbox("Smoke?", ["yes", "no"])
region = st.selectbox("Region: ", ["southeast", "southwest" , "northwest" , "northeast"])
