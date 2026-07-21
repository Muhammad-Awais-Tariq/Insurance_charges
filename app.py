import pandas as pd
import streamlit as st
import joblib
import numpy as np

model = joblib.load("tree_pipeline.joblib")