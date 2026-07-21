import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , FunctionTransformer , OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
import joblib

insurance = pd.read_csv("F://Insurance_charges//Data//insurance.csv")
insurance_df = insurance.copy()

X = insurance_df.drop(columns=["charges"])
y = insurance_df["charges"]

