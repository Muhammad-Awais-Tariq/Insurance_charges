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

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.30 , random_state=42)

num_columns = ["age", "bmi", "children"]
ohe_column = ["region"]
ordinal_columns = ["sex", "smoker"]

preprocessing = ColumnTransformer([
    ("num", "passthrough", num_columns),
    ("ohe", OneHotEncoder(drop='first'), ohe_column),
    ("ordinal", OrdinalEncoder(categories=[['female','male'], ['no','yes']]), ordinal_columns),
])

