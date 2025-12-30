# app/predict.py
import pandas as pd
from preprocess import preprocess
from utils import load_model, get_metrics

# Load data mới để predict
df_new = pd.read_csv('app/data/student_performance_new.csv')
df_new, numeric_cols, scaler = preprocess(df_new)

X_new = df_new.drop("ExamScore", axis=1, errors='ignore')

# Load model
lr_model = load_model('models/lr_model.pkl')

# Predict
y_pred = lr_model.predict(X_new)
df_new["Predicted_ExamScore"] = y_pred
print(df_new.head())
