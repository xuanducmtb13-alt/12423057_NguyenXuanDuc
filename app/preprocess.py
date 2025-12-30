# app/preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path='app/data/student_performance.csv'):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    numeric_cols = [
        "StudyHours", "Attendance", "Resources", "Motivation", "Age",
        "OnlineCourses", "AssignmentCompletion", "ExamScore"
    ]
    multi_cat_cols = ["LearningStyle", "StressLevel", "FinalGrade"]

    # Ép kiểu category
    for col in multi_cat_cols:
        df[col] = df[col].astype('category')

    # One-hot encode
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Scale numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, numeric_cols, scaler
