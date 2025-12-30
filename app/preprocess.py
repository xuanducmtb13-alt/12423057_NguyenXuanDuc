import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    numeric_cols = [
        "StudyHours", "Attendance", "Resources", "Motivation", "Age",
        "OnlineCourses", "AssignmentCompletion", "ExamScore"
    ]
    binary_cols = [
        "Extracurricular", "Internet", "Gender", "Discussions", "EduTech"
    ]
    multi_cat_cols = ["LearningStyle", "StressLevel", "FinalGrade"]
    
    # Chuyển sang category
    for col in multi_cat_cols:
        df[col] = df[col].astype('category')

    # One-hot encoding
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Scale numeric
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, numeric_cols
