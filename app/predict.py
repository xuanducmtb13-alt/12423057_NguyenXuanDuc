import pandas as pd
import joblib

def predict(new_data):
    # Load model & features
    model = joblib.load("../models/rf_exam_score_model.joblib")
    feature_cols = joblib.load("../models/feature_columns.joblib")
    
    # Chuyển new_data sang DataFrame
    df = pd.DataFrame([new_data])
    df = df[feature_cols]  # đảm bảo thứ tự cột

    prediction = model.predict(df)
    return prediction[0]

# Ví dụ dùng
if __name__ == "__main__":
    sample_input = {
        "StudyHours": 10, "Attendance": 0.9, "Resources": 0.8,
        "Motivation": 0.7, "Age": 17, "OnlineCourses": 1,
        "AssignmentCompletion": 1, "Extracurricular": 1,
        "Internet": 1, "Gender": 0, "Discussions": 1, "EduTech": 1,
        # Các cột one-hot cho categorical nếu có
        # ví dụ "LearningStyle_Visual": 1, "StressLevel_High": 0, ...
    }
    score = predict(sample_input)
    print("Predicted ExamScore:", score)
