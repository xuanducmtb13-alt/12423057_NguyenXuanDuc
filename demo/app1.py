import streamlit as st
import pandas as pd
import joblib

# ===== Load =====
model = joblib.load("rf_exam_score_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

st.title("🎓 Exam Score Prediction")

with st.form("form"):
    StudyHours = st.slider("Study Hours", 0, 24, 6)
    Attendance = st.slider("Attendance (%)", 0, 100, 70)
    Resources = st.selectbox("Resources", [0,1,2])
    Extracurricular = st.selectbox("Extracurricular", [0,1])
    Motivation = st.selectbox("Motivation", [0,1])
    Internet = st.selectbox("Internet", [0,1])
    Gender = st.selectbox("Gender", [0,1])
    Age = st.slider("Age", 15, 30, 20)
    LearningStyle = st.selectbox("Learning Style", [0,1,2,3])
    OnlineCourses = st.slider("Online Courses", 0, 20, 5)
    Discussions = st.selectbox("Discussions", [0,1])
    AssignmentCompletion = st.slider("Assignment Completion (%)", 0, 100, 60)
    EduTech = st.selectbox("EduTech", [0,1])
    StressLevel = st.selectbox("Stress Level", [0,1,2,3])
    FinalGrade = st.selectbox("Final Grade Level", [0,1,2,3])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "StudyHours": StudyHours,
        "Attendance": Attendance,
        "Resources": Resources,
        "Extracurricular": Extracurricular,
        "Motivation": Motivation,
        "Internet": Internet,
        "Gender": Gender,
        "Age": Age,
        "LearningStyle": LearningStyle,
        "OnlineCourses": OnlineCourses,
        "Discussions": Discussions,
        "AssignmentCompletion": AssignmentCompletion,
        "EduTech": EduTech,
        "StressLevel": StressLevel,
        "FinalGrade": FinalGrade
    }])

    # đảm bảo đúng thứ tự feature
    input_df = input_df[feature_cols]

    score = round(float(model.predict(input_df)[0]), 2)
    st.success(f"📊 Exam Score dự đoán: **{score}**")
