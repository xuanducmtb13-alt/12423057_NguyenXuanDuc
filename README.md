# 💻💻 STUDENT EXAM SCORE PREDICTION BASED ON LEARNING HABITS
Student Exam Score Prediction using Machine Learning

## 1️. Project Introduction

### 1.1 Problem Statement
In education, evaluating student performance is crucial to guide and improve learning outcomes. However, accurately predicting exam scores based on students’ learning habits and behaviors is challenging.

This project focuses on building a Machine Learning system to:

Analyze student learning habit data

Predict ExamScore based on behavioral and study habit features

Support teachers and schools in making personalized teaching and learning decisions

### 1.2 Project Objectives
Understand and analyze student learning habit dataset

Perform systematic data preprocessing

Build and compare multiple Machine Learning models

Evaluate models using metrics suitable for regression tasks

Implement a training and prediction (inference) pipeline

## 2️. Dataset Overview (Student Performance Dataset) 
### 2.1 Data Source
Link : https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style

The dataset is collected from student surveys and records, including information on learning habits, extracurricular activities, personal characteristics, and final exam scores.

⚠️ For student privacy reasons, the dataset is not included on GitHub. Instructions for downloading and using the dataset are provided in data/README.md.

### 2.2 Feature Description
Continuous Numerical Features:

StudyHours: Number of study hours per week

Attendance: Class attendance percentage (%)

AssignmentCompletion: Number of completed assignments

ExamScore: Final exam score (target)

EduTech: Educational technology usage score

StressLevel: Stress level (0–100)

FinalGrade: Final grade

Binary Features:

Resources: 1 = Sufficient learning resources, 0 = Not enough

Extracurricular: 1 = Participates in extracurricular activities, 0 = No

Motivation: 1 = High motivation, 0 = Low motivation

Internet: 1 = Has internet access, 0 = No

Gender: 1 = Male, 0 = Female

OnlineCourses: 1 = Attends online courses, 0 = No

Discussions: 1 = Participates in discussions, 0 = No

Categorical Features:

Age: Student age

LearningStyle: 1 = Visual, 2 = Auditory, 3 = Kinesthetic

### 2.3 General Observations
The dataset contains n rows and m columns, including both numerical and categorical features.

Some columns may contain outliers, e.g.:

StudyHours max = 19, median ~19 → requires checking

Attendance ranges from 0–100%, may have invalid values

The target ExamScore is a continuous variable, making this a regression problem.

Some binary features may be imbalanced, which should be considered during model training.

Categorical Features: LearningStyle, Gender → represent students’ learning style and gender

# 3️. Data Preprocessing
Data preprocessing aims to:

Ensure clean and consistent data

Improve model learning performance

Reduce noise and bias

## 3.1 Handling Missing Values
Check for null values in the dataset

Continuous features are imputed using median to reduce outlier influence

## 3.2 Categorical Feature Encoding
Binary features: keep as 0/1

Non-ordinal categorical features (LearningStyle) → One-Hot Encoding

## 3.3 Numerical Feature Scaling
Continuous features (StudyHours, Attendance, AssignmentCompletion, ExamScore, EduTech, StressLevel, FinalGrade) are standardized (mean=0, std=1)

Helps models converge faster and avoid bias from large values

# 3.4 Outlier Handling
Identify extreme values, e.g.: StudyHours > 24, Attendance > 100%

Handle via capping or log-transform if necessary

# 4️. Training & Prediction Pipeline
Unified pipeline workflow:

mathematica
Sao chép mã
Raw Data → Preprocessing → Train/Test Split → Model Training → Evaluation → Save Model → Inference
Built using scikit-learn Pipeline and ColumnTransformer

Ensures preprocessing is learned from training data only, then applied to validation/test

Prevents data leakage

Allows easy model saving and reuse for new data inference

# 5️. Machine Learning Models Used
Linear Regression: simple, interpretable

Decision Tree Regressor: nonlinear, intuitive

Random Forest Regressor: ensemble, reduces overfitting

The best-performing model will be saved and used for inference.

# 6️. Model Evaluation Metrics
MAE (Mean Absolute Error): average absolute error

MSE (Mean Squared Error): average squared error

R² Score: coefficient of determination

MAE and R² are especially important for assessing prediction accuracy.

# 7️. Installation & Usage
## 7.1 Install Environment
bash
Sao chép mã
pip install -r requirements.txt
## 7.2 Train Model
bash
Sao chép mã
python app/train.py
Trained model saved at models/rf_pipeline.pkl

## 7.3 Predict New Data
bash
Sao chép mã
python app/predict.py
## 7.4 Run Demo
bash
Sao chép mã
python demo/app.py
# 8️. Project Directory Structure
kotlin
```
StudentScorePrediction/
├── app/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/
│   └── rf_pipeline.pkl
├── data/
│   └── student_performance.csv
├── demo/
├── reports/
├── slides/
├── README.md
├── requirements.txt
└── .gitignore
```
# 9️. Author
Name: [Nguyễn Xuân Đức]

Student ID: [12423057]

Class: [124231]


