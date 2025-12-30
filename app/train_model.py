import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import load_data, preprocess_data

# Load & preprocess
df = load_data("../data/student_performance.csv")
df, numeric_cols = preprocess_data(df)

# X - y
X = df.drop("ExamScore", axis=1)
y = df["ExamScore"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate
print("TRAIN MAE:", mean_absolute_error(y_train, rf_model.predict(X_train)))
print("TEST  MAE:", mean_absolute_error(y_test, rf_model.predict(X_test)))
print("TEST  R2 :", r2_score(y_test, rf_model.predict(X_test)))

# Save model
joblib.dump(rf_model, "../models/rf_exam_score_model.joblib")
joblib.dump(X.columns.tolist(), "../models/feature_columns.joblib")
print("Saved model & feature columns")
