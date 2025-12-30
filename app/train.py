# app/train.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess
from utils import get_metrics, save_model

df = load_data()
df, numeric_cols, scaler = preprocess(df)

target = "ExamScore"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
save_model(lr_model, 'models/lr_model.pkl')

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
save_model(dt_model, 'models/dt_model.pkl')

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
save_model(rf_model, 'models/rf_model.pkl')

print("Training completed and models saved!")
