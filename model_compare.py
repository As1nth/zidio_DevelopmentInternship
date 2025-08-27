import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Scale 'Amount'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop 'Time' if exists
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

# Split data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("📘 Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))
joblib.dump(log_model, "output/log_model.pkl")

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("📗 XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
joblib.dump(xgb_model, "output/xgb_model.pkl")
