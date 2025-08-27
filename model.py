from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_preprocess

# Load the preprocessed data
print("🔄 Loading data...")
X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")
print("✅ Data loaded.")

# Create and train the model
print("🔄 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Make predictions
print("🔍 Making predictions...")
y_pred = model.predict(X_test)

import pandas as pd

# Sample prediction (replace with your actual code variables)
# y_test = ground truth, y_pred = model.predict(X_test), y_proba = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "TransactionID": range(len(y_test)),
    "Actual": y_test,
    "Predicted": y_pred,
    "Probability": y_proba
})

# Save the results to CSV
results.to_csv("output/predictions.csv", index=False)
print("✅ Exported to output/predictions.csv")


# Evaluate the model
print("\n📊 Evaluation Results:\n")

print("🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

