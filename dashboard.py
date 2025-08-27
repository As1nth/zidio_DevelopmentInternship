import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_and_preprocess
from alert import send_alert

# ---------------------------
# 🔐 USER AUTHENTICATION
# ---------------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "pass123",
    "bhanu": "mysecretpass",
    "guest": "test123"
}

st.set_page_config(page_title="Login - Fraud Detection", layout="centered")
st.title("🔐 Login to Access the Fraud Detection Dashboard")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.success("✅ Login successful!")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
    st.stop()

# ---------------------------
# ✅ MAIN DASHBOARD
# ---------------------------

st.title("💳 Financial Fraud Detection Dashboard")

# Load and train model
with st.spinner("Training model..."):
    X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

st.success("Model is ready! Enter transaction details below 👇")

# --------- ROC Curve & Confusion Matrix ----------
# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Show ROC Curve
st.subheader("📈 ROC Curve")
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Show Confusion Matrix
st.subheader("📊 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# --------- Input form ----------
amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=0.01)

v_features = []
for i in range(1, 29):
    v = st.slider(f"V{i}", -10.0, 10.0, 0.0)
    v_features.append(v)

# Predict single input
if st.button(" Check for Fraud"):
    input_data = pd.DataFrame([[amount] + v_features], columns=X_train.columns)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected!")
        send_alert(transaction_id=9999, amount=amount, to_email="sreebhanu1221@gmail.com")  # Replace with real email
    else:
        st.success("✅ Transaction is Safe.") 