import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from alert import send_alert

print("🚀 Starting Real-Time Fraud Detection Simulator...\n", flush=True)

# Load dataset
try:
    df = pd.read_csv("data/creditcard.csv")
except Exception as e:
    print("❌ Failed to load dataset:", e)
    exit()

# Preprocess
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

X = df.drop('Class', axis=1)
y = df['Class']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained. Streaming transactions...\n", flush=True)

# Force some fraud simulation for testing
count = 0
for idx, row in X_test.iterrows():
    time.sleep(2)

    input_data = row.values.reshape(1, -1)
    prediction = model.predict(input_data)

    # ✅ Force fraud every 5th transaction (FOR TESTING ONLY)
    simulate_fraud = (count % 5 == 0)

    if prediction[0] == 1 or simulate_fraud:
        print(f"⚠️ Simulated FRAUD at transaction #{idx}!", flush=True)
        send_alert(transaction_id=idx, amount=row['Amount'], to_email="sreebhanu1221@gmail.com")  # Replace
    else:
        print(f"✅ Transaction #{idx} is safe.", flush=True)

    count += 1
    if count >= 20:
        print("\n✅ Simulation completed (20 transactions tested).\n", flush=True)
        break
