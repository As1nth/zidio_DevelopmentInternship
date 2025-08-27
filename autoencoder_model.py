import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Scale 'Amount'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop 'Time' column
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

# Separate fraud and normal data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train only on normal transactions (Class = 0)
X_train_normal = X_train[y_train == 0]

# Build Autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(14, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
encoded = Dense(7, activation="relu")(encoded)
decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal,
                          epochs=10,
                          batch_size=64,
                          shuffle=True,
                          validation_split=0.2,
                          verbose=1)

# Predict on test data
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
threshold = np.percentile(mse, 95)

# Convert anomaly score to binary classification
y_pred = [1 if e > threshold else 0 for e in mse]

# Evaluate
print("\n📊 Autoencoder Evaluation:")
print(classification_report(y_test, y_pred))

# Optional: Save model
autoencoder.save("output/autoencoder_model.h5")

# Optional: Save threshold
np.save("output/autoencoder_threshold.npy", threshold)
