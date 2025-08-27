import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Use a style
sns.set(style="darkgrid")

# Load dataset
df = pd.read_csv('data/creditcard.csv')
print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n🔹 Missing values:")
print(df.isnull().sum())

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud")
plt.savefig("output/class_distribution.png")
plt.show()

# Transaction amount distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("output/amount_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.show()
