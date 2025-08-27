import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Scale the 'Amount' column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    # Drop the 'Time' column (optional, not useful)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
# Run this only if this file is executed directly
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")
    
    print("✅ Preprocessing Complete!")
    print("🔹 X_train shape:", X_train.shape)
    print("🔹 X_test shape :", X_test.shape)
    print("🔹 y_train value counts:\n", y_train.value_counts())

