import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import joblib

def run_unified_training():
    """
    Trains the unified XGBoost model using the enriched dataset that includes
    the high_alert_volume feature from the AEWS.
    """
    print("--- Unified Model Training & Evaluation Started (with AEWS data) ---")

    try:
        # Load the new, enriched dataset created by merge_data.py
        df = pd.read_csv('datasets/unified_training_data.csv')
    except FileNotFoundError:
        print("\n[ERROR] The file 'unified_training_data.csv' was not found.")
        print("Please run the 'merge_data.py' script first to generate the enriched dataset.")
        return

    # One-hot encode the 'State' column to be used as a feature
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)

    target = 'Cases Reported'

    # The 'high_alert_volume' column is now automatically included as a predictive feature
    X = df_encoded.drop(columns=['Unique Code', target])
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost Regressor model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model's performance on the test set
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    print(f"\nModel Performance with AEWS data:")
    print(f"R-squared (R²) Score: {score:.4f}")

    # Save both the trained model and the list of feature columns together
    model_filename = 'unified_model_data.joblib'
    joblib.dump((model, X.columns), model_filename)
    print(f"\nModel and columns successfully saved as '{model_filename}'")

if __name__ == '__main__':
    run_unified_training()