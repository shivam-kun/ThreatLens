# generate.py (Final Correct Version)

import pandas as pd
from xgboost import XGBRegressor
from joblib import dump
import os

def generate_state_models():
    """
    Loops through all states and saves a specific XGBoost model for each one
    with the correct filename format (e.g., 'models/Bihar.joblib').
    """
    print("--- Generating state-specific models... ---")

    # Loading DataSet
    df = pd.read_csv('datasets/forecast.csv')
    
    # Ensure the 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory.")

    states = sorted(list(set(df['State'].to_list())))

    for state in states:
        # Preparing Data For a Single State
        one_state = df[df['State'] == state]
        X = one_state.drop(['Unique Code', 'State', 'Cases Reported'], axis=1)
        y = one_state['Cases Reported']

        # Training the model on the full data for that state
        final_model = XGBRegressor()
        final_model.fit(X.values, y.values)

        # Corrected the filename format to match what main.py expects
        path = f'models/{state}.joblib'
        
        dump(final_model, path)
        print(f"Successfully saved model for {state} at '{path}'")

    print("\n--- All state-specific models generated successfully. ---")

if __name__ == '__main__':
    generate_state_models()