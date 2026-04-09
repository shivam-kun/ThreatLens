import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from joblib import dump

def create_sequences(data, n_steps):
    """Helper function to convert time series data into supervised learning sequences."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_lstm_training():
    print("--- LSTM Model Training Started ---")

    # 1. Load and Prepare Data for a single state (e.g., Maharashtra)
    try:
        df = pd.read_csv('datasets/forecast.csv')
        state_data = df[df['State'] == 'Maharashtra']['Cases Reported'].values
        print("Dataset for Maharashtra loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Scale the data
    # LSTMs work best when data is scaled between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(state_data.reshape(-1, 1))

    # 3. Create Sequences
    # We'll use the last 3 years of data to predict the next year
    n_steps = 3
    X, y = create_sequences(data_scaled, n_steps)

    # Reshape for LSTM input [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # 4. Build the LSTM Model
    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 5. Train the Model
    # 'epochs' is the number of times the model sees the entire dataset
    print("Training the model...")
    model.fit(X, y, epochs=200, verbose=0) # verbose=0 keeps the output clean
    print("Model training complete.")

    # 6. Evaluate and Show an Example Prediction
    # Let's predict the next value after the last sequence in our data
    last_sequence = data_scaled[-n_steps:].reshape((1, n_steps, n_features))
    predicted_scaled = model.predict(last_sequence)
    # We must inverse_transform the prediction to get the real number back
    predicted_cases = scaler.inverse_transform(predicted_scaled)

    print("\n--- Example Forecast ---")
    print(f"Based on the last {n_steps} years of data, the model forecasts the next year's cases will be: {int(predicted_cases[0][0])}")
    print("------------------------")

    # 7. Save the Model and the Scaler
    # We need to save the scaler to properly handle future inputs
    model.save('lstm_model.h5') # Keras models are saved in .h5 format
    dump(scaler, 'lstm_scaler.joblib')
    print("\nLSTM model and scaler successfully saved.")
    print("--- Script Finished ---")


if __name__ == '__main__':
    run_lstm_training()

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        # seq_x is a sequence of [n_steps] observations, each with [n_features]
        # seq_y is the 'Cases Reported' value from the next time step
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, 0] # target is the first column
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_lstm_training():
    print("--- LSTM Model Training Started (Multivariate) ---")

    # 1. Load and Prepare Multivariate Data
    try:
        # CHANGED: Load the new multivariate dataset
        df = pd.read_csv('datasets/maharashtra_multivariate.csv')
        # CHANGED: Select both feature columns. IMPORTANT: Target ('Cases Reported') must be the first column
        features = df[['Cases Reported', 'high_alert_volume']].values
        print("Multivariate dataset for Maharashtra loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Scale the data
    # The scaler will now learn to scale both columns independently
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(features)

    # 3. Create Sequences
    n_steps = 3
    X, y = create_sequences(data_scaled, n_steps)

    # 4. Define model parameters
    # CHANGED: The number of features is now 2
    n_features = 2 
    # NOTE: The reshape from the old script is no longer needed,
    # as create_sequences now returns data in the correct 3D shape: [samples, timesteps, features]

    # 5. Build the LSTM Model
    print("Building Multivariate LSTM model...")
    model = Sequential()
    # CHANGED: The input_shape must now reflect the new number of features
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 6. Train the Model
    print("Training the model...")
    model.fit(X, y, epochs=200, verbose=0)
    print("Model training complete.")

    # 7. Example Prediction
    # CHANGED: The last sequence for prediction now contains both features
    last_sequence = data_scaled[-n_steps:].reshape((1, n_steps, n_features))
    predicted_scaled = model.predict(last_sequence)
    
    # To inverse transform, we need to create a dummy array with the same shape as the original data
    dummy_array = np.zeros((1, n_features))
    dummy_array[0, 0] = predicted_scaled[0, 0] # Put our prediction in the first column
    predicted_cases = scaler.inverse_transform(dummy_array)[:, 0] # Inverse transform and get the value

    print("\n--- Example Forecast ---")
    print(f"Based on the last {n_steps} years of data (including AEWS alerts), the model forecasts the next year's cases will be: {int(predicted_cases[0])}")
    print("------------------------")

    # 8. Save the Model and the (new) Scaler
    model.save('lstm_model_multivariate.h5')
    dump(scaler, 'lstm_scaler_multivariate.joblib')
    print("\nMultivariate LSTM model and scaler successfully saved.")
    print("--- Script Finished ---")

if __name__ == '__main__':
    run_lstm_training()