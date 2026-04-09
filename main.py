from flask import Flask, render_template, make_response, jsonify, request
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from joblib import dump, load
import subprocess

import numpy as np
import sqlite3
import sys
import os

app = Flask(__name__)

analysis = pd.read_csv('datasets/analysis.csv')
df = pd.read_csv('datasets/cases.csv')
forecast = pd.read_csv('datasets/forecast.csv')

@app.route('/api', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        name = request.get_json()['name']
        row = df[df['State/UT'].str.contains(name, case=False)]
        full_data = row.to_dict(orient='records')[0]
        graph_row = row.drop(['Total Crimes', 'State/UT'], axis=1)
        graph_data = graph_row.to_dict(orient='records')[0]

        graph_arr = [['Year', 'Cases']]
        for key, val in graph_data.items():
            graph_arr.append([key, val])

        return make_response(jsonify({
            'name': full_data['State/UT'],
            'data': full_data,
            'graph': graph_arr
        }))
    else:
        return make_response(jsonify({'error': 'UnAuthorised!'}))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/CA')
def CA():
    return render_template('CA.html')

@app.route('/graph', methods=['POST'])
def graph():
    # get the selected year and type of cybercrime from the form
    year = int(request.form['year'])
    cybercrime = request.form['cybercrime']
    state = request.form['state']
    plot_type = request.form['plot-type']

    error = []
    if plot_type == 'crime-types':
        if not year or not cybercrime:
            error.append('Please select year and cybercrime type.')
            return render_template('graph.html', error=error[0])
        else:
            df_year = analysis[analysis['Year'] == year]
            cyber_crime = df_year.groupby('State')[cybercrime].sum().reset_index()
            fig = px.bar(cyber_crime, x='State', y=cybercrime, title=f"{cybercrime} Cases in {year}")
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)
    elif plot_type == 'state-count':
        if not year or not state:
            error.append('Please select year and state.')
            return render_template('graph.html', error=error[0])
        else:
            data_2017_MH = analysis[(analysis['Year'] == year) & (analysis['State'] == state)]
            df_grouped = data_2017_MH.groupby('State')[
                ['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking', 'ATM',
                 'personation by using computer']].sum().reset_index()
            df_melted = df_grouped.melt(id_vars=['State'],
                                        value_vars=['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking',
                                                    'ATM', 'personation by using computer'], var_name='Crime Type',
                                        value_name='Count')
            fig = px.bar(df_melted, x='State', y='Count', color='Crime Type', barmode='group',
                         title=f'cybercrimes and count in {year} for {state}')
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)

    elif plot_type == 'line-graph':
        df = pd.read_csv('datasets/analysis.csv')
        crime_heads = ['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking', 'ATM',
                       'personation by using computer']
        df = df[['Year'] + crime_heads]
        df = df.groupby('Year').sum().reset_index()
        traces = []
        for crime_head in crime_heads:
            trace = go.Scatter(x=df['Year'], y=df[crime_head], mode='lines', name=crime_head)
            traces.append(trace)
        layout = go.Layout(title='Trend of Cybercrimes over the Years', xaxis_title='Year',
                           yaxis_title='Number of Crimes')
        fig = go.Figure(data=traces, layout=layout)
        div = fig.to_html(full_html=False)
        return render_template('graph.html', plot_div=div)

    elif plot_type == 'pie-graph':
        if not year or not state:
            error.append('Please select state.')
            return render_template('graph.html', error=error[0])
        else:
            data = pd.read_csv('datasets/analysis.csv')
            state = request.form['state']
            maharashtra_data = data[data['State'] == state]
            fig = go.Figure(data=[go.Pie(labels=['Total Internet Subscriptions', 'Total Broadband Subscriptions',
                                                 'Total Wireless internet Subscriptions'],
                                         values=[maharashtra_data['Total Internet Subscriptions'].values[0],
                                                 maharashtra_data['Total Broadband Subscriptions'].values[0],
                                                 maharashtra_data['Total Wireless internet Subscriptions'].values[0]])])

            # set chart title
            fig.update_layout(title=f'Distribution of Internet Subscriptions in {state}')
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)

    else:
        pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def lstm_predict_numpy(X, weights):
    # X shape: (timesteps, features)
    # weights[0]: kernel, weights[1]: recurrent_kernel, weights[2]: bias
    # weights[3]: dense_kernel, weights[4]: dense_bias
    
    W = weights[0]
    U = weights[1]
    b = weights[2]
    W_dense = weights[3]
    b_dense = weights[4]
    
    units = U.shape[0]
    h = np.zeros(units)
    c = np.zeros(units)
    
    # Keras LSTM weight splitting: i, f, c, o
    W_i, W_f, W_c, W_o = np.split(W, 4, axis=1)
    U_i, U_f, U_c, U_o = np.split(U, 4, axis=1)
    b_i, b_f, b_c, b_o = np.split(b, 4)
    
    for x_t in X:
        # gates
        i = sigmoid(np.dot(x_t, W_i) + np.dot(h, U_i) + b_i)
        f = sigmoid(np.dot(x_t, W_f) + np.dot(h, U_f) + b_f)
        o = sigmoid(np.dot(x_t, W_o) + np.dot(h, U_o) + b_o)
        
        # user used activation='relu' in training script
        c_tilde = relu(np.dot(x_t, W_c) + np.dot(h, U_c) + b_c)
        
        c = f * c + i * c_tilde
        h = o * relu(c) # recurrent activation is typically sigmoid, but user's activation=relu applies to state
        
    return np.dot(h, W_dense) + b_dense

def find_max_key(value: int, all_years_predictions: dict) -> str:
    for key, val in all_years_predictions.items():
        if val == value:
            return key

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        try:
            state_info = request.get_json()['state_info']
            state = state_info.get('state')

            if not state:
                return jsonify({'error': 'Please select a state from the dropdown menu.'}), 400

            # Load the unified model and the column names it was trained on
            model, model_columns = load('unified_model_data.joblib')
            
            # Create a DataFrame for the user's input
            input_df = pd.DataFrame(columns=model_columns)
            input_df.loc[0] = 0

            # Fill the DataFrame with user-provided values
            input_df.at[0, 'Year'] = int(state_info.get('year'))
            input_df.at[0, 'Population'] = int(state_info.get('population'))
            input_df.at[0, 'Total Internet Subscriptions'] = int(state_info.get('tis'))
            input_df.at[0, 'Total Broadband Subscriptions'] = int(state_info.get('tbs'))
            input_df.at[0, 'Total Wireless internet Subscriptions'] = int(state_info.get('twis'))
            if 'high_alert_volume' in input_df.columns:
                input_df.at[0, 'high_alert_volume'] = int(state_info.get('aews_volume'))

            state_column = f"State_{state}"
            if state_column in input_df.columns:
                input_df.at[0, state_column] = 1

            # Make the prediction
            prediction = model.predict(input_df)
            
            # --- SIMPLIFIED: Logic for creating the comparison plot ---
            forecast = pd.read_csv('datasets/forecast.csv')
            state_df = forecast[forecast['State'] == state]
            
            # Data for the "Known" bars in the chart (Actual historical data)
            years_known = state_df['Year'].to_list()
            years_data_known = state_df['Cases Reported'].to_list()
            
            # Data for the "Predicted" bar in the chart
            predicted_year = int(state_info.get('year'))
            predicted_cases = int(prediction[0])

            return make_response(jsonify({
                'result': float(prediction[0]),
                'years_known': years_known,
                'years_data_known': years_data_known,
                'predicted_year': predicted_year,
                'predicted_cases': predicted_cases
            }))
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('predictions.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    training_output = None
    if request.method == 'POST':
        # This command runs your training script.
        # It uses 'venv/Scripts/python' which is standard for Windows.
        try:
            process = subprocess.Popen(
                [sys.executable, 'train_models.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace' # This helps prevent encoding errors
            )
            # This part captures the output as it's generated
            training_output = []
            for line in iter(process.stdout.readline, ''):
                training_output.append(line)
            process.stdout.close()
            process.wait()
        except FileNotFoundError:
            training_output = ["Error: python executable not found."]
        except Exception as e:
            training_output = [f"An unexpected error occurred: {str(e)}"]

    return render_template('dashboard.html', training_output=training_output)

@app.route('/lstm_forecast', methods=['GET', 'POST'])
def lstm_forecast():
    prediction_data = None # Start with data as None
    if request.method == 'POST':
        try:
            # All of your existing POST request logic goes here...
            # Load model, scaler, make predictions, etc.
            # Use Numpy instead of TensorFlow/TFLite
            lstm_weights = load('lstm_weights.joblib')
            scaler = load('lstm_scaler.joblib')
            
            years_to_forecast = int(request.form.get('years'))
            
            df = pd.read_csv('datasets/forecast.csv')
            state_data = df[df['State'] == 'Maharashtra']['Cases Reported'].values.reshape(-1, 1)
            data_scaled = scaler.transform(state_data)
            
            n_steps = 3
            last_sequence = list(data_scaled[-n_steps:].flatten())
            
            forecast_scaled = []
            for _ in range(years_to_forecast):
                input_sequence = np.array(last_sequence[-n_steps:]).reshape((n_steps, 1))
                
                # Manual Numpy Inference
                predicted_point = lstm_predict_numpy(input_sequence, lstm_weights)
                
                forecast_scaled.append(predicted_point[0])
                last_sequence.append(predicted_point[0])

            forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
            
            historical_years = df[df['State'] == 'Maharashtra']['Year'].tolist()
            historical_cases = df[df['State'] == 'Maharashtra']['Cases Reported'].tolist()
            
            last_historical_year = historical_years[-1]
            forecast_years = [last_historical_year + i for i in range(1, years_to_forecast + 1)]
            
            prediction_data = {
                'historical_years': historical_years,
                'historical_cases': historical_cases,
                'forecast_years': forecast_years,
                'forecast_cases': forecast.flatten().tolist()
            }

        except Exception as e:
            prediction_data = {'error': str(e)}

    # **CRITICAL CHANGE**: Always pass the variable to the template
    return render_template('lstm_forecast.html', data=prediction_data)

# In main.py

# ... (other routes)

@app.route('/aews')
def aews():
    """Renders the AI Early Warning System dashboard page."""
    return render_template('aews.html')

@app.route('/api/aews_data')
def aews_data():
    """API endpoint to provide the latest threat data from the database."""
    alerts = []
    try:
        # Connect to the database created by aews_monitor.py
        conn = sqlite3.connect('threat_feed.db')
        conn.row_factory = sqlite3.Row # This allows accessing columns by name
        cursor = conn.cursor()
        
        # Fetch the 10 most recent high or medium level alerts
        cursor.execute(
            "SELECT * FROM alerts WHERE threat_level IN ('High', 'Medium') ORDER BY timestamp DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        
        # Convert rows to a list of dictionaries
        for row in rows:
            alerts.append(dict(row))
            
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
        # Return an empty list or an error message if something goes wrong
        return jsonify({"error": "Could not retrieve data from the threat feed."})
        
    return jsonify(alerts)

# ... (rest of the file)


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode)

