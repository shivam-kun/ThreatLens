<div align="center">
  <h1>🛡️ ThreatLens</h1>
  <p><strong>Advanced Cybercrime Forecasting & Intelligence Platform</strong></p>
  <p>
    An end-to-end framework integrating machine learning models, time-series forecasting, and an active early warning system to predict and analyze cybercrime trends.
  </p>
</div>

---

## 📖 Overview

ThreatLens is a comprehensive platform designed to forecast cybercrime rates, track emerging threats, and analyze historical distribution across different geographical regions. By combining demographic records, technology adoption metrics, and real-time threat intelligence scraping, the platform offers an interactive web dashboard for visualizing and predicting the evolving landscape of digital crime.

## ✨ Core Features

- **Interactive Web Dashboard**: A full-featured Flask web application providing interactive tools for spatial and temporal data analysis, utilizing Plotly for dynamic charts and visualizations.
- **Ensemble Predictive Modeling**: Forecasts future state-level cybercrime rates using a unified pre-trained machine learning model (`joblib`), taking into account internet penetration, broadband adoption, and population metrics.
- **Deep Learning Time-Series**: Employs an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras to conduct accurate, successive multi-year forecasting based on historical sequences.
- **AI Early Warning System (AEWS)**: An automated background monitor (`aews_monitor.py`) leveraging Selenium Stealth to actively scrape threat intelligence from security blogs (e.g., BleepingComputer), scoring threats and storing alerts in a local SQLite database.
- **Comparative Analysis**: Tools to dissect specific crime categories (identity theft, online banking fraud, cyberstalking) and compare trends across states year-over-year.

## 🛠️ Technology Stack

- **Backend & API**: Python 3, Flask
- **Machine Learning & AI**: scikit-learn, TensorFlow/Keras (LSTM), XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Google Charts
- **Scraping & Intelligence**: Selenium, webdriver-manager, SQLite3
- **Deployment**: Configured for serverless deployment on Vercel (`vercel.json`)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The project runs in a standard virtual environment. Note that running the AEWS monitor requires Google Chrome to be installed.

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivam-kun/ThreatLens.git
   cd ThreatLens
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Web Application:**
   ```bash
   python main.py
   ```
   *The application will now be available at `http://127.0.0.1:5000`.*

### Running the Early Warning System (Optional)

To start the continuous background threat monitor, open a separate terminal in the project directory and run:

```bash
python aews_monitor.py
```

## ☁️ Deployment

This project includes a `vercel.json` configuration file, making it ready for instant, serverless deployment on Vercel. 

1. Push your code to a GitHub repository.
2. Import the repository into your Vercel account.
3. **Important**: Set the **Root Directory** to `ThreatLens` within Vercel's project settings to ensure models and datasets are found correctly.
4. Deploy the application.

## 📊 Repository Structure

```
ThreatLens/
├── aews_monitor.py            # Headless threat intelligence scraper
├── main.py                    # Main Flask web application server
├── datasets/                  # Historical and forecast CSV data
├── models/                    # Model architecture scripts
├── static/ & templates/       # Web styling, frontend JS, and HTML views
├── lstm_model.h5              # Pre-trained deep learning time-series model
├── unified_model_data.joblib  # Pre-trained ensemble machine learning model
├── requirements.txt           # Python package dependencies
└── vercel.json                # Vercel serverless configuration
```

## 📜 License

This project is open-source and available under the terms of the MIT License.

---

<div align="center">
  <b>Developed with ❤️ for cybersecurity intelligence.</b>
</div>