# ThreatLens: Cybercrime Forecasting & Analysis

A machine learning framework for predicting and analyzing cybercrime trends using ensemble methods and multi-source data integration.

---

## Overview

ThreatLens applies ensemble learning techniques to forecast cybercrime rates across different geographical regions and time periods. The project combines historical crime data, demographic information, and internet usage statistics to build predictive models that help understand evolving cyber threat landscapes.

---

## Features

- **Predictive Modeling**: Forecasts cybercrime rates using state-level demographic and technology adoption data
- **Multi-Model Comparison**: Benchmarks performance across five ensemble learning algorithms
- **Category-Specific Analysis**: Breaks down trends by crime type (identity theft, online fraud, cyberstalking, etc.)
- **Temporal Pattern Recognition**: Tracks year-over-year changes from 2015 to 2020
- **Quantitative Evaluation**: Measures model performance using multiple regression metrics

---

## Models Implemented

| Model | Use Case |
|-------|----------|
| Random Forest | Baseline ensemble predictor |
| Gradient Boosting | Sequential error correction |
| AdaBoost | Adaptive weak learner boosting |
| Bagging | Variance reduction through bootstrap aggregation |
| XGBoost | Regularized gradient boosting |

---

## Performance Metrics

Models are evaluated using:

- **R² Score**: Variance explained by the model
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error

---

## Data Sources

The analysis integrates three primary datasets:

**1. Demographic & Technology Dataset**
- State-level population figures
- Internet and broadband subscription rates
- Reported cybercrime incidents

**2. Crime Category Dataset**
- Identity theft cases
- Cyberstalking incidents
- Online banking fraud
- Additional category breakdowns

**3. Temporal Trends Dataset**
- State-wise annual data (2015-2020)
- Used for longitudinal analysis

---

## Technology Stack

```
Python 3.x
├── scikit-learn (model training & evaluation)
├── XGBoost (gradient boosting)
├── pandas & NumPy (data manipulation)
├── matplotlib & seaborn (visualization)
└── Jupyter Notebook (development environment)
```

---

## Getting Started

### Prerequisites

```bash
Python 3.7+
pip
```

### Installation

```bash
# Clone repository
git clone https://github.com/shivam-kun/ThreatLens.git
cd ThreatLens

# Install dependencies
pip install -r requirements.txt

# Run analysis pipeline
python generate.py
```

---

## Model Inputs & Outputs

**Input Features:**
- Year
- State/Region
- Population density
- Internet penetration rate
- Broadband subscription data

**Output:**
- Predicted cybercrime rate
- Model confidence scores
- Feature importance rankings

---

## Key Findings

1. **Model Performance**: Ensemble methods consistently outperform single-classifier approaches
2. **Correlation Trends**: Strong positive relationship between internet adoption rates and cybercrime incidence
3. **Category Distribution**: Financial fraud and identity theft account for the majority of reported cases
4. **Regional Variations**: Significant geographical disparities in crime rates even after controlling for population

---

## Roadmap

- [ ] Integrate real-time data feeds
- [ ] Develop interactive web dashboard
- [ ] Implement deep learning models (LSTM, Transformer architectures)
- [ ] Add geospatial visualization layer
- [ ] Expand dataset to include 2021-2024

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.

---

## License

This project is available under the MIT License.

---

## Author

**Shivam Kundu**  
[GitHub](https://github.com/shivam-kun) • [LinkedIn](#)

---

## Acknowledgments

Data sources and references used in this analysis are documented in the `/docs` folder.

---

*For questions or collaboration opportunities, please open an issue or reach out directly.*