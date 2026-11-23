# 📈 Sales & Service Demand Forecasting System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview
This project is an **AI-powered Sales & Service Demand Forecasting System** developed as a part of the Data Science Internship at **Veda Technology**. 

The primary objective is to predict future demand for technology services, digital services, and training programs using historical time-series data. By analyzing past trends and seasonality, the system provides actionable insights to help the business optimize staffing, resource allocation, and marketing strategies.

🔗 **Live Dashboard:** [Insert your Streamlit App Link Here]

---

## 🚀 Key Features
*   **Synthetic Data Generation:** Created a robust 3-year historical dataset simulating real-world service demand with trends, seasonality, and sudden market dips.
*   **Exploratory Data Analysis (EDA):** Visualized trends, seasonality, and residuals using time-series decomposition.
*   **Advanced Modeling:** Compared a baseline Moving Average model with Statistical (SARIMA) and Machine Learning (Random Forest Regressor) models.
*   **Interactive Dashboard:** A premium, fully responsive Streamlit web app featuring dynamic forecasting, KPI metrics, and business recommendations.
*   **Business Insights:** Automated generation of peak and low-demand alerts to assist in business planning.

---

## 🛠️ Tech Stack & Tools
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn, Plotly
*   **Machine Learning & Statistics:** Scikit-Learn, Statsmodels
*   **Web Framework:** Streamlit
*   **Model Serialization:** Joblib

---

## 📊 Methodology & Workflow
1.  **Data Generation:** Simulated daily demand data from 2021-2023, incorporating an upward trend, yearly seasonality, and random noise.
2.  **Preprocessing:** Resampled daily data to monthly averages to reduce noise and improve forecasting accuracy.
3.  **Feature Engineering:** Created lag features (`Lag_1`, `Lag_2`), rolling means, and extracted time-based features (Month, Quarter) for the ML model.
4.  **Model Training:**
    *   *Baseline:* 3-Month Moving Average.
    *   *Statistical:* SARIMA (Seasonal ARIMA).
    *   *Machine Learning:* Random Forest Regressor.
5.  **Evaluation:** Models were evaluated using MAE, RMSE, and MAPE. The Random Forest model was selected for deployment due to its superior performance.
6.  **Deployment:** The final model was integrated into a Streamlit dashboard for real-time user interaction.

---

## 💻 How to Run Locally

If you want to run this project on your local machine, follow these steps:

