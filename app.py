import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Veda Tech | Demand Forecasting", page_icon="📈", layout="wide")

# --- Custom CSS for Premium Corporate Look ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Adjust Main Container Width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Header Styling */
    .main-header { 
        font-size: 38px; 
        color: #1f4287; 
        font-weight: 800; 
        margin-bottom: 5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header { 
        font-size: 16px; 
        color: #6c757d; 
        margin-bottom: 30px;
        font-weight: 500;
    }

    /* KPI Cards Styling */
    .metric-box { 
        background-color: #ffffff;
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 4px solid #1f4287;
    }
    .metric-value { 
        font-size: 32px; 
        color: #1f4287; 
        font-weight: 800; 
    }
    .metric-label { 
        font-size: 14px; 
        color: #6c757d; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    /* Insight Box Styling */
    .insight-box { 
        background-color: #ffffff; 
        padding: 25px; 
        border-left: 5px solid #ff7f0e; 
        border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .insight-box h4 {
        color: #1f4287;
        margin-top: 0;
        font-size: 20px;
        font-weight: 700;
    }
    .insight-box ul {
        color: #333333;
        font-size: 15px;
        line-height: 1.6;
    }
    .insight-box b {
        color: #d9534f;
    }
    
    /* Sidebar Styling (Dark Corporate Blue) */
    [data-testid="stSidebar"] {
        background-color: #1f4287;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    df = pd.read_csv('historical_data.csv', parse_dates=['Date'], index_col='Date')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    return model

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files. Make sure 'historical_data.csv' and 'rf_model.pkl' are in the same folder. Error: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")
    forecast_months = st.slider("Select Forecast Horizon (Months)", min_value=3, max_value=12, value=6)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("This dashboard uses a Random Forest Machine Learning model trained on historical synthetic data to predict future service demand.")

# --- Header ---
st.markdown('<div class="main-header">Sales & Service Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Veda Technology - AI Powered Business Planning Dashboard</div>', unsafe_allow_html=True)

# --- KPI Metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-value">{len(df)}</div><div class="metric-label">Months of Data</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-value">{df["Demand"].mean():.0f}</div><div class="metric-label">Avg Monthly Demand</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><div class="metric-value">{df["Demand"].max():.0f}</div><div class="metric-label">Peak Demand Recorded</div></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- Forecasting Logic ---
def create_features(data):
    df_feat = data.copy()
    df_feat['Month'] = df_feat.index.month
    df_feat['Quarter'] = df_feat.index.quarter
    df_feat['Lag_1'] = df_feat['Demand'].shift(1)
    df_feat['Lag_2'] = df_feat['Demand'].shift(2)
    df_feat['Rolling_Mean_3'] = df_feat['Demand'].rolling(window=3).mean()
    return df_feat.dropna()

future_forecast = []
last_data = df.copy()

for i in range(forecast_months):
    temp_df = create_features(last_data)
    last_row = temp_df.iloc[-1:].drop('Demand', axis=1)
    pred = model.predict(last_row)[0]
    future_forecast.append(pred)
    
    next_date = last_data.index[-1] + pd.DateOffset(months=1)
    last_data = pd.concat([last_data, pd.DataFrame({'Demand': [pred]}, index=[next_date])])

future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='M')
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Demand': future_forecast}).set_index('Date')

# --- Visualization ---
st.subheader("📊 Demand Trend & Future Forecast")

fig = go.Figure()

# Historical Data
fig.add_trace(go.Scatter(x=df.index, y=df['Demand'], mode='lines+markers', name='Historical Demand', line=dict(color='#1f4287', width=3)))

# Forecasted Data
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecasted Demand'], mode='lines+markers', name='Forecasted Demand', line=dict(color='#ff7f0e', width=3, dash='dash')))

# Force Light Theme for Plotly Chart
fig.update_layout(
    title="Historical vs Forecasted Service Demand",
    xaxis_title="Date",
    yaxis_title="Demand Volume",
    hovermode="x unified",
    paper_bgcolor='white',  # Fixes black background
    plot_bgcolor='white',   # Fixes black background
    font=dict(color='#333333'), # Dark text
    xaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
    yaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- Business Recommendations Section ---
st.markdown("---")
st.subheader("💡 Business Planning Recommendations")

peak_month = forecast_df['Forecasted Demand'].idxmax().strftime('%B %Y')
low_month = forecast_df['Forecasted Demand'].idxmin().strftime('%B %Y')
avg_forecast = forecast_df['Forecasted Demand'].mean()

st.markdown(f"""
<div class="insight-box">
    <h4>Based on the AI Forecast for the next {forecast_months} months:</h4>
    <ul>
        <li><b>Peak Demand Alert:</b> Expect the highest demand in <b>{peak_month}</b>. Ensure adequate staffing and resource allocation for this period.</li>
        <li><b>Low Demand Period:</b> Demand is expected to dip in <b>{low_month}</b>. This is an ideal time for team training, system maintenance, or launching marketing campaigns.</li>
        <li><b>Average Expected Demand:</b> The average forecasted demand is <b>{avg_forecast:.0f} units</b> per month.</li>
        <li><b>Model Limitation:</b> Note that this model relies on historical trends. Sudden market changes, new competitors, or economic shifts cannot be predicted by this model alone. Always combine AI insights with human business judgment.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer

st.markdown("<br><hr><center style='color: #888;'>Developed for Veda Technology Internship | Powered by Streamlit & Scikit-Learn</center>", unsafe_allow_html=True)
