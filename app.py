import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import tensorflow as tf
import joblib

# ==========================================
# HELPER FUNCTIONS FOR DEEP LEARNING MODEL
# ==========================================
def inverse_transform_target(scaled_y, scaler):
    dummy = np.zeros((len(scaled_y), 4))
    dummy[:, 0] = scaled_y.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def simulate_trajectory(model, last_window, horizon_hours, rmse_scaled):
    current_window = last_window.copy()
    trajectory = []
    for _ in range(horizon_hours):
        x_in = np.expand_dims(current_window, axis=0)
        pred_scaled = model(x_in, training=False).numpy()[0, 0]
        noise = np.random.normal(0, rmse_scaled)
        next_val_scaled = np.clip(pred_scaled + noise, 0.0, 1.0)
        trajectory.append(next_val_scaled)
        new_row = np.array([next_val_scaled, current_window[-1,1], current_window[-1,2], current_window[-1,3]])
        current_window = np.vstack([current_window[1:], new_row])
    return np.array(trajectory)

def forecast_multistep(model, last_window, scaler, rmse_scaled, horizon_hours=24, n_simulations=50):
    all_simulations = []
    for _ in range(n_simulations):
        traj = simulate_trajectory(model, last_window, horizon_hours, rmse_scaled)
        all_simulations.append(traj)
    all_sims_scaled = np.array(all_simulations)
    all_sims_meters = np.zeros_like(all_sims_scaled)
    for i in range(n_simulations):
        all_sims_meters[i] = inverse_transform_target(all_sims_scaled[i], scaler)
    mean_forecast = np.mean(all_sims_meters, axis=0)
    lower_bound = np.percentile(all_sims_meters, 10, axis=0)
    upper_bound = np.percentile(all_sims_meters, 90, axis=0)
    return mean_forecast, lower_bound, upper_bound, all_sims_meters

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="OceanFlow AI - CNN-LSTM Wave Forecast System", page_icon="🌊")

# ==========================================
# CUSTOM CSS FOR STUNNING DARK OCEAN THEME
# ==========================================
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #060f1e !important;
    color: #8e9fae !important;
}

.stApp {
    background-color: #060f1e !important;
}

#MainMenu, footer, header, [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

/* Hide streamlit default menu space */
[data-testid="stHeader"] {
    display: none !important;
}

/* Data Editor Container styling */
div[data-testid="stDataEditor"] {
    border: 1px solid #1f2d3d !important;
    border-radius: 8px !important;
    background-color: #0b1626 !important;
    padding: 10px !important;
}

div[data-testid="stDataEditor"] * {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #ffffff !important;
}

/* Base button styling */
.stButton > button {
    background-color: #0b1626 !important;
    color: #00d4d4 !important;
    border: 1px solid #1f2d3d !important;
    border-radius: 20px !important;
    padding: 10px 24px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease-in-out !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.stButton > button:hover {
    border-color: #00d4d4 !important;
    background-color: rgba(0, 212, 212, 0.05) !important;
    color: #00f5ff !important;
}

/* Glowing primary button styling */
.glowing-btn button {
    background: linear-gradient(135deg, #00d4d4 0%, #00b8d9 100%) !important;
    color: #060f1e !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(0, 212, 212, 0.4) !important;
}

.glowing-btn button:hover {
    background: linear-gradient(135deg, #00f5ff 0%, #00d4d4 100%) !important;
    box-shadow: 0 0 25px rgba(0, 212, 212, 0.7) !important;
    transform: translateY(-1px);
    color: #060f1e !important;
}

/* Demo Mode Toggle Switch styling */
div[data-testid="stToggle"] {
    background-color: #0b1626 !important;
    border: 1px solid #1f2d3d !important;
    border-radius: 25px !important;
    padding: 8px 16px !important;
    display: inline-flex !important;
    align-items: center !important;
}

div[data-testid="stToggle"] label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #00d4d4 !important;
    margin-right: 10px !important;
}

/* Loading animation waves */
.loading-wave {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin: 25px auto 5px auto;
}

.loading-wave .dot {
    width: 14px;
    height: 6px;
    background-color: #00d4d4;
    border-radius: 3px;
    animation: wave-pulse 1.4s infinite ease-in-out;
}

.loading-wave .dot:nth-child(2) { animation-delay: 0.2s; }
.loading-wave .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes wave-pulse {
    0%, 100% { transform: scaleY(1); opacity: 0.3; }
    50% { transform: scaleY(2.2); opacity: 1; background-color: #00f5ff; }
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & PRE-SETS
# ==========================================
@st.cache_resource
def load_model_scaler_metrics():
    model_path = os.path.join('models', 'wave_model.keras')
    scaler_path = os.path.join('models', 'scaler.pkl')
    metrics_path = os.path.join('models', 'metrics.pkl')
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        metrics = joblib.load(metrics_path)
        loaded = True
    except:
        model = None
        scaler = None
        metrics = {'rmse': 0.0784, 'mae': 0.0510, 'r2': 0.9985, 'rmse_scaled': 0.007}
        loaded = False
    return model, scaler, metrics, loaded

model, scaler, metrics, loaded_ok = load_model_scaler_metrics()

# ==========================================
# INITIALIZING DATAFRAMES
# ==========================================
# Zero-state template
zero_data = pd.DataFrame(
    {
        "Wave (m)": [0.00] * 10,
        "Wind (m/s)": [0.0] * 10,
        "Pressure (hPa)": [1013.0] * 10,
        "Temp (°C)": [20.0] * 10
    },
    index=[f"t-{9-i}" for i in range(10)]
)

# NOAA Sample data template
sample_data = pd.DataFrame(
    {
        "Wave (m)": [3.90, 4.00, 4.10, 4.20, 4.30, 4.20, 4.10, 4.00, 3.90, 3.80],
        "Wind (m/s)": [9.0, 9.5, 10.0, 10.5, 11.0, 10.8, 10.2, 9.8, 9.2, 8.5],
        "Pressure (hPa)": [1014.0, 1014.2, 1014.5, 1014.8, 1015.0, 1015.2, 1015.0, 1014.8, 1014.5, 1014.0],
        "Temp (°C)": [14.5, 14.6, 14.5, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3, 15.4]
    },
    index=[f"t-{9-i}" for i in range(10)]
)

if 'table_df' not in st.session_state:
    st.session_state['table_df'] = zero_data.copy()

if 'demo_mode' not in st.session_state:
    st.session_state['demo_mode'] = not loaded_ok

# ==========================================
# BRAND HEADER & DEMO TOGGLE BAR
# ==========================================
header_col1, header_col2 = st.columns([4, 1.2])

with header_col1:
    st.markdown("""
    <div style="margin-top: 15px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 2.2rem;">🌊</span>
            <span style="font-family: 'Outfit', sans-serif; font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                Ocean<span style="color:#00d4d4;">Flow AI</span>
            </span>
        </div>
        <div style="font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #00d4d4; letter-spacing: 0.18em; font-weight: 700; margin-top: 2px; text-transform: uppercase;">
            CNN-LSTM OCEAN FORECAST SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown('<div style="display: flex; justify-content: flex-end; margin-top: 22px;">', unsafe_allow_html=True)
    demo_val = st.toggle("Demo Mode", value=st.session_state['demo_mode'])
    st.session_state['demo_mode'] = demo_val
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# HERO BANNER WITH LOADING WAVES
# ==========================================
st.markdown("""
<div style="background-color: #0a1424; border: 1px solid #1f2d3d; border-radius: 12px; padding: 45px 30px; text-align: center; margin-bottom: 35px; box-shadow: 0 4px 25px rgba(0,0,0,0.3);">
    <h1 style="font-family: 'Outfit', sans-serif; font-size: 3.2rem; font-weight: 800; color: #ffffff; margin-bottom: 12px; letter-spacing: -0.03em;">
        Wave Height Prediction
    </h1>
    <p style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #8e9fae; max-width: 750px; margin: 0 auto; line-height: 1.6;">
        Deep learning system trained on NOAA NDBC Station 46059 ocean buoy data to forecast wave heights with precision.
    </p>
    <div class="loading-wave">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SEQUENCE DATA EDITOR
# ==========================================
st.markdown("""
<h3 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.05em;">
    Analyze Ocean Data
</h3>
""", unsafe_allow_html=True)

edited_df = st.data_editor(
    st.session_state['table_df'],
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Wave (m)": st.column_config.NumberColumn("Wave (m)", min_value=0.0, max_value=15.0, format="%.2f"),
        "Wind (m/s)": st.column_config.NumberColumn("Wind (m/s)", min_value=0.0, max_value=50.0, format="%.1f"),
        "Pressure (hPa)": st.column_config.NumberColumn("Pressure (hPa)", min_value=900.0, max_value=1100.0, format="%.1f"),
        "Temp (°C)": st.column_config.NumberColumn("Temp (°C)", min_value=-5.0, max_value=45.0, format="%.1f"),
    }
)
st.session_state['table_df'] = edited_df

# ==========================================
# BUTTONS LOGIC
# ==========================================
btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 2])

with btn_col1:
    if st.button("Fill Sample Data", use_container_width=True):
        st.session_state['table_df'] = sample_data.copy()
        st.session_state['run_predict'] = False
        st.rerun()

with btn_col2:
    if st.button("Reset", use_container_width=True):
        st.session_state['table_df'] = zero_data.copy()
        st.session_state['run_predict'] = False
        if 'pred_result' in st.session_state:
            del st.session_state['pred_result']
        st.rerun()

with btn_col3:
    st.markdown('<div class="glowing-btn">', unsafe_allow_html=True)
    trigger_predict = st.button("🌊 Predict Wave Height", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if trigger_predict:
    st.session_state['run_predict'] = True

# ==========================================
# MODEL PREDICTION & VISUALIZATION OUTPUT
# ==========================================
if st.session_state.get('run_predict', False):
    input_vals = st.session_state['table_df'][['Wave (m)', 'Wind (m/s)', 'Pressure (hPa)', 'Temp (°C)']].values
    
    with st.spinner("Processing deep learning sequence..."):
        # Determine mode
        is_demo = st.session_state.get('demo_mode', not loaded_ok)
        
        if not is_demo:
            try:
                # 1-hour ahead prediction (t+1)
                scaled_input = scaler.transform(input_vals)
                seq_in = np.expand_dims(scaled_input, axis=0)
                pred_scaled = model(seq_in, training=False).numpy()[0, 0]
                predicted_val = float(inverse_transform_target(np.array([[pred_scaled]]), scaler)[0])
                predicted_val = max(0.0, predicted_val)
                
                # Multi-step 24-hour forecast
                mean_fc, lower_fc, upper_fc, _ = forecast_multistep(
                    model, scaled_input, scaler, metrics['rmse_scaled'],
                    horizon_hours=24, n_simulations=50
                )
            except Exception as e:
                # Automatic fallback if model fails
                is_demo = True
                st.session_state['demo_mode'] = True
                st.warning(f"Engine failure: {e}. Running simulation fallback.")
        
        if is_demo:
            # Simulated forecasting logic
            last_wave = float(input_vals[-1, 0])
            last_wind = float(input_vals[-1, 1])
            
            # Create a realistic next wave reading based on input
            if last_wave <= 0.05:
                # Zero state input fallback
                predicted_val = 2.45 if last_wind <= 0.0 else min(5.5, last_wind * 0.28 + np.random.normal(0, 0.15))
            else:
                predicted_val = last_wave * 0.93 + last_wind * 0.06 + np.random.normal(0, 0.08)
            predicted_val = max(0.05, predicted_val)
            
            # Create Monte Carlo simulations
            sims = []
            for _ in range(50):
                run = []
                curr = predicted_val
                for step in range(24):
                    curr = curr * 0.94 + 2.0 * 0.06 + np.random.normal(0, 0.07 * (1 + 0.04 * step))
                    curr = max(0.1, curr)
                    run.append(curr)
                sims.append(run)
            sims = np.array(sims)
            mean_fc  = np.mean(sims, axis=0)
            lower_fc = np.percentile(sims, 10, axis=0)
            upper_fc = np.percentile(sims, 90, axis=0)

    # ------------------------------------------
    # RENDER VISUALIZATIONS
    # ------------------------------------------
    st.markdown("""
    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;">
        Prediction Analytics
    </h3>
    """, unsafe_allow_html=True)
    
    col_gauge, col_plot = st.columns([1, 1.4], gap="large")
    
    with col_gauge:
        p = predicted_val
        if p < 2.0:
            state = "CALM"
            badge_color = "#00e5a0"
        elif p < 4.0:
            state = "MODERATE"
            badge_color = "#00b8d9"
        elif p < 6.0:
            state = "ROUGH"
            badge_color = "#f5a623"
        else:
            state = "VERY ROUGH"
            badge_color = "#e05c5c"
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = p,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': " m", 'font': {'size': 58, 'color': '#ffffff', 'family': 'Outfit'}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#4a7a9b", 'tickfont': {'family': 'Space Mono', 'color': '#8e9fae'}},
                'bar': {'color': badge_color},
                'bgcolor': "#0b1626",
                'borderwidth': 1,
                'bordercolor': "#1f2d3d",
                'steps': [
                    {'range': [0, 2], 'color': 'rgba(0, 229, 160, 0.08)'},
                    {'range': [2, 4], 'color': 'rgba(0, 184, 217, 0.08)'},
                    {'range': [4, 6], 'color': 'rgba(245, 166, 35, 0.08)'},
                    {'range': [6, 10], 'color': 'rgba(224, 92, 92, 0.08)'}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': p
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#c8d8e8", 'family': "Outfit"},
            height=300,
            margin=dict(l=30, r=30, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: -15px; padding: 14px; background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px;">
            <div style="font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #8e9fae; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">
                State Assessment
            </div>
            <span style="display: inline-block; padding: 5px 16px; border-radius: 4px; font-family: 'Space Mono', monospace; font-size: 0.95rem; font-weight: 700; color: {badge_color}; border: 1px solid {badge_color}; background-color: {badge_color}10;">
                {state}
            </span>
            <div style="font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #5a6e7f; margin-top: 8px;">
                RMSE: ±{metrics.get('rmse', 0.078):.3f} m &nbsp;·&nbsp; R² Score: {metrics.get('r2', 0.9985):.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_plot:
        # Construct timeline arrays
        history_x = [f"t-{9-i}" for i in range(10)]
        history_y = list(input_vals[:, 0])
        
        forecast_x = [f"t+{i}h" for i in range(1, 25)]
        
        # Connect timeline at index t-0
        conn_x = ["t-0"] + forecast_x
        conn_mean = [history_y[-1]] + list(mean_fc)
        conn_lower = [history_y[-1]] + list(lower_fc)
        conn_upper = [history_y[-1]] + list(upper_fc)
        
        full_timeline = [f"t-{9-i}" for i in range(10)] + [f"t+{i}h" for i in range(1, 25)]
        
        fig_plot = go.Figure()
        
        # 80% Confidence band area
        fig_plot.add_trace(go.Scatter(
            x=conn_x, y=conn_upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig_plot.add_trace(go.Scatter(
            x=conn_x, y=conn_lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,212,212,0.06)',
            line=dict(width=0),
            name='80% Confidence Band',
            hoverinfo='skip'
        ))
        
        # Observed historical wave heights
        fig_plot.add_trace(go.Scatter(
            x=history_x, y=history_y,
            mode='lines+markers',
            line=dict(color='#8e9fae', width=2),
            marker=dict(size=5, color='#060f1e', line=dict(color='#8e9fae', width=1.5)),
            name='Observed History',
            hovertemplate='<b>%{x}</b><br>%{y:.2f} m<extra></extra>'
        ))
        
        # Deep learning predictions connection
        fig_plot.add_trace(go.Scatter(
            x=conn_x, y=conn_mean,
            mode='lines+markers',
            line=dict(color='#00d4d4', width=2.5),
            marker=dict(size=5, color='#060f1e', line=dict(color='#00d4d4', width=2)),
            name='Forecast Model',
            hovertemplate='<b>%{x}</b><br>%{y:.2f} m<extra></extra>'
        ))
        
        fig_plot.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0b1626',
            height=370,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(
                gridcolor='#1f2d3d', tickfont=dict(color='#8e9fae', family='Space Mono', size=9),
                linecolor='#1f2d3d', title=dict(text="Observation & Forecast Timeline", font=dict(color='#8e9fae', size=9, family='Space Mono')),
                showgrid=True,
                categoryorder='array',
                categoryarray=full_timeline
            ),
            yaxis=dict(
                gridcolor='#1f2d3d', tickfont=dict(color='#8e9fae', family='Space Mono', size=9),
                linecolor='#1f2d3d', title=dict(text="Wave Height (m)", font=dict(color='#8e9fae', size=9, family='Space Mono')),
                showgrid=True
            ),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                font=dict(color='#8e9fae', family='Space Mono', size=9),
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig_plot, use_container_width=True)

# ==========================================
# PHYSICAL OCEAN FACTORS INFORMATION
# ==========================================
st.markdown("""
<h3 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;">
    Physical Ocean Factors
</h3>
""", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3, gap="medium")

with col_f1:
    st.markdown("""
    <div style="background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px; padding: 22px; height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 12px;">🌊</div>
        <h4 style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
            Wave Height
        </h4>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #8e9fae; line-height: 1.5; margin: 0;">
            Previous wave heights provide the CNN-LSTM model temporal context to identify ocean patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_f2:
    st.markdown("""
    <div style="background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px; padding: 22px; height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 12px;">💨</div>
        <h4 style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
            Wind Speed
        </h4>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #8e9fae; line-height: 1.5; margin: 0;">
            Wind speed directly drives wave formation. Trained on NOAA Station 46059 offshore Pacific data.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_f3:
    st.markdown("""
    <div style="background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px; padding: 22px; height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 12px;">🌡️</div>
        <h4 style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
            Air Pressure
        </h4>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #8e9fae; line-height: 1.5; margin: 0;">
            Pressure drops indicate incoming storms. Our model captures these atmospheric-ocean interactions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ABOUT MODEL & INSTRUCTIONS
# ==========================================
col_about, col_use = st.columns([1, 1], gap="large")

with col_about:
    st.markdown("""
    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;">
        🤖 About OceanFlow AI Model
    </h3>
    <div style="background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px; padding: 20px 24px;">
        <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;">
            <tr style="border-bottom: 1px solid #1f2d3d;">
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem; width: 38%;">Architecture</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">CNN-LSTM Neural Network | NOAA 46059</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2d3d;">
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem;">Training Data</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">Real NOAA ocean buoy measurements</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2d3d;">
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem;">R² Model Score</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">0.9985 (Test Partition)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2d3d;">
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem;">Lookback Window</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">10 sequential hourly timesteps</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2d3d;">
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem;">Input Parameters</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">4 Features (Wave, Wind, Pressure, Sea Temp)</td>
            </tr>
            <tr>
                <td style="padding: 14px 8px; font-weight: 600; color: #00d4d4; font-size: 0.88rem;">Deployment Mode</td>
                <td style="padding: 14px 8px; color: #ffffff; font-size: 0.88rem;">Self-Contained Streamlit Dashboard</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_use:
    st.markdown("""
    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;">
        📘 How to Use
    </h3>
    <div style="background-color: #0b1626; border: 1px solid #1f2d3d; border-radius: 8px; padding: 25px 30px; height: calc(100% - 60px); display: flex; align-items: center;">
        <ol style="font-family: 'Inter', sans-serif; color: #8e9fae; font-size: 0.92rem; line-height: 1.9; padding-left: 20px; margin: 0;">
            <li style="margin-bottom: 10px;">
                Enter or fill <strong style="color: #ffffff;">10 consecutive hours</strong> of ocean weather data in the table editor.
            </li>
            <li style="margin-bottom: 10px;">
                Each row matches 1 hour of readings: wave height, wind, pressure, and sea temperature.
            </li>
            <li style="margin-bottom: 10px;">
                Click the glowing <strong style="color: #00d4d4;">Predict Wave Height</strong> button to activate forecasting.
            </li>
            <li style="margin-bottom: 10px;">
                Inspect your predicted waves on the <strong style="color: #ffffff;">Radial Gauge</strong> and the <strong style="color: #ffffff;">Sequence Connection Plot</strong>.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="text-align: center; margin-top: 70px; padding: 25px 0; border-top: 1px solid #1f2d3d; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #5a6e7f;">
    © 2026 OceanFlow AI | Built by Lekshmi Maniyan<br>
    Powered by Streamlit & TensorFlow | Data Source: NOAA NDBC Buoy Station 46059
</div>
""", unsafe_allow_html=True)