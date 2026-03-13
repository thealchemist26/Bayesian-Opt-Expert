import streamlit as st
import pandas as pd
import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Forensic Optimizer Pro", layout="wide")

st.title("📊 Forensic Bayesian Optimizer")
st.markdown("---")

# --- INITIALIZE STATE ---
if 'vars_config' not in st.session_state:
    st.session_state.vars_config = {} 
if 'experiments' not in st.session_state:
    st.session_state.experiments = pd.DataFrame()

# --- SIDEBAR: PARAMETER MANAGER ---
with st.sidebar:
    st.header("🛠️ Parameter Manager")
    p_name = st.text_input("Parameter Name", placeholder="e.g. Sonication Time")
    p_type = st.selectbox("Type", ["Numerical", "Categorical"])
    
    with st.form("var_form", clear_on_submit=True):
        if p_type == "Numerical":
            p_unit = st.text_input("Unit", placeholder="min")
            p_min = st.number_input("Min Range", value=0.0)
            p_max = st.number_input("Max Range", value=100.0)
            p_step = st.number_input("Step/Interval", value=1.0, min_value=0.0001)
            
            if st.form_submit_button("Add Numerical Parameter"):
                if p_name:
                    st.session_state.vars_config[p_name] = {
                        'type': 'Numerical', 'min': p_min, 'max': p_max, 'unit': p_unit, 'step': p_step
                    }
                    st.rerun()
        else:
            p_options = st.text_input("Options (comma separated)", placeholder="MeOH, ACN")
            if st.form_submit_button("Add Categorical Parameter"):
                if p_name:
                    opts = [x.strip() for x in p_options.split(",") if x.strip()]
                    st.session_state.vars_config[p_name] = {'type': 'Categorical', 'options': opts, 'unit': ''}
                    st.rerun()

    st.write("---")
    if st.button("🗑️ Reset All Data"):
        st.session_state.vars_config = {}
        st.session_state.experiments = pd.DataFrame()
        st.rerun()

# --- DATA ENTRY ---
if not st.session_state.vars_config:
    st.info("👈 Add parameters in the sidebar to begin.")
else:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📥 Log Result")
        with st.form("input_data"):
            entry = {}
            for name, cfg in st.session_state.vars_config.items():
                label = f"{name} ({cfg['unit']})" if cfg['type'] == "Numerical" else name
                if cfg['type'] == "Numerical":
                    entry[name] = st.number_input(label, step=cfg['step'], value=float(cfg['min']))
                else:
                    entry[name] = st.selectbox(label, options=cfg['options'])
            entry['Target'] = st.number_input("Outcome (PAR/Recovery)", format="%.6f")
            if st.form_submit_button("Log Run"):
                st.session_state.experiments = pd.concat([st.session_state.experiments, pd.DataFrame([entry])], ignore_index=True)
                st.rerun()
    with c2:
        st.subheader("📝 Data Log")
        st.dataframe(st.session_state.experiments, use_container_width=True)

# --- STATS DASHBOARD ---
if len(st.session_state.experiments) >= 3:
    st.divider()
    plots = st.multiselect("Visualizations:", ["Convergence Plot", "Response Surface (3D)"], default=["Convergence Plot"])

    df_enc = st.session_state.experiments.copy()
    features = [c for c in df_enc.columns if c != 'Target']
    for col in features:
        if st.session_state.vars_config[col]['type'] == 'Categorical':
            opts = st.session_state.vars_config[col]['options']
            df_enc[col] = df_enc[col].apply(lambda x: opts.index(x))
    
    X, y = df_enc[features].values, df_enc['Target'].values
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
    gp.fit(X, y)

    if "Convergence Plot" in plots:
        st.plotly_chart(px.line(y=np.maximum.accumulate(y), title="Best Result Found So Far"), use_container_width=True)

    if "Response Surface (3D)" in plots and len(features) >= 2:
        cfg1, cfg2 = st.session_state.vars_config[features[0]], st.session_state.vars_config[features[1]]
        x_range = np.arange(cfg1.get('min', 0), cfg1.get('max', 1)+cfg1.get('step', 1), cfg1.get('step', 1))
        y_range = np.arange(cfg2.get('min', 0), cfg2.get('max', 1)+cfg2.get('step', 1), cfg2.get('step', 1))
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.c_[xx.ravel(), yy.ravel()]
        for i in range(2, len(features)):
            grid = np.c_[grid, np.full(xx.ravel().shape, X[np.argmax(y), i])]
        mu = gp.predict(grid)
        st.plotly_chart(go.Figure(data=[go.Surface(z=mu.reshape(xx.shape), x=x_range, y=y_range)]), use_container_width=True)

    st.subheader("🔮 Next Practical Experiment")
    pts = []
    for _ in range(2000):
        pt = []
        for f in features:
            cfg = st.session_state.vars_config[f]
            if cfg['type'] == 'Numerical':
                val = np.random.uniform(cfg['min'], cfg['max'])
                snapped = np.round(val / cfg['step']) * cfg['step']
                pt.append(np.clip(snapped, cfg['min'], cfg['max']))
            else:
                pt.append(np.random.choice(range(len(cfg['options']))))
        pts.append(pt)
    
    pts = np.unique(np.array(pts), axis=0)
    mu, std = gp.predict(pts, return_std=True)
    cur_best = np.max(y)
    with np.errstate(divide='ignore'):
        imp = mu - cur_best
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std <= 0.0] = 0.0
    
    best_pt = pts[np.argmax(ei)]
    cols = st.columns(len(features))
    for i, f in enumerate(features):
        cfg = st.session_state.vars_config[f]
        val = cfg['options'][int(best_pt[i])] if cfg['type'] == 'Categorical' else f"{best_pt[i]:.2f} {cfg['unit']}"
        cols[i].metric(label=f, value=val)
