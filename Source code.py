import streamlit as st
import pandas as pd
import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Forensic Optimizer Master", layout="wide")

st.title("📊 Forensic Bayesian Optimizer Master Suite")
st.markdown("---")

# --- 1. INITIALIZE SESSION STATE ---
if 'vars_config' not in st.session_state:
    st.session_state.vars_config = {} 
if 'experiments' not in st.session_state:
    st.session_state.experiments = pd.DataFrame()
if 'current_suggestion' not in st.session_state:
    st.session_state.current_suggestion = None

# --- 2. DATA LOGGING CALLBACK ---
def commit_experiment(entry_dict):
    # Standardize numerical types for math engine
    for key, val in entry_dict.items():
        if key != 'Target' and st.session_state.vars_config[key]['type'] == 'Numerical':
            entry_dict[key] = float(val)
    
    new_row = pd.DataFrame([entry_dict])
    st.session_state.experiments = pd.concat([st.session_state.experiments, new_row], ignore_index=True)
    st.session_state.current_suggestion = None 
    st.rerun()

# --- SIDEBAR: RECOVERY & PARAMETERS ---
with st.sidebar:
    st.header("💾 Session Recovery")
    uploaded_file = st.file_uploader("Restore previous .csv results", type="csv")
    if uploaded_file is not None:
        try:
            restored_df = pd.read_csv(uploaded_file)
            restored_df = restored_df.loc[:, ~restored_df.columns.str.contains('^Unnamed')]
            if 'Exp_No' in restored_df.columns:
                restored_df = restored_df.drop(columns=['Exp_No'])
            
            if not st.session_state.vars_config or st.sidebar.button("Sync Config to Upload"):
                new_config = {}
                for col in restored_df.columns:
                    if col != 'Target':
                        new_config[col] = {
                            'type': 'Numerical', 'min': float(restored_df[col].min()), 
                            'max': float(restored_df[col].max()), 'unit': '', 'step': 1.0
                        }
                st.session_state.vars_config = new_config
                st.session_state.experiments = restored_df
                st.success("✅ Session Restored & Synced")
        except Exception as e:
            st.error(f"Error: {e}")

    st.header("🛠️ Parameter Manager")
    p_name = st.text_input("Parameter Name")
    p_type = st.selectbox("Type", ["Numerical", "Categorical"])
    
    with st.form("manual_var_form", clear_on_submit=True):
        if p_type == "Numerical":
            p_unit = st.text_input("Unit", placeholder="e.g. mL/min")
            p_min = st.number_input("Min", value=0.0)
            p_max = st.number_input("Max", value=100.0)
            p_step = st.number_input("Step", value=1.0)
            if st.form_submit_button("Add Parameter"):
                if p_name:
                    st.session_state.vars_config[p_name] = {'type': 'Numerical', 'min': p_min, 'max': p_max, 'unit': p_unit, 'step': p_step}
                    st.rerun()
        else:
            p_options = st.text_input("Options (comma separated)")
            if st.form_submit_button("Add Parameter"):
                if p_name:
                    opts = [x.strip() for x in p_options.split(",") if x.strip()]
                    st.session_state.vars_config[p_name] = {'type': 'Categorical', 'options': opts, 'unit': ''}
                    st.rerun()

    if st.button("🗑️ Reset All App Data"):
        st.session_state.vars_config = {}
        st.session_state.experiments = pd.DataFrame()
        st.session_state.current_suggestion = None
        st.rerun()

# --- MAIN AREA: DATA LOG ---
if st.session_state.vars_config:
    st.subheader("📝 Lab Notebook")
    
    with st.expander("⚙️ Refine Parameter Intervals (Steps)", expanded=False):
        cols = st.columns(len(st.session_state.vars_config))
        for i, (f_name, f_cfg) in enumerate(st.session_state.vars_config.items()):
            with cols[i]:
                if f_cfg['type'] == 'Numerical':
                    st.session_state.vars_config[f_name]['step'] = st.number_input(f"{f_name} Step", value=float(f_cfg.get('step', 1.0)), key=f"ed_{f_name}")

    if not st.session_state.experiments.empty:
        display_df = st.session_state.experiments.copy()
        display_df.insert(0, 'Exp_No', range(1, len(display_df) + 1))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv = st.session_state.experiments.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Save to CSV", data=csv, file_name="forensic_data.csv")

# --- WORKFLOW: ENTRY & ANALYTICS ---
if st.session_state.vars_config:
    st.divider()
    c_form, c_dash = st.columns([1, 2.5])
    
    with c_form:
        st.subheader("📥 Data Entry")
        
        if len(st.session_state.experiments) >= 3:
            if st.button("🔮 Suggest Next Run", type="primary", use_container_width=True):
                df_enc = st.session_state.experiments.copy()
                feats = [c for c in df_enc.columns if c != 'Target']
                for col in feats:
                    if st.session_state.vars_config[col]['type'] == 'Categorical':
                        opts = st.session_state.vars_config[col]['options']
                        df_enc[col] = df_enc[col].apply(lambda x: opts.index(x) if x in opts else 0)
                
                X, y = df_enc[feats].values, df_enc['Target'].values
                gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
                gp.fit(X, y)
                
                # Grid Search with Snap
                pts = []
                for _ in range(2000):
                    pt = []
                    for f in feats:
                        cfg = st.session_state.vars_config[f]
                        if cfg['type'] == 'Numerical':
                            val = np.random.uniform(cfg['min'], cfg['max'])
                            pt.append(np.round(val / cfg['step']) * cfg['step'])
                        else:
                            pt.append(np.random.choice(range(len(cfg['options']))))
                    pts.append(pt)
                
                pts = np.unique(np.array(pts), axis=0)
                mu, std = gp.predict(pts, return_std=True)
                # Acquisition Function: Expected Improvement
                ei = (mu - np.max(y)) * norm.cdf((mu - np.max(y))/std) + std * norm.pdf((mu - np.max(y))/std)
                best = pts[np.argmax(ei)]
                st.session_state.current_suggestion = {n: (st.session_state.vars_config[n]['options'][int(best[i])] if st.session_state.vars_config[n]['type'] == 'Categorical' else best[i]) for i, n in enumerate(feats)}
                st.rerun()

        if st.session_state.current_suggestion:
            with st.form("sugg_form"):
                entry = {}
                for n, cfg in st.session_state.vars_config.items():
                    val = st.session_state.current_suggestion[n]
                    st.write(f"**{n}**: {val}")
                    entry[n] = val
                entry['Target'] = st.number_input("Measured Outcome", format="%.6e")
                if st.form_submit_button("Log Suggested Run"):
                    commit_experiment(entry)
        else:
            with st.form("manual_form"):
                entry = {}
                for n, cfg in st.session_state.vars_config.items():
                    if cfg['type'] == 'Numerical':
                        entry[n] = st.number_input(n, step=cfg['step'], value=float(cfg['min']))
                    else:
                        entry[n] = st.selectbox(n, options=cfg['options'])
                entry['Target'] = st.number_input("Measured Outcome", format="%.6e")
                if st.form_submit_button("Log Manual Run"):
                    commit_experiment(entry)

    with c_dash:
        if not st.session_state.experiments.empty:
            st.subheader("📊 Analytics")
            plots = st.multiselect("Visuals:", ["Convergence", "Experimental Path", "Surface (2D/3D)", "Parameter Importance"], 
                                   default=["Convergence", "Experimental Path"])
            y_vals = st.session_state.experiments['Target'].values

            if "Convergence" in plots:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_vals, mode='markers', name='Actual Runs'))
                fig.add_trace(go.Scatter(y=np.maximum.accumulate(y_vals), mode='lines', name='Trend', line=dict(color='red')))
                fig.update_layout(title="Convergence", yaxis=dict(tickformat=".2e"), xaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig, use_container_width=True)

            if "Experimental Path" in plots and len(st.session_state.vars_config) >= 2:
                f_list = [c for c in st.session_state.experiments.columns if c != 'Target']
                px1 = st.selectbox("X Axis:", f_list, index=0, key="path_x")
                px2 = st.selectbox("Y Axis:", f_list, index=1, key="path_y")
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=st.session_state.experiments[px1], y=st.session_state.experiments[px2],
                    mode='lines+markers+text',
                    text=[f"#{i+1}: {v:.1e}" for i, v in enumerate(y_vals)],
                    textposition="top center",
                    line=dict(color='rgba(150,150,150,0.5)'),
                    marker=dict(size=14, color=y_vals, colorscale='RdYlGn', showscale=True, colorbar=dict(tickformat=".2e"))
                ))
                fig_p.update_layout(title="Trajectory Map", xaxis_title=px1, yaxis_title=px2)
                st.plotly_chart(fig_p, use_container_width=True)

            if ("Surface (2D/3D)" in plots or "Parameter Importance" in plots) and len(st.session_state.experiments) >= 3:
                f_list = [c for c in st.session_state.experiments.columns if c != 'Target']
                df_m = st.session_state.experiments.copy()
                for col in f_list:
                    if st.session_state.vars_config[col]['type'] == 'Categorical':
                        opts_m = st.session_state.vars_config[col]['options']
                        df_m[col] = df_m[col].apply(lambda x: opts_m.index(x) if x in opts_m else 0)
                
                X_m, y_m = df_m[f_list].values, df_m['Target'].values
                gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
                gp.fit(X_m, y_m)

                if "Surface (2D/3D)" in plots:
                    st.write("---")
                    sx = st.selectbox("Surface X:", f_list, index=0, key="surf_x")
                    sy = st.selectbox("Surface Y:", f_list, index=1, key="surf_y")
                    is_3d = st.toggle("3D Visualization", value=True)
                    
                    cx, cy = st.session_state.vars_config[sx], st.session_state.vars_config[sy]
                    xr = np.linspace(cx.get('min',0), cx.get('max',1), 30)
                    yr = np.linspace(cy.get('min',0), cy.get('max',1), 30)
                    xx, yy = np.meshgrid(xr, yr)
                    
                    best_idx = np.argmax(y_m)
                    grid = []
                    for feat in f_list:
                        if feat == sx: grid.append(xx.ravel())
                        elif feat == sy: grid.append(yy.ravel())
                        else: grid.append(np.full(xx.ravel().shape, X_m[best_idx, f_list.index(feat)]))
                    
                    mu = gp.predict(np.array(grid).T)
                    if is_3d:
                        fig_s = go.Figure(data=[go.Surface(z=mu.reshape(xx.shape), x=xr, y=yr, colorscale='Viridis')])
                        fig_s.update_layout(scene=dict(zaxis=dict(tickformat=".2e"), xaxis_title=sx, yaxis_title=sy))
                    else:
                        fig_s = go.Figure(data=go.Contour(z=mu.reshape(xx.shape), x=xr, y=yr, colorscale='Viridis', colorbar=dict(tickformat=".2e")))
                        fig_s.update_layout(xaxis_title=sx, yaxis_title=sy)
                    st.plotly_chart(fig_s, use_container_width=True)

                if "Parameter Importance" in plots:
                    st.write("---")
                    sens = []
                    for i, name in enumerate(f_list):
                        tp = np.tile(X_m[np.argmax(y_m)], (50, 1))
                        cfg = st.session_state.vars_config[name]
                        rmin, rmax = (cfg.get('min',0), cfg.get('max',1)) if cfg['type'] == 'Numerical' else (0, len(cfg['options'])-1)
                        tp[:, i] = np.linspace(rmin, rmax, 50)
                        sens.append(np.ptp(gp.predict(tp)))
                    st.plotly_chart(px.bar(x=f_list, y=sens, title="Impact Analysis", labels={'x':'Factor','y':'Relative Impact'}), use_container_width=True)
