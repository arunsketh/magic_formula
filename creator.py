import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from opentire import OpenTire

# --- App Configuration ---
st.set_page_config(layout="wide")

# --- Initialize OpenTire (cached for performance) ---
@st.cache_resource
def init_opentire():
    """Initializes the OpenTire library instance."""
    return OpenTire()

openTire = init_opentire()

# --- App Title and Description ---
st.title("OpenTire Combined-Slip Model Fitter 🚗")
st.markdown("Upload your tire measurement data (CSV) to fit a Pacejka 2002 model to both lateral and longitudinal forces.")

# --- File Uploader and Sidebar ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # --- Column Mapping ---
    st.sidebar.header("2. Map Data Columns")
    st.sidebar.info("Map the columns in your CSV to the required tire properties.")
    
    col_options = list(data.columns)
    # Map all required inputs and outputs
    map_sa = st.sidebar.selectbox("Slip Angle (SA) Column", col_options, index=col_options.index("SLIP_ANGLE"))
    map_sr = st.sidebar.selectbox("Slip Ratio (SR) Column", col_options, index=col_options.index("SLIP_RATIO"))
    map_fz = st.sidebar.selectbox("Normal Load (FZ) Column", col_options, index=col_options.index("NORMAL_LOAD"))
    map_p = st.sidebar.selectbox("Inflation Pressure (P) Column", col_options, index=col_options.index("INFLATION_PRESSURE"))
    map_fy = st.sidebar.selectbox("Lateral Force (FY) Column", col_options, index=col_options.index("LATERAL_FORCE"))
    map_fx = st.sidebar.selectbox("Longitudinal Force (FX) Column", col_options, index=col_options.index("LONGITUDINAL_FORCE"))
    
    # --- Fitting Controls ---
    st.sidebar.header("3. Run Fitter")
    fit_button = st.sidebar.button("Fit PAC2002 Model")

    # --- Main App Logic ---
    if fit_button:
        try:
            with st.spinner("Fitting model... This may take a moment."):
                tire_model = openTire.createmodel('PAC2002')
                fitter = openTire.createfitter(tire_model)

                # Load data with the new mappings for SR and P
                # NOTE: Pressure (P) must be in Pascals for the model.
                fitter.load_data(SA=data[map_sa].values,
                                 SR=data[map_sr].values,
                                 FY=data[map_fy].values,
                                 FX=data[map_fx].values,
                                 FZ=data[map_fz].values,
                                 P=data[map_p].values * 1000) # Convert kPa to Pa
                
                fitter.fit()
            
            st.success("✅ Fitting complete!")

            # --- Display Results ---
            st.header("📊 Fit Comparison")
            col1, col2 = st.columns(2)

            # Use mean values from the data as constant conditions for plotting curves
            fz_mean = data[map_fz].mean()
            p_mean = data[map_p].mean() * 1000 # In Pascals

            # --- Plot 1: Lateral Force ---
            with col1:
                st.subheader("Lateral Force vs. Slip Angle")
                # Filter data for pure-slip lateral conditions (low slip ratio)
                lateral_data = data[abs(data[map_sr]) < 0.01]

                # Generate fitted curve
                sa_range = np.linspace(data[map_sa].min(), data[map_sa].max(), 100)
                fy_fitted = [tire_model.solve({'SA': np.deg2rad(sa), 'FZ': fz_mean, 'P': p_mean, 'SR': 0})['FY'] for sa in sa_range]

                fig, ax = plt.subplots()
                ax.scatter(lateral_data[map_sa], lateral_data[map_fy], label='Original Data')
                ax.plot(sa_range, fy_fitted, label='Fitted Model', color='red', linewidth=2)
                ax.set_xlabel("Slip Angle (deg)")
                ax.set_ylabel("Lateral Force (N)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            # --- Plot 2: Longitudinal Force ---
            with col2:
                st.subheader("Longitudinal Force vs. Slip Ratio")
                # Filter data for pure-slip longitudinal conditions (low slip angle)
                longitudinal_data = data[abs(data[map_sa]) < 1.0]

                # Generate fitted curve
                sr_range = np.linspace(data[map_sr].min(), data[map_sr].max(), 100)
                fx_fitted = [tire_model.solve({'SR': sr, 'FZ': fz_mean, 'P': p_mean, 'SA': 0})['FX'] for sr in sr_range]

                fig, ax = plt.subplots()
                ax.scatter(longitudinal_data[map_sr], longitudinal_data[map_fx], label='Original Data')
                ax.plot(sr_range, fx_fitted, label='Fitted Model', color='red', linewidth=2)
                ax.set_xlabel("Slip Ratio")
                ax.set_ylabel("Longitudinal Force (N)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            # --- Display Fitted Parameters ---
            st.header("📄 Fitted Parameters")
            fitted_params = {p: tire_model.get_parameter(p) for p in tire_model.get_parameter_list()}
            st.json(fitted_params)

        except Exception as e:
            st.error(f"An error occurred during fitting: {e}")

else:
    st.info("Please upload a CSV file with combined slip data to begin.")
