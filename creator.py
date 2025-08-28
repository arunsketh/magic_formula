import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opentire # Import the main module

# --- App Configuration ---
st.set_page_config(layout="wide")

# --- Initialize OpenTire (cached for performance) ---
@st.cache_resource
def init_opentire():
    """Initializes the OpenTire library instance."""
    return opentire.OpenTire() # Access the class from the module

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

    # Helper function to find a column index safely, defaulting to 0 if not found
    def get_col_index(name, options):
        try:
            return options.index(name)
        except ValueError:
            return 0

    # Map all required inputs and outputs using the safe index function
    map_sa = st.sidebar.selectbox(
        "Slip Angle (SA) Column", col_options, index=get_col_index("SLIP_ANGLE", col_options)
    )
    map_sr = st.sidebar.selectbox(
        "Slip Ratio (SR) Column", col_options, index=get_col_index("SLIP_RATIO", col_options)
    )
    map_fz = st.sidebar.selectbox(
        "Normal Load (FZ) Column", col_options, index=get_col_index("NORMAL_LOAD", col_options)
    )
    map_p = st.sidebar.selectbox(
        "Inflation Pressure (P) Column", col_options, index=get_col_index("INFLATION_PRESSURE", col_options)
    )
    map_fy = st.sidebar.selectbox(
        "Lateral Force (FY) Column", col_options, index=get_col_index("LATERAL_FORCE", col_options)
    )
    map_fx = st.sidebar.selectbox(
        "Longitudinal Force (FX) Column", col_options, index=get_col_index("LONGITUDINAL_FORCE", col_options)
    )
    
    # --- Fitting Controls ---
    st.sidebar.header("3. Run Fitter")
    fit_button = st.sidebar.button("Fit PAC2022 Model")

    # --- Main App Logic ---
    if fit_button:
        try:
            with st.spinner("Fitting model... This may take a moment."):
                # 1. Create the model and correctly instantiate the Fitter
                tire_model = openTire.createmodel('PAC2002')
                fitter = opentire.Fitter(tire_model) # Access Fitter from the module

                # 2. Load data, converting pressure from kPa (if needed) to Pascals
                fitter.load_data(SA=data[map_sa].values,
                                 SR=data[map_sr].values,
                                 FY=data[map_fy].values,
                                 FX=data[map_fx].values,
                                 FZ=data[map_fz].values,
                                 P=data[map_p].values * 1000)
                
                # 3. Run the fitting process
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
                lateral_data = data[abs(data[map_sr]) < 0.01]
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
                longitudinal_data = data[abs(data[map_sa]) < 1.0]
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
