import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- App Configuration ---
st.set_page_config(layout="wide")

# --- The Pacejka Magic Formula Function ---
def magic_formula(x, B, C, D, E):
    """
    Implements the Pacejka Magic Formula for lateral force.
    x: Slip angle in degrees
    B, C, D, E: Model coefficients
    """
    x_rad = np.deg2rad(x)
    b_x = B * x_rad
    return D * np.sin(C * np.arctan(b_x - E * (b_x - np.arctan(b_x))))

# --- App Title and Description ---
st.title("Magic Formula Tire Model Fitter 🚗")
st.markdown("""
This app fits a tire's lateral force data to the Pacejka Magic Formula using SciPy. 
Upload a CSV file with slip angle and lateral force data to get started.
""")

# --- File Uploader and Sidebar ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Robustly read the CSV file
        data = pd.read_csv(uploaded_file, sep=',', engine='python', skipinitialspace=True)
        
        st.sidebar.success("File uploaded successfully!")
        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # --- Column Mapping ---
        st.sidebar.header("2. Map Data Columns")
        col_options = list(data.columns)
        
        map_sa = st.sidebar.selectbox("Slip Angle (SA) Column", col_options)
        map_fy = st.sidebar.selectbox("Lateral Force (FY) Column", col_options, index=min(1, len(col_options)-1))
        
        # --- Fitting Controls ---
        st.sidebar.header("3. Run Fitter")
        fit_button = st.sidebar.button("Fit Magic Formula Model")

        # --- Main App Logic ---
        if fit_button:
            try:
                with st.spinner("Finding the magic... ✨"):
                    x_data = pd.to_numeric(data[map_sa])
                    y_data = pd.to_numeric(data[map_fy])
                    
                    initial_guesses = [10, 1.9, max(abs(y_data)), 0.97]
                    
                    popt, pcov = curve_fit(magic_formula, x_data, y_data, p0=initial_guesses)
                
                st.success("✅ Fitting complete!")

                # --- Display Results ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("📊 Fit Comparison")
                    sa_range = np.linspace(x_data.min(), x_data.max(), 200)
                    fy_fitted = magic_formula(sa_range, *popt)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(x_data, y_data, label='Original Data', color='blue')
                    ax.plot(sa_range, fy_fitted, label='Fitted Magic Formula', color='red', linewidth=3)
                    ax.set_xlabel("Slip Angle (degrees)")
                    ax.set_ylabel("Lateral Force (N)")
                    ax.set_title("Original Data vs. Fitted Model")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                with col2:
                    st.subheader("📄 Fitted Parameters")
                    st.metric(label="B (Stiffness Factor)", value=f"{popt[0]:.4f}")
                    st.metric(label="C (Shape Factor)", value=f"{popt[1]:.4f}")
                    st.metric(label="D (Peak Factor)", value=f"{popt[2]:.2f}")
                    st.metric(label="E (Curvature Factor)", value=f"{popt[3]:.4f}")
                    
                    st.info("Formula: Y(x) = D·sin(C·arctan(B·x - E·(B·x - arctan(B·x))))")

            except Exception as e:
                st.error(f"An error occurred during fitting: {e}")
                st.warning("Tip: Make sure the selected columns contain only numeric data and that your CSV is formatted correctly.")
    
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.warning("Please ensure your file is a standard comma-separated CSV.")

else:
    st.info("Please upload a CSV file to begin.")
