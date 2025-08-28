import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from opentire import OpenTire
from opentire.Core import TireState

# Initialize OpenTire
@st.cache_resource
def init_opentire():
    return OpenTire()

openTire = init_opentire()

# --- Streamlit App Layout ---

st.title("OpenTirePython Streamlit Explorer 🚗")

st.markdown("""
This application allows you to interact with tire models from the OpenTirePython library.
Adjust the parameters in the sidebar to see how they affect the tire's performance curves.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Tire and Simulation Parameters")

# Select Tire Model
tire_model_name = st.sidebar.selectbox(
    "Select Tire Model",
    ['PAC2002'] # Add other models as you implement them
)

# Simulation Parameters
st.sidebar.subheader("Simulation Conditions")
normal_load = st.sidebar.slider("Normal Load (FZ) [N]", 500, 5000, 1500, 100)
inclination_angle = st.sidebar.slider("Inclination Angle (IA) [deg]", -5.0, 5.0, 0.0, 0.5)
velocity = st.sidebar.slider("Velocity (V) [m/s]", 1.0, 50.0, 10.0, 1.0)
pressure = st.sidebar.slider("Tire Pressure (P) [kPa]", 180, 320, 260, 10)


# --- Core Logic to Generate Curves ---

def generate_tire_curves(tire_model, fz, ia, v, p):
    """
    Generates tire performance curves for a range of slip angles and slip ratios.
    """
    state = TireState()
    state['FZ'] = fz
    state['IA'] = np.deg2rad(ia)  # Convert to radians
    state['V'] = v
    state['P'] = p * 1000 # Convert to Pascals

    # 1. Lateral Force vs. Slip Angle (FY vs SA)
    sa_range = np.linspace(-15, 15, 100)
    fy_values = []
    for sa_deg in sa_range:
        state['SA'] = np.deg2rad(sa_deg)
        state['SR'] = 0.0
        tire_model.solve(state)
        fy_values.append(state['FY'])

    # 2. Longitudinal Force vs. Slip Ratio (FX vs SR)
    sr_range = np.linspace(-0.2, 0.2, 100)
    fx_values = []
    for sr in sr_range:
        state['SA'] = 0.0
        state['SR'] = sr
        tire_model.solve(state)
        fx_values.append(state['FX'])


    return sa_range, fy_values, sr_range, fx_values


# --- Main Application Area ---

if st.button("Generate Tire Curves"):
    # Create the tire model
    myTireModel = openTire.createmodel(tire_model_name)

    # Generate the data
    sa, fy, sr, fx = generate_tire_curves(myTireModel, normal_load, inclination_angle, velocity, pressure)

    # Display plots
    st.header("Tire Performance Curves")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lateral Force vs. Slip Angle")
        fig, ax = plt.subplots()
        ax.plot(sa, fy)
        ax.set_xlabel("Slip Angle (deg)")
        ax.set_ylabel("Lateral Force (FY) [N]")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("Longitudinal Force vs. Slip Ratio")
        fig, ax = plt.subplots()
        ax.plot(sr, fx)
        ax.set_xlabel("Slip Ratio (SR)")
        ax.set_ylabel("Longitudinal Force (FX) [N]")
        ax.grid(True)
        st.pyplot(fig)


st.info("Click the 'Generate Tire Curves' button to update the plots with the selected parameters.")
