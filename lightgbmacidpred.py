import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    feature_order = joblib.load('feature_order.pkl')
    return model, feature_order

model, feature_order = load_model()

# -----------------------------
# App Title and Description
# -----------------------------
st.title("Acid Concentration Prediction")

st.markdown("""
Predict **acid concentration (g/L)** using:
- **Temperature (°C)**
- **Conductivity (µS/cm)**
""")

# -----------------------------
# User Inputs
# -----------------------------
temperature = st.number_input(
    "Enter Temperature (°C):",
    min_value=0.0,
    step=0.1,
    format="%.1f"
)

conductivity = st.number_input(
    "Enter Conductivity (µS/cm):",
    min_value=0.0,
    step=1.0,
    format="%.0f"
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Acid Concentration"):
    if conductivity <= 0:
        st.warning("Please enter a valid conductivity value.")
    else:
        # Prepare input exactly as trained
        input_df = pd.DataFrame(
            [[conductivity, temperature]],
            columns=feature_order
        )

        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted H₂SO₄ Concentration: **{prediction:.2f} g/L**"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---

""")
