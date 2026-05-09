import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/model.pkl")

st.title("🔐 IDS System")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Detect"):
    data = np.array([f1, f2]).reshape(1, -1)
    result = model.predict(data)

    if result[0] == 0:
        st.success("Normal Traffic")
    else:
        st.error("Threat Detected!")