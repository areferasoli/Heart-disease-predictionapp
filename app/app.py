import streamlit as st
import numpy as np
import joblib

# ✅ CSS برای دکمه بزرگ و sticky
# ✅ CSS دکمه وسط + سایز ثابت
st.markdown("""
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    div.stButton > button {
        width: 300px;   /* سایز ثابت */
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        background-color: #ff4b4b;
        color: white;
    }

    div.stButton > button:hover {
        background-color: #ff2b2b;
    }
    </style>
""", unsafe_allow_html=True)


# ✅ load model
model = joblib.load("heart_model.pkl")




col1, col2 = st.columns([1, 4])  # نسبت جدید

with col1:
    st.image("unibo_logo.png", width=155)

with col2:
    st.markdown("""
    ### ❤️ Heart Disease Prediction  
    **Arefe Rasouli Javazm**  
    *Artificial Intelligence for Medicine*  
    **Prof. Stefano Diciotti**
    """)


# ✅ input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.slider("Chest Pain Type (cp)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.slider("Resting ECG", 0, 2)
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)")
slope = st.slider("Slope", 0, 2)
ca = st.slider("Number of vessels (ca)", 0, 3)
thal = st.slider("Thal", 0, 2)


# ✅ predict
if st.button("🚀 Predict Heart Risk"):

    X = np.array([[age, sex, cp, trestbps, chol, fbs,
                   restecg, thalach, exang, oldpeak,
                   slope, ca, thal]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.write("---")

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ No Heart Disease\n\nProbability: {probability:.2f}")