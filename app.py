import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Sleep Analysis System",
    page_icon="🌙",
    layout="centered"
)

# ---- LOAD MODELS (with error handling) ----


@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/xgboost_model.pkl")
        le = joblib.load("models/label_encoder.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, le, kmeans, scaler
    except FileNotFoundError as e:
        st.error(f"⛔ Model file not found: {e}\n\n"
                 "Please run `proje_uyku_english.ipynb` first to generate the model files.")
        st.stop()


model, le, kmeans, scaler = load_models()

# ---- HEADER ----
st.markdown(
    "<h1 style='text-align:center; color:#4A90E2;'>🩺 AI Sleep Health & Early Warning System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px; color:#7F8C8D;'>"
    "AI-powered clinical decision support tool. Please enter patient information below.</p>",
    unsafe_allow_html=True
)
st.info("💡 **Tip:** Fill in the fields and click the analysis button at the bottom.")

# ---- DEMOGRAPHICS ----
st.markdown("### 📋 Patient Demographics & Lifestyle")
with st.container(border=True):
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    age = st.number_input("📅 Age", min_value=18,
                          max_value=100, value=35, step=1)
    occupation = st.selectbox("💼 Occupation", [
        "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
        "Accountant", "Salesperson", "Scientist",
        "Sales Representative", "Software Engineer", "Teacher"
    ])

# ---- SLEEP & STRESS ----
st.markdown("### ⏱️ Sleep & Stress Metrics")
with st.container(border=True):
    sleep_duration = st.number_input(
        "⏳ Daily Sleep Duration (Hours)",           min_value=1.0, max_value=24.0, value=7.0, step=0.5)
    quality_of_sleep = st.number_input(
        "⭐ Sleep Quality Score (1=Very Poor, 10=Excellent)", min_value=1, max_value=10, value=7, step=1)
    physical_activity = st.number_input(
        "🏃 Daily Exercise Duration (Minutes)",      min_value=0, max_value=120, value=50, step=5)
    stress_level = st.number_input(
        "🔥 Daily Stress Level (1=Very Low, 10=Extreme)", min_value=1, max_value=10, value=5, step=1)

# ---- VITALS ----
st.markdown("### 🫀 Vital Signs & Physical Status")
with st.container(border=True):
    heart_rate = st.number_input(
        "💓 Heart Rate (BPM)",          min_value=40, max_value=150, value=70, step=1)
    daily_steps = st.number_input(
        "👣 Daily Steps",               min_value=0, max_value=30000, value=7000, step=500)
    st.markdown("<p style='color:#E74C3C; font-weight:bold;'>🩸 Blood Pressure:</p>",
                unsafe_allow_html=True)
    systolic_bp = st.number_input(
        "   ➔ Systolic BP",            min_value=80, max_value=200, value=120, step=5)
    diastolic_bp = st.number_input(
        "   ➔ Diastolic BP",           min_value=50, max_value=130, value=80, step=5)
    bmi = st.selectbox("⚖️ BMI Category", [
                       "Normal", "Normal Weight", "Overweight", "Obese"])

st.markdown("<br>", unsafe_allow_html=True)

# ---- ANALYSIS BUTTON ----
if st.button("🚀 START CLINICAL ANALYSIS", use_container_width=True):

    # -- Predict cluster dynamically --
    bmi_map = {"Normal": 0, "Normal Weight": 1, "Overweight": 2, "Obese": 3}
    gender_enc = 1 if gender == "Male" else 0
    occupation_map = {o: i for i, o in enumerate(sorted([
        "Accountant", "Doctor", "Engineer", "Lawyer", "Manager",
        "Nurse", "Sales Representative", "Salesperson",
        "Scientist", "Software Engineer", "Teacher"
    ]))}

    X_for_cluster = np.array([[
        gender_enc, age, occupation_map.get(occupation, 0),
        sleep_duration, quality_of_sleep, physical_activity,
        stress_level, heart_rate, daily_steps,
        systolic_bp, diastolic_bp, bmi_map.get(bmi, 0)
    ]])
    X_scaled = scaler.transform(X_for_cluster)
    cluster_pred = int(kmeans.predict(X_scaled)[0])

    # -- Build input DataFrame (same column order as training) --
    input_data = pd.DataFrame([[
        gender, age, occupation, sleep_duration, quality_of_sleep,
        physical_activity, stress_level, heart_rate, daily_steps,
        systolic_bp, diastolic_bp, bmi, cluster_pred
    ]], columns=[
        "Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps",
        "Systolic_BP", "Diastolic_BP", "BMI Category", "Cluster"
    ])

    with st.spinner("AI model is analyzing the data..."):
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]
        result = le.inverse_transform(prediction)[0]

    # -- Confidence scores --
    st.markdown("### 📊 Model Confidence Scores")
    proba_df = pd.DataFrame({
        "Diagnosis": le.classes_,
        "Probability (%)": (proba * 100).round(1)
    }).sort_values("Probability (%)", ascending=False)
    st.dataframe(proba_df, use_container_width=True, hide_index=True)

    # -- Cluster info --
    cluster_labels = {
        0: "😊 Healthy Sleeper",
        1: "🚨 High Risk (Stressed & Low Sleep)",
        2: "⚠️ Moderate Risk (Sedentary & Stressed)",
        3: "🧓 Elderly / High Blood Pressure Profile"
    }
    st.info(
        f"🔍 **Sleep Profile Cluster:** {cluster_labels.get(cluster_pred, str(cluster_pred))}")

    # -- Main result --
    st.markdown("### 📢 Diagnosis & AI Report")
    if result == "None":
        st.balloons()
        st.success("""
        ### 🎉 RESULT: HEALTHY PROFILE
        **Findings:** Based on the entered data, no clinical sleep disorder risk has been detected.
        Overall lifestyle habits and vital signs appear balanced.
        """)
    elif result == "Insomnia":
        st.warning("""
        ### ⚠️ RESULT: INSOMNIA RISK
        **Findings:** The AI model has detected a high likelihood of **Insomnia**.
        Reducing stress levels and improving sleep hygiene are strongly recommended.
        """)
    else:
        st.error("""
        ### 🚨 RESULT: SLEEP APNEA RISK
        **Findings:** The patient's vital signs and physical condition indicate a high risk of **Sleep Apnea**.
        Consulting an ENT specialist or a Sleep Laboratory as soon as possible is strongly advised.
        """)
