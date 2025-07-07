import streamlit as st
from utils import load_models, preprocess_input, preprocess_cluster_input
import pandas as pd
from datetime import datetime
import os

st.title("🛌 Sleep Insight Engine")
st.write("Predict your sleep quality and find your lifestyle cluster using ML.")

# Load models and encoders
model, scaler, kmeans, scaler_cluster, le_gender, le_bmi, le_occupation = load_models()

with st.form("sleep_form"):
    st.header("Enter Your Daily Lifestyle Data")

    age = st.number_input("Age", 10, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.text_input("Occupation (e.g., Student, Developer, Designer, etc.)", "Student")
    sleep_duration = st.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.0, 0.1)
    physical_activity = st.number_input("Physical Activity Level (0-100)", 0, 100, 50)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_category = st.text_input("BMI Category (e.g., Normal, Overweight, Obese)", "Normal")
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
    daily_steps = st.number_input("Daily Steps", 1000, 20000, 5000)

    submitted = st.form_submit_button("Predict and Save")

if submitted:
    # Prepare user input
    user_input = {
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Sleep_Duration": sleep_duration,
        "Physical_Activity_Level": physical_activity,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps
    }

    # Preprocess for supervised prediction
    processed = preprocess_input(user_input, scaler, le_gender, le_bmi, le_occupation)
    prediction = model.predict(processed)[0]

    # Preprocess for clustering
    cluster_processed = preprocess_cluster_input(user_input, scaler_cluster)
    cluster = kmeans.predict(cluster_processed)[0]

    # Display results
    st.success(f"🛌 **Predicted Sleep Quality Score:** {prediction:.2f} (out of 10)")
    st.info(f"🔍 **Lifestyle Cluster:** Cluster {cluster}")

    # Append data + prediction to CSV
    user_input_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Sleep_Duration": sleep_duration,
        "Physical_Activity_Level": physical_activity,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Quality_of_Sleep": prediction,
        "Cluster": cluster
    }

    df_new = pd.DataFrame([user_input_record])

    try:
        df_existing = pd.read_csv('data/sleep_dataset.csv')
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv('data/sleep_dataset.csv', index=False)
    except FileNotFoundError:
        df_new.to_csv('data/sleep_dataset.csv', index=False)

    st.success("✅ Your data has been saved to the dataset for future model improvements.")

# Retrain button for live retraining
if st.button("🔄 Retrain Model with Latest Data"):
    os.system("python main.py")
    st.success("✅ Models retrained successfully with the updated dataset.")
