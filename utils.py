import joblib
import numpy as np
import pandas as pd

def load_models():
    model = joblib.load('model/sleep_predictor.pkl')
    scaler = joblib.load('model/scaler.pkl')
    kmeans = joblib.load('model/cluster_model.pkl')
    scaler_cluster = joblib.load('model/scaler_cluster.pkl')
    le_gender = joblib.load('model/le_gender.pkl')
    le_bmi = joblib.load('model/le_bmi.pkl')
    le_occupation = joblib.load('model/le_occupation.pkl')
    return model, scaler, kmeans, scaler_cluster, le_gender, le_bmi, le_occupation

def preprocess_input(data, scaler, le_gender, le_bmi, le_occupation):
    df = pd.DataFrame([data])

    # Handle unseen Gender
    gender_value = data['Gender']
    if gender_value not in le_gender.classes_:
        gender_value = le_gender.classes_[0]
    df['Gender'] = le_gender.transform([gender_value])

    # Handle unseen BMI Category
    bmi_value = data['BMI Category']
    if bmi_value not in le_bmi.classes_:
        bmi_value = le_bmi.classes_[0]
    df['BMI Category'] = le_bmi.transform([bmi_value])

    # Handle unseen Occupation
    occupation_value = data['Occupation']
    if occupation_value not in le_occupation.classes_:
        occupation_value = le_occupation.classes_[0]
    df['Occupation'] = le_occupation.transform([occupation_value])

    # Feature selection
    features = ['Age', 'Gender', 'Occupation', 'Sleep_Duration',
                'Physical_Activity_Level', 'Stress Level',
                'BMI Category', 'Heart Rate', 'Daily Steps']

    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled



def preprocess_cluster_input(input_dict, scaler_cluster):
    cluster_features = ['Sleep_Duration', 'Physical_Activity_Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
    values = np.array([input_dict[feat] for feat in cluster_features]).reshape(1, -1)
    scaled_values = scaler_cluster.transform(values)
    return scaled_values
