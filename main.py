import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib

# 1️⃣ Load data
df = pd.read_csv('data\Sleep_health_and_lifestyle_dataset.csv')
df = df.dropna().reset_index(drop=True)

# 2️⃣ Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_bmi = LabelEncoder()
df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
le_occupation = LabelEncoder()
df['Occupation'] = le_occupation.fit_transform(df['Occupation'])

# 3️⃣ Supervised ML: Predict Quality_of_Sleep
features = ['Age', 'Gender', 'Occupation', 'Sleep_Duration', 'Physical_Activity_Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']
target = 'Quality_of_Sleep'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'model/sleep_predictor.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_gender, 'model/le_gender.pkl')
joblib.dump(le_bmi, 'model/le_bmi.pkl')
joblib.dump(le_occupation, 'model/le_occupation.pkl')

# 4️⃣ Unsupervised ML: KMeans Clustering
clustering_features = ['Sleep_Duration', 'Physical_Activity_Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
X_cluster = df[clustering_features]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

joblib.dump(kmeans, 'model/cluster_model.pkl')
joblib.dump(scaler_cluster, 'model/scaler_cluster.pkl')

print("✅ Models trained and saved successfully.")