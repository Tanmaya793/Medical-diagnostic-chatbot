# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load pre-trained model and symptom classes (or retrain quickly here)
@st.cache_resource
def load_model():
    df = pd.read_csv("enhanced_rural_diseases.csv")

    symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
    for col in symptom_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")
    df[symptom_cols] = df[symptom_cols].replace("nan", pd.NA)

    df["All_Symptoms"] = df[symptom_cols].values.tolist()
    df["All_Symptoms"] = df["All_Symptoms"].apply(lambda x: list(filter(pd.notna, x)))

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["All_Symptoms"])
    y = df["Disease"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf, mlb.classes_

model, symptom_classes = load_model()

# --- Prediction function ---
def predict_disease(user_symptoms):
    cleaned = [s.strip().lower().replace(" ", "_") for s in user_symptoms]
    input_vector = [1 if sym in cleaned else 0 for sym in symptom_classes]
    prediction = model.predict([input_vector])[0]
    return prediction

# --- Streamlit UI ---
st.title("💬 Symptom Checker Chatbot")
st.write("Describe your symptoms (Note: try to give more than 2-3 symptoms for better analysis), and I’ll tell you the likely disease.")

user_input = st.text_input("👤 You:", placeholder="I have headache and joint pain")

if user_input:
    # Extract symptoms from text (basic split for now)
    symptom_input = [word.strip() for word in user_input.lower().replace(".", "").split(" and ")]

    prediction = predict_disease(symptom_input)
    st.markdown(f"🤖 **You may have: _{prediction}_**")

    st.caption("Note: This is not a medical diagnosis. Always consult a doctor.")
