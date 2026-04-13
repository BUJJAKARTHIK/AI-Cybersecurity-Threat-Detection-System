import streamlit as st
import pandas as pd
from src.preprocessing import load_data, preprocess_data
from src.model import train_model

st.title("🔐 AI Cybersecurity Threat Detection System")

data = load_data("data/KDDTrain+.txt")

st.write("### Dataset Preview")
st.write(data.head())

X_train, X_test, y_train, y_test = preprocess_data(data)

model = train_model(X_train, y_train)

st.success("Model Trained Successfully")

st.write("### Sample Predictions")

pred = model.predict(X_test[:10])
st.write(pred)