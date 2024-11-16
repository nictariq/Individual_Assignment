import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model, scaler, and selected features .pkl files
with open('breastcancer_classifier_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('selected_features.pkl', 'rb') as feature_file:
    selected_features = pickle.load(feature_file)

# Streamlit web app title and a brief description
st.title("Breast Cancer Diagnosis Prediction")
st.write("This app uses a trained machine learning model to predict whether a breast cancer case is likely **Benign** or **Malignant** based on selected features.")

# Sidebar for users to input values for the selected features
st.sidebar.header("Input Features")
st.sidebar.write("Enter values for the following selected features:")

# Collect user input for each of the selected features
user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0)

# Ensure that the input DataFrame has the exact same column order as selected_features
input_df = pd.DataFrame([user_input])

# Reindex the DataFrame to ensure columns are in the correct order
input_df = input_df.reindex(columns=selected_features)

# Scale the input data using the loaded scaler to transform input data
input_scaled = scaler.transform(input_df)

# Prediction and result display
if st.button("Classify"):
    prediction = model.predict(input_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    
    # Display the prediction result
    st.subheader("Prediction Result")
    st.markdown(f"<h2 style='color: #4CAF50;'>Prediction: {result}</h2>", unsafe_allow_html=True)

    # Display further information on interpretation of results
    if result == "Malignant":
        st.write("**Malignant**: This result suggests the presence of malignant cells. Please consult a healthcare professional for further evaluation.")
    else:
        st.write("**Benign**: This result suggests the cells are likely benign. Routine follow-up may be recommended.")

# Footer note
st.write("---")
st.write("This app is intended for informational purposes only and should not replace medical advice.")

# Display scaled input data (for download if required)
st.subheader("Scaled Input Features")
st.write(pd.DataFrame(input_scaled, columns=selected_features))