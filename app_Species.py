
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_filename = 'best_model_random_forest.pkl'  # Replace with the actual model filename
loaded_model = joblib.load(model_filename)

# Function to encode the "species" feature
def encode_species(species):
    return label_encoder.transform([species])[0]

# Initialize a label encoder
label_encoder = LabelEncoder()

# Streamlit UI
st.title("Iris Flower Prediction")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

species_input = st.text_input("Species (e.g., 'setosa')", "setosa")

if st.button("Predict"):
    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width],
        'species': [species_input]
    })

    # Encode the species input
    user_input['species'] = user_input['species'].apply(encode_species)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(user_input)

    # Decode the species label back to its original value
    decoded_prediction = label_encoder.inverse_transform(prediction)

    st.write(f"Predicted Species: {decoded_prediction[0]}")

# Close the encoder to avoid potential issues
label_encoder = None
