import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set the page configuration
st.set_page_config(page_title="Fault Prediction App", page_icon="📊")

# Custom CSS for styling tabs, buttons, and text
font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 18px;
  font-weight: bold;
}
h1 {
  color: #007BFF;
  font-size: 32px;
  font-weight: bold;
}
.stDataFrame {
  background-color: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 10px;
}
.stUpload {
  background-color: #f8f9fa;
  border: 1px solid #007BFF;
  border-radius: 10px;
}
.predicted-class {
  background-color: #d1ecf1;  /* Light blue background for predicted class */
  color: #0c5460;  /* Dark blue text color */
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)

# Load the trained model, LabelEncoder, and StandardScaler
model = load_model('ANNmodel.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define the prediction function
def predict(features):
    # Normalize features
    df_scaled = scaler.transform(features)

    # Make predictions (probabilities for each class)
    predictions = model.predict(df_scaled)
    
    # Get the predicted class and the probability of the predicted class
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_class_probabilities = np.max(predictions, axis=1)

    # Convert class indices to class labels
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

    # Create a DataFrame with the predicted labels and probabilities
    result_df = pd.DataFrame({
        'Predicted Class': predicted_labels,
        'Probability': predicted_class_probabilities
    })

    return result_df

# Streamlit app starts here
st.title('Fault Prediction App 📊')

# File uploader for CSV or Excel
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'], key="fileUploader")

if uploaded_file:
    try:
        # Handle CSV and Excel files
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        st.write("### Uploaded Data Preview:")
        if "faults" in data.columns:
          data1 = data.drop("faults",axis = 1)
        st.dataframe(data1.head(), width=3000)  # Show a preview of the uploaded data
        
        # Select only relevant columns for prediction
        features = data[['Ia', 'Van', 'Vdc', 'Io', 'Ibat', 'Vbat', 'SOC', 'Temperature', 'Ia.1', 'Ib', 'Ic', 'Vsd']]
        
        # Make predictions
        prediction_df = predict(features)
        
        # Show the combined original data and predictions
        result = pd.concat([data, prediction_df], axis=1)
        st.write("### Prediction Results:")
        st.dataframe(result, width=3000)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
