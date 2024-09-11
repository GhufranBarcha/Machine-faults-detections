# Fault Prediction App

link: https://machine-faults-detections.streamlit.app/

## Overview

This web app performs fault prediction using a pre-trained Artificial Neural Network (ANN) model. Built with Streamlit, it allows users to upload data and get predictions about faults based on the provided features.

## Features

- **File Upload**: Upload CSV or Excel files containing your data.
- **Data Preview**: View a preview of the uploaded data.
- **Prediction**: Get predictions for faults based on the uploaded data.
- **Results Display**: View the original data alongside the prediction results.

## How to Use

1. **Upload Your Data**:
   - Click the "Upload your CSV or Excel file" button to upload your data file. The file can be in CSV or Excel format.

2. **Preview Data**:
   - After uploading, a preview of your data (excluding the 'faults' column) will be displayed.

3. **Get Predictions**:
   - The app will automatically process the uploaded data and display the prediction results, including the predicted class and its probability for each entry.

4. **View Results**:
   - The results will be shown with the original data and the predictions side-by-side.

## Setup

To run the app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/GhufranBarcha/Machine-faults-detections
   cd Machine-faults-detections
