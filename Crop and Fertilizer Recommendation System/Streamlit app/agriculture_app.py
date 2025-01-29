import streamlit as st
import numpy as np
import pickle
import pandas as pd
import joblib

# Load the saved models and encoders
crop_model = pickle.load(open('model.pkl', 'rb'))
crop_sc = pickle.load(open('standscaler.pkl', 'rb'))
crop_mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

fertilizer_model = joblib.load('fertilizer_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
le_fertilizer = joblib.load('fertilizer_label_encoder.pkl')

# Custom CSS and HTML for styling
st.markdown("""
    <style>
        /* Background Image */
        body {
            background: url('https://source.unsplash.com/1600x900/?farm,greenery,agriculture') no-repeat center center fixed;
            background-size: cover;
            color: #000000;
        }

        /* Title Styling */
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #2e4053;
            text-shadow: 3px 3px 7px #d6eaf8;
            margin-bottom: 20px;
        }

        /* Card Styling */
        .card {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            padding: 20px;
            margin: 20px auto;
            max-width: 600px;
            text-align: center;
        }

        /* Button Styling */
        button {
            background: #1e8449;
            border: none;
            color: white;
            font-size: 1em;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        button:hover {
            background: #145a32;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">🌾 Agriculture Prediction App 🌾</div>', unsafe_allow_html=True)

# Main App
st.write("Choose an option to predict the crop or fertilizer.")

# Radio button for selection
option = st.radio("Select Prediction Type", ("Crop Prediction", "Fertilizer Prediction"))

if option == "Crop Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Provide input details to predict the best crop to be cultivated.")

    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=10.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, value=10.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=10.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if st.button("Predict Best Crop"):
        # Prepare the input data
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale the input features
        mx_features = crop_mx.transform(single_pred)
        sc_mx_features = crop_sc.transform(mx_features)

        # Make the prediction
        prediction = crop_model.predict(sc_mx_features)

        # Display the result
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        st.success(result)
    st.markdown('</div>', unsafe_allow_html=True)

elif option == "Fertilizer Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Provide input details to predict the recommended fertilizer.")

    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=40.0)

    soil_type = st.selectbox("Soil Type", label_encoders['Soil Type'].classes_)
    crop_type = st.selectbox("Crop Type", label_encoders['Crop Type'].classes_)

    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=20.0)
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=10.0)
    phosphorous = st.number_input("Phosphorous (P)", min_value=0.0, max_value=100.0, value=15.0)

    if st.button("Predict Fertilizer"):
        # Prepare the input data
        input_data = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Soil Type': label_encoders['Soil Type'].transform([soil_type])[0],
            'Crop Type': label_encoders['Crop Type'].transform([crop_type])[0],
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorous
        }

        # Create DataFrame with correct column order
        training_columns = fertilizer_model.feature_names_in_
        input_df = pd.DataFrame([input_data], columns=training_columns)

        # Add missing columns with default values (if any)
        for column in training_columns:
            if column not in input_df.columns:
                input_df[column] = 0

        # Ensure column order matches the training model
        input_df = input_df[training_columns]

        # Predict the fertilizer
        predicted_class = fertilizer_model.predict(input_df)[0]
        predicted_fertilizer = le_fertilizer.inverse_transform([predicted_class])[0]

        # Display the prediction
        st.success(f"The recommended fertilizer is: **{predicted_fertilizer}**")
    st.markdown('</div>', unsafe_allow_html=True)
