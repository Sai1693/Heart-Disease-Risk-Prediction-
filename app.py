import streamlit as st

from predict import predict_heart_risk



st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️"
)



st.title(
    "❤️ Heart Disease Risk Prediction"
)


st.write(
    "Machine Learning based prediction system to estimate heart disease risk."
)



age = st.number_input(
    "Age",
    1,
    100
)


sex = st.selectbox(
    "Gender",
    ["Male","Female"]
)


cp = st.number_input(
    "Chest Pain Type",
    0,
    3
)


trestbps = st.number_input(
    "Resting Blood Pressure"
)


chol = st.number_input(
    "Cholesterol Level"
)


fbs = st.number_input(
    "Fasting Blood Sugar",
    0,
    1
)


restecg = st.number_input(
    "Rest ECG",
    0,
    2
)


thalach = st.number_input(
    "Maximum Heart Rate"
)


exang = st.number_input(
    "Exercise Induced Angina",
    0,
    1
)


oldpeak = st.number_input(
    "ST Depression"
)


slope = st.number_input(
    "Slope",
    0,
    2
)


ca = st.number_input(
    "Major Vessels",
    0,
    4
)


thal = st.number_input(
    "Thalassemia",
    0,
    3
)



gender = 1 if sex=="Male" else 0



if st.button("Predict Risk"):


    input_data = [
        age,
        gender,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]


    prediction, probability = predict_heart_risk(
        input_data
    )


    if prediction == 1:

        st.error(
            f"High Risk of Heart Disease\nRisk Probability: {probability*100:.2f}%"
        )

    else:

        st.success(
            f"Low Risk of Heart Disease\nRisk Probability: {probability*100:.2f}%"
        )
