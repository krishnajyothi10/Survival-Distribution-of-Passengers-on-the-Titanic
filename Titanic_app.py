import streamlit as st
import numpy as np
import joblib

model = joblib.load('model_titanic.pkl')

st.markdown("<h1 style='color:blue;'>Titanic Survival Predictor</h1>", unsafe_allow_html=True)

st.image("https://www.currentaffairs.org/hubfs/Imported_Blog_Media/titanicii-1024x646-1.jpg",
         caption="TITANIC SURVIVAL RATE",
         width=200,
         use_container_width=False)

# Numerical Inputs
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

# Categorical Inputs
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
pclass = st.selectbox("Pclass", [1, 2, 3])

# Handle get_dummies manually
male = 1 if sex == "male" else 0
Q = 1 if embarked == "Q" else 0
S = 1 if embarked == "S" else 0
class_2 = 1 if pclass == 2 else 0
class_3 = 1 if pclass == 3 else 0

# Input order must match model training
input_data = np.array([[age, sibsp, parch, fare,
                        male, Q, S, class_2, class_3]])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.markdown(
            "<div style='background-color:#d4edda; border:4px solid #ffe600; box-shadow:0 0 10px #28a745; border-radius:12px; padding:24px; color:#155724; font-size:22px; text-align:center; margin-top:20px;'>"
            "🎉 <b>Prediction: Survived</b> 🎉"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background-color:#f8d7da; border:4px solid #ffe600; box-shadow:0 0 10px #dc3545; border-radius:12px; padding:24px; color:#721c24; font-size:22px; text-align:center; margin-top:20px;'>"
            "😢 <b>Prediction: Did not survive</b> 😢"
            "</div>", unsafe_allow_html=True)