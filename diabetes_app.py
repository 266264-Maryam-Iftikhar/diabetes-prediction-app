import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the saved scaler and model
scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Check if the scaler is loaded correctly
print(f"Scaler type: {type(scaler)}")

if isinstance(scaler, MinMaxScaler):
    print("Scaler loaded correctly!")
else:
    print("Scaler is not loaded correctly, it is of type:", type(scaler))

# Streamlit Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="💉",
    layout="centered"
)

# Header and Subheader
st.title("💉 **Diabetes Prediction App**")
st.markdown("""
    **Welcome!** Enter your details below to check if you are at risk for diabetes. 
    The model will predict whether you might have diabetes based on your inputs.
""")

# Input Fields (Group them logically)
st.header("📝 **Patient Information**")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Enter your age.")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
    polyuria = st.selectbox("Polyuria (Excessive Urination)", ["No", "Yes"], help="Do you experience excessive urination?")
    polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["No", "Yes"], help="Do you experience excessive thirst?")
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"], help="Have you experienced sudden weight loss?")
    weakness = st.selectbox("Weakness", ["No", "Yes"], help="Do you feel weakness?")
    
with col2:
    polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["No", "Yes"], help="Do you feel excessive hunger?")
    genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"], help="Do you have genital thrush?")
    visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"], help="Do you experience blurring of vision?")
    itching = st.selectbox("Itching", ["No", "Yes"], help="Do you experience itching?")
    irritability = st.selectbox("Irritability", ["No", "Yes"], help="Do you experience irritability?")
    delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"], help="Do you experience delayed healing?")
    
# Symptoms continued
st.header("⚠️ **Other Symptoms**")

col1, col2 = st.columns(2)

with col1:
    partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"], help="Do you have partial paresis (partial loss of movement)?")
    muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"], help="Do you experience muscle stiffness?")
    
with col2:
    alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"], help="Do you experience hair loss?")
    obesity = st.selectbox("Obesity", ["No", "Yes"], help="Are you overweight or obese?")

# Convert categorical inputs to numerical values
def convert_input(value):
    return 1 if value == "Yes" else 0

def convert_gender(value):
    return 1 if value == "Male" else 0

# Convert input into the format that matches the training data
numerical_cols = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 
                  'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring', 
                  'itching', 'irritability', 'delayed_healing', 'partial_paresis', 
                  'muscle_stiffness', 'alopecia', 'obesity']

input_data = np.array([
    age, convert_gender(gender), convert_input(polyuria), convert_input(polydipsia),
    convert_input(sudden_weight_loss), convert_input(weakness), convert_input(polyphagia),
    convert_input(genital_thrush), convert_input(visual_blurring), convert_input(itching),
    convert_input(irritability), convert_input(delayed_healing), convert_input(partial_paresis),
    convert_input(muscle_stiffness), convert_input(alopecia), convert_input(obesity)
]).reshape(1, -1)

# Convert input_data into a pandas DataFrame to match the training format
input_df = pd.DataFrame(input_data, columns=numerical_cols)

# Predict button with improved style
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("🔍 **Predict Risk**", help="Click to predict the diabetes risk based on the inputs"):
    # Scale the input data using the loaded scaler
    if isinstance(scaler, MinMaxScaler):  # Make sure scaler is valid
        input_scaled = scaler.transform(input_df)
        
        # Predict using the loaded model
        prediction = model.predict(input_scaled)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error("🚨 **High Risk!** The model predicts that the patient **has Diabetes.** Please consult a doctor.")
        else:
            st.success("✅ **Good News!** The model predicts that the patient **does NOT have Diabetes.** Stay healthy!")
    else:
        st.error("Error: Scaler is not correctly loaded. Please check your model and scaler files.")
