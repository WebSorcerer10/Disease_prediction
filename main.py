import streamlit as st
import joblib
import numpy as np

def authenticate_user(username, password):
    if username in user_database:
        if user_database[username]["password"] == password:
            return True
    return False


     
#     # Preprocess the symptoms if needed
#     # Make prediction using the loaded model
#     # Return the predicted disease
     #return loaded_model.predict([symptoms])[0]  # Assuming your model is ready for direct prediction
import joblib

# Load the trained SVM model

def predictDisease(symptoms):
    #final_svm_model = joblib.load("final_svm_model.pkl")
    final_svm_model = joblib.load('final_svm_model.pkl')
    encoder = joblib.load('encoder.pkl')
    data_dict = joblib.load('data_dict.pkl')

    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1


    input_data = np.array(input_data).reshape(1,-1)
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    return svm_prediction



user_database = {
    "user1": {"password": "password1"},
    "user2": {"password": "password2"}
}

st.sidebar.title("Navigation")

selected_page = st.sidebar.selectbox("",
                                     ["🔒 Login", "📝 Sign Up", "🔍 Prediction"])

if selected_page == "🔒 Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Login successful!")
            st.header("Let's diagnose together!")
        else:
            st.error("Invalid username or password")

elif selected_page == "📝 Sign Up":
    st.title("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if new_username in user_database:
            st.error("Username already exists")
        else:
            user_database[new_username] = {"password": new_password}
            st.success("Sign up successful! You can now login.")

elif selected_page == "🔍 Prediction":

    st.markdown('<h1 style="color: Teal; font-family: Snell Roundhand, cursive;">🩺 Predict Your Disease 🩺</h1>', unsafe_allow_html=True)
    s1 = st.text_input('Symptom 1', value='')
    s2 = st.text_input('Symptom 2', value='')
    s3 = st.text_input('Symptom 3', value='')
    
    if st.button("Get your Diagnosis"):
        symptoms = f"{s1},{s2},{s3}"
        prediction = predictDisease(symptoms)
        st.success(f"Predicted Disease: {prediction}")
