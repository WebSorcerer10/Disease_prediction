import streamlit as st
import joblib
import numpy as np


def app():
    
    if st.session_state.username=='':
        st.warning("login to see your posts")
    else:
        st.title('Welcome : '+st.session_state['username'] )
        def predictDisease(symptoms):
            #final_svm_model = joblib.load("final_svm_model.pkl")
            final_svm_model = joblib.load('final_svm_model.pkl')
            encoder = joblib.load('encoder.pkl')
            data_dict = joblib.load('data_dict.pkl')

            input_symptoms = symptoms.split(",")
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in input_symptoms:
                #symptom = symptom.strip().lower()  # Convert input symptoms to lowercase and remove leading/trailing spaces
                if symptom in data_dict["symptom_index"]:
                    index = data_dict["symptom_index"][symptom]
                    input_data[index] = 1
                else:
                    st.warning(f"Warning: '{symptom}' might not be a valid symptom.")

            input_data = np.array(input_data).reshape(1, -1)
            svm_prediction_index = final_svm_model.predict(input_data)[0]
            svm_prediction = data_dict["predictions_classes"][svm_prediction_index]

            return svm_prediction
        st.sidebar.title("Navigation")

        selected_page = st.sidebar.selectbox("",["🔍 Prediction"])  

        if selected_page == "🔍 Prediction":

            st.markdown('<h1 style="color: Teal; font-family: Snell Roundhand, cursive;">🩺 Predict Your Disease 🩺</h1>', unsafe_allow_html=True)
            s1 = st.text_input('Symptom 1', value='')
            s2 = st.text_input('Symptom 2', value='')
            s3 = st.text_input('Symptom 3', value='')
            s4 = st.text_input('Symptom 4', value='')
            s5 = st.text_input('Symptom 5', value='')


                
            if st.button("Get your Prognosis"):
                symptoms = f"{s1},{s2},{s3},{s4},{s5}"
                prediction = predictDisease(symptoms)
                st.success(f"It is most likely to be : {prediction}")
