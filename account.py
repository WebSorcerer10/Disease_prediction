import streamlit as st
import joblib
import numpy as np

# import pickle
# from pathlib import Path
# import streamlit_authenticator as stauth

import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth


cred = credentials.Certificate('disease-prediction-da0a1-9a8553fdf296.json')
firebase_admin.initialize_app(cred)

    # # --- user authentication --- 
    # names=["Peter parker" , "Rebecca Miller "]
    # usernames=["pparker","rmiller"]
    # # ---load hashed passwords -- 
    # file_path = Path(__file__).parent/ "hashed_pw.pkl"
    # with file_path.open("rb") as file:
    #     #rb - reading them in binary mode
    #     hashed_passwords=pickle.load(file)

    # # prediction_dashboard is the cookie 
    # # abcdef is the random key to store the cookie signature
    # authenticator=stauth.Authenticate(names,usernames,hashed_passwords,"prediction_dashboard","abcdef",cookie_expiry_days=30)

    # name,authentication_status,username=authenticator.login("Login","main")
    # if authentication_status == False:
    #     st.error("Username/password is incorrect")

    # if authentication_status == None:
    #     st.warning("Please enter your username and passoword")

    # if st.session_state["authentication_status"]:
    #     # authenticator.logout("Logout","sidebar")

    #     try:
    #         authenticator.logout('Logout','sidebar')
    #     except KeyError:
    #         pass
    #     except Exception as err:
    #         st.error(f'Unexpected exception {err}')
    #         raise Exception(err)
        
    #     st.sidebar.title(f"Welcome {name}")



def app():
    st.title('welcome to out site')
    # choice=st.selectbox('Login/Signup',['Login','Sign Up'])

    if 'username' not in st.session_state:
        st.session_state.username=''
    if 'useremail' not in st.session_state:
        st.session_state.useremail=''

    def f():
        try:
            user=auth.get_user_by_email(email)
            # print(user.uid)
            st.write('Login Successful')
            st.session_state.username=user.uid
            st.session_state.useremail=user.email
            
            st.session_state.signedout=True
            st.session_state.signout=True
            
            
        except:
            st.warning('Login Failed')
            
    def t():
        st.session_state.signedout=False
        st.session_state.signout=False
        st.session_state.username=''
        
    if 'signedout' not in st.session_state:
        st.session_state.signedout=False
    if 'signout' not in st.session_state:
        st.session_state.signout=False

    # below function occurs when the user is signedout or not sign in 
    
    if not st.session_state['signedout']:
        choice=st.selectbox('Login/Signup',['Login','Sign Up'])   
        
        if choice == 'Login':
            email=st.text_input('Email Address')
            password=st.text_input('Password',type='password')
            st.button('Login',on_click=f)
                
        else:
            email=st.text_input('Email Address')
            password=st.text_input('Password',type='password')
            username=st.text_input('Enter Your unique username')
            if st.button('Create my account'):
                user=auth.create_user(email = email,password = password,uid=username)
                st.success('Account created successfully')
                st.markdown('Please login using your email and password')
                st.balloons()
        
    if st.session_state.signout:
        st.text('Name: '+st.session_state.username)
        st.text('Email id: '+st.session_state.useremail)
        st.button('Sign out',on_click=t)
                        
        # def predictDisease(symptoms):
        #     #final_svm_model = joblib.load("final_svm_model.pkl")
        #     final_svm_model = joblib.load('final_svm_model.pkl')
        #     encoder = joblib.load('encoder.pkl')
        #     data_dict = joblib.load('data_dict.pkl')

        #     input_symptoms = symptoms.split(",")
        #     input_data = [0] * len(data_dict["symptom_index"])
        #     for symptom in input_symptoms:
        #         #symptom = symptom.strip().lower()  # Convert input symptoms to lowercase and remove leading/trailing spaces
        #         if symptom in data_dict["symptom_index"]:
        #             index = data_dict["symptom_index"][symptom]
        #             input_data[index] = 1
        #         else:
        #             st.warning(f"Warning: '{symptom}' might not be a valid symptom.")

        #     input_data = np.array(input_data).reshape(1, -1)
        #     svm_prediction_index = final_svm_model.predict(input_data)[0]
        #     svm_prediction = data_dict["predictions_classes"][svm_prediction_index]

        #     return svm_prediction
        # st.sidebar.title("Navigation")

        # selected_page = st.sidebar.selectbox("",["🔍 Prediction"])  

        # if selected_page == "🔍 Prediction":

        #     st.markdown('<h1 style="color: Teal; font-family: Snell Roundhand, cursive;">🩺 Predict Your Disease 🩺</h1>', unsafe_allow_html=True)
        #     s1 = st.text_input('Symptom 1', value='')
        #     s2 = st.text_input('Symptom 2', value='')
        #     s3 = st.text_input('Symptom 3', value='')
        #     s4 = st.text_input('Symptom 4', value='')
        #     s5 = st.text_input('Symptom 5', value='')


                
        #     if st.button("Get your Prognosis"):
        #         symptoms = f"{s1},{s2},{s3},{s4},{s5}"
        #         prediction = predictDisease(symptoms)
        #         st.success(f"It is most likely to be : {prediction}")
