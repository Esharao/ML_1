import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Guard", layout="wide")

# Getting the working directory of the .py file (to load models)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(os.path.join(working_dir, 'diabetes.pkl'), 'rb'))

heart_model = pickle.load(open(os.path.join(working_dir, 'heart_model.pkl'), 'rb'))

parkinsons_model = pickle.load(open(os.path.join(working_dir,'parkinsons_model.pkl'),'rb'))

# If heart_model and parkinsons_model are actual models, save them as .pkl first.
# For now, assuming they are CSVs, we load them as dataframes.
heart_data = pd.read_csv(os.path.join(working_dir, 'heart.csv'))
parkinsons_data = pd.read_csv(os.path.join(working_dir, 'parkinsons.csv'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction']
    )

# Main area handling based on the selection
if selected == 'Diabetes Prediction':
    st.title('Welcome to the Diabetes Prediction Model')
    col1,col2,col3 = st.columns(3)
    
    glucose = col1.slider('Glucose Level',0,500,120)
    bp = col2.slider('Blood Pressure Level',0,200,120)
    skthic = col3.slider('Skin Thickness Value',0,100,20)
    insulin = col1.slider('Insulin Level',0,900,30)
    bmi = col2.slider('BMI Value',0.0,70.0,25.0)
    dpf = col3.slider('Diabetes Pedigree Function Value',0.0,2.5,0.5)
    age = col1.slider('Age of the Person',0,100,25)
    
    if st.button('Diabetes Test Result'):
        user_input =[glucose,bp,skthic,insulin,bmi,dpf,age]
        pred  = diabetes_model.predict([user_input])[0]
        diab_diagnosis = 'The Person is Diabetic'  if pred ==1 else 'The Person is not Diabetic'
        st.success(diab_diagnosis)


# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title('Welcome to Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    
    # age = col1.slider('Age', 0, 100, 50)
    # sex = col2.radio('sex', ['Male', 'Female'])
    # cp = col3.selectbox('Chest Pain Types', ['Type1', 'Type2', 'Type3', 'Type4'])
    # trestbps = col1.slider('Resting Blood Pressure', 0, 200, 120)
    # chol = col2.slider('Serum Cholestrol in mg/dl', 50, 600, 200)
    # fbs = col3.radio('Fasting Blood Sugar > 120', ['Yes', 'No'])
    # restecg = col1.radio('Resting Electrocardiograph Results', ['Normal', 'Abnormal'])
    # thalach = col1.slider('thalach',100,200,140)
    # exang = col3.radio('Exercise Induced Angina', ['Yes', 'No'])
    # oldpeak = col1.slider('ST depression induced by exercise', 0.0, 10.0, 1.0)
    # slope = col2.selectbox('Slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])
    # thal = col1.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # # Map categorical values to numerical values for model input
    # sex = 1 if sex == 'Male' else 0
    # fbs = 1 if fbs == 'Yes' else 0
    # restecg = 1 if restecg == 'Normal' else 0
    # exang = 1 if exang == 'Yes' else 0
    # slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    # slope = slope_mapping[slope]
    # cp_mapping = {'Type1': 0, 'Type2': 1, 'Type3': 2, 'Type4': 3}
    # cp = cp_mapping[cp]
    # thal_mapping = {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}
    # thal = thal_mapping[thal]
    
    # # Collect the user inputs into a list for the model prediction
    # user_input = [
    #     age, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, thal
    # ]

    
    age = col1.slider('Age', 0, 100, 50)
    sex = col2.radio('sex', ('Male', 'Female'))
    trestbps = col3.slider('Resting Blood Pressure', 0, 200, 120)
    chol = col1.slider('Serum Cholesterol in mg/dl', 50, 600, 200)
    thalach = col2.slider('thalach', 100, 200, 140)
    exang = col3.radio('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = col1.slider('ST depression induced by exercise', 0.0, 10.0, 1.0)
    
    # Map categorical values to numerical values for model input
    sex = 1 if sex == 'Male' else 0
    exang = 1 if exang == 'Yes' else 0
    
    # Collect the user inputs into a list for the model prediction
    user_input = [age, trestbps, chol, thalach, exang, oldpeak, sex]
    
   
    # When the user clicks the "Heart Disease Test Result" button
    if st.button('Heart Disease Test Result', key='predict_heart'):
        # Make prediction using the loaded model
        prediction = heart_model.predict([user_input])[0]
        
        # Output the result
        result = 'The Person has Heart Disease' if prediction == 1 else 'The Person does not have Heart Disease'
        st.success(result)
    
elif selected == 'Parkinsons Prediction':
    st.title('Welcome to Parkinson’s Disease Prediction')
    col1, col2, col3 = st.columns(3)
    
    # Collecting inputs for relevant features
    fo = col1.slider('MDVP:Fo(Hz)', 50.0, 250.0, 120.0)
    fhi = col2.slider('MDVP:Fhi(Hz)', 100.0, 300.0, 150.0)
    flo = col3.slider('MDVP:Flo(Hz)', 50.0, 200.0, 100.0)
    jitter_percent = col1.slider('MDVP:Jitter(%)', 0.0, 0.1, 0.01)
    jitter_abs = col2.slider('MDVP:Jitter(Abs)', 0.0, 1.0, 0.5)
    rap = col3.slider('MDVP:RAP', 0.0, 0.05, 0.01)
    ppq = col1.slider('MDVP:PPQ', 0.0, 0.05, 0.01)
    ddp = col2.slider('Jitter:DDP', 0.0, 0.2, 0.05)
    shimmer = col3.slider('MDVP:Shimmer', 0.0, 0.1, 0.05)
    shimmer_db = col1.slider('MDVP:Shimmer(dB)', 0.0, 1.5, 0.3)  
    shimmer_apq3 = col1.slider('Shimmer:APQ3',0.0,1.,0.5)
    shimmer_apq5 = col2.slider('Shimmer:APQ5',0.0,1.0,0.5)
    mdvp_aqp = col3.slider('MDVP:APQ',0.0,1.0,0.5)
    shimmer_dda = col1.slider('Shimmer:DDA', 0.0, 0.2, 0.1)
    nhr = col2.slider('NHR', 0.0, 0.3, 0.1)
    hnr = col3.slider('HNR', 10.0, 40.0, 20.0)
    rpde = col1.slider('RPDE', 0.2, 1.0, 0.5)
    dfa = col2.slider('DFA', 0.5, 1.0, 0.8)
    spread1 = col3.slider('Spread1', -7.0, -2.0, -5.0)
    spread2 = col1.slider('Spread2', 0.0, 0.6, 0.3)
    d2 = col2.slider('D2', 1.0, 3.0, 2.0)
    ppe = col3.slider('PPE', 0.0, 1.0, 0.2)

    # When the user clicks on the "Parkinson’s Test Result" button
    if st.button('Parkinson’s Test Result'):
        # Gather user inputs into a list
        user_input = [
            fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer,  shimmer_db , shimmer_apq3, shimmer_apq5, mdvp_aqp,shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]
        
        # Make prediction using the loaded model (parkinsons_model)
        pred = parkinsons_model.predict([user_input])[0]  # Use your trained model
        
        # Output result
        parkinsons_diagnosis = 'The Person has Parkinson’s Disease' if pred == 1 else 'The Person does not have Parkinson’s Disease'
        st.success(parkinsons_diagnosis)

