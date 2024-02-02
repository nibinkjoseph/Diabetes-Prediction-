# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:27:38 2023

@author: lidya
"""

import numpy as np
import pickle

import streamlit as st

loaded_model =  pickle.load(open('C:/Users/lidya/Desktop/DATA SCIENCE PROJECTS 2024/Diabetes prediction/trained_model.sav','rb'))


def  dia_prediction(input_data):
  
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    
    if (prediction[0] == 0):
        return'The person is not diabetic'
    else:
        return'The person is diabetic'
        
        
def main():

    #Title    
    st.title("Diabetes Prediction Web App")
    st.subheader("By Nibin Joseph (Post Gratuate in Data Science Project)")
    
    #Getting input data from use
    

    
    age = st.text_input("Enter the Age")
    hypertension = st.text_input("Hypertension value")
    heart_disease = st.text_input("Heart_disease value")
    bmi = st.text_input("BMI value")
    HbA1c_level = st.text_input("Enter the HbA1c_level value")
    blood_glucose_level = st.text_input("Enter the blood_glucose_level value")
    numeric_gender = st.text_input("Gender")
    smoking_history_numeric = st.text_input("Enter the smoking_history_numeric value(O or 1")
    
    
    #code for prediction
    
    diagnosis = ''
    
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        
       diagnosis =  dia_prediction([age,hypertension, heart_disease,bmi,HbA1c_level,blood_glucose_level,numeric_gender,smoking_history_numeric ])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
