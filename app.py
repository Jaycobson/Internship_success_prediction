#!/usr/bin/env python
# coding: utf-8

# # Model Deployment

# ### Creating Streamlit App

# In[1]:


#importing packages

import streamlit as st
import pickle
import numpy as np
from catboost import CatBoostClassifier


# In[2]:


# Loading the pre-trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


# In[3]:


features = ['Academic_Performance',
 'Research Experience',
 'Relevant_Skills',
 'Letters_of_Recommendation',
 'Interview_Score',
 'Motivation Level',
 'Extracurricular_Activities',
 'Age',
 'Coding Experience',
 'Work Status',
 'Access to Constant Electricity'] 


# In[4]:


def main():
    st.title("Internship Success Prediction")
    st.text('Enter your details below')
    
    # Getting user input for features
    user_input = []
    for feature in features:
        if feature == 'Age':
            user_input.append(st.number_input(f"Enter {feature} within 18 and 30", step=1, min_value = 0, max_value = 10))
        elif feature == 'Motivation Level':
            user_input.append(st.number_input(f"Enter {feature} within 0 and 10", step=1,min_value = 0, max_value = 10))
        elif feature == 'Letters_of_Recommendation':
            user_input.append(st.number_input(f"Enter {feature} within 0 and 4", step=1,min_value = 0, max_value = 4))
        elif feature == 'Academic_Performance':
            user_input.append(st.number_input(f"Enter {feature} within 3 and 5", step=1,min_value = 3, max_value = 5))
        elif feature == 'Experience':
            user_input.append(st.number_input(f"Enter {feature} within 0 and 10", step=1,min_value = 0, max_value = 10))
        else:
            user_input.append(st.number_input(f"Enter {feature}", step=0))
        
    # Making a prediction based on user input
    if st.button("Predict"):
        user_input_array = np.array([user_input])
        prediction = model.predict(user_input_array)[0]
        st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()


# In[ ]:




