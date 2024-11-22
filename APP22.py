import pandas as pd
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np


age_options={
    1:'Young(1)',
    2:'Adult(2)',
    3:'Old(3)'
}

race_options={
    1:'White(1)',
    2:'Black(2)',
    3:'Other(3)'
}

site_options={
    1:'Cardia(1)',
    2:'Body(2)',
    3:'Lower(3)',
    4:'Overlapping',
    5:'Nos'
}

risk_options={
    1:'Very low(1)',
    2:'Low(2)',
    3:'Medium(3)',
    4:'High'
}


marital_options={
    1:'Single(1)',
    2:'Married(2)',
    3:'Unknown(3)'
}


surgery_options={
    1:'Performed(1)',
    2:'Not recommended(2)',
    3:'Unknown(3)'
}

feature_names = [ "age", "race", "sex", "site", "risk", "marital", "surgery"]

st.header("Streamlit Machine Learning App")

age=st.selectbox("age:", options=list(age_options.keys()), format_func=lambda x: age_options[x])
race=st.selectbox("race:", options=list(race_options.keys()), format_func=lambda x: race_options[x])

sex = st.selectbox("sex ( 1=Male, 2=Female):", options=[1, 2], format_func=lambda x: 'Female (2)' if x == 2 else 'Male (1)')
site=st.selectbox("site:", options=list(site_options.keys()), format_func=lambda x: site_options[x])
risk=st.selectbox("risk:", options=list(risk_options.keys()), format_func=lambda x: risk_options[x])
marital=st.selectbox("marital:", options=list(marital_options.keys()), format_func=lambda x: marital_options[x])
surgery=st.selectbox("surgery:", options=list(surgery_options.keys()), format_func=lambda x: surgery_options[x])

feature_values = [age, race, sex, site, risk, marital, surgery]   
features = np.array([feature_values])


if st.button("Submit"):
    clf = open("XGB.pkl","rb")
    s=pickle.load(clf)
    predicted_class = s.predict(features)[0]
    predicted_proba = s.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
        st.write('Likely to die within 5 years')
    else:
        st.write('Likely to survive for 5 years')   
        
    explainer = shap.Explainer(s)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    st.image("shap_force_plot.png")

    
 
    
   
    
    
