import streamlit as st
import numpy as np
import pandas as pd
from heart_attack_predictions import pred_heart
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=4)

df = pd.read_csv('heart.csv')

st.markdown('# Heart attack prediction model')
st.markdown('##### Aim: Various models are accessed in order to find the best performing model to predict whether a patient will develop cancer')
st.markdown('##### Data: 304 patients data used')

st.markdown('#### Data: Heart-attack likeliness')

st.bar_chart(df.output.value_counts(),100,350)

st.markdown('#### Number of female vs. male patients')
st.bar_chart(df.sex.value_counts(),100,350)

st.markdown('##### Model Accuracy Scores (original vs hyper-parameter vs. enhanced (incl hyper-parameters')




st.markdown('#### Best model: Random forest Classifier')
st.markdown('The random forest model classifier can be used below to predict the outcome of cancer or not')


st.markdown('Please input Patient data below for condition prediciton')

age = st.number_input('Patient age')
st.write(age)
sex = st.number_input('Patient sex: 0 for male, 1 for female')
st.write(sex)
cp = st.number_input('Chest Pain type , between 1 and 4')
st.write(cp)
trtbps = st.number_input('resting blood pressure (in mm Hg)')
st.write(trtbps)
chol = st.number_input('cholestoral in mg/dl fetched via BMI sensor')
st.write(chol)
fbs = st.number_input('fasting blood sugar > 120 mg/dl) (1 = true; 0 = false')
st.write(fbs)
restecg = st.number_input('resting electrocardiographic results, between 0 and 2')
st.write(restecg)
thalachh = st.number_input('maximum heart rate achieved')
st.write(thalachh)
exng = st.number_input('exercise induced angina (1 = yes; 0 = no)')
st.write(exng)
oldpeak = st.number_input('previous peak')
st.write(oldpeak)
slp = st.number_input('slope, between 0 and 2')
st.write(slp)
caa = st.number_input('number of major vessels (0-3)')
st.write(caa)
thall = st.number_input('Thall rate, between 0 and 3')
st.write(thall)

vars_list = (age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall )
vars_list = np.array(vars_list)
vars_list = vars_list.reshape((1, -1))
pred = pred_heart(vars_list)

st.markdown(f'## Likelihood of a heart attack is {pred}')


