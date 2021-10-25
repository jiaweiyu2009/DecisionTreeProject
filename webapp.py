import streamlit as st
import pandas as pd
from mp2 import run_train_test_2
from mp2 import run_train_test
import json
import math

st.write("""
# Predict the number of rings of an abalone
  

""")

st.sidebar.header('User inputs')


sex = st.sidebar.selectbox(
    'select type of sex', (0, 1, 2))
length = st.sidebar.number_input('length (mm)', format="%.4f",
                                 min_value=0.075, max_value=0.815, value=0.305)
diameter = st.sidebar.number_input(
    'diameter (mm)', format="%.4f", min_value=0.055, max_value=0.65, value=0.225)
height = st.sidebar.number_input(
    'height (mm)', format="%.4f", min_value=0.000, max_value=1.13, value=0.07)
whole_height = st.sidebar.number_input(
    'whole height (g)', format="%.4f", min_value=0.002, max_value=2.826, value=0.1485)
shucked_weight = st.sidebar.number_input(
    'shucked weight (g)', format="%.4f",  min_value=0.001, max_value=1.488, value=0.0585)
viscera_weight = st.sidebar.number_input(
    'viscera weight (g)', format="%.4f",  min_value=0.001, max_value=0.7, value=0.0335)
shell_weight = st.sidebar.number_input(
    'shell weight (g)', format="%.4f",  min_value=0.002, max_value=1.005, value=0.045)

data = {'sex': sex, 'length': length, 'diameter': diameter, 'height': height, 'whole height': whole_height,
        'shucked weight': shucked_weight, 'viscera weight': viscera_weight, 'shell weight': shell_weight}

df = pd.DataFrame(data, index=['values'])
st.write(df)

train_data = json.load(open('train.json'))
dev_data = json.load(open('dev.json'))

testing_data = [sex, length, diameter, height, whole_height,
                shucked_weight, viscera_weight, shell_weight]

# prediction = run_train_test(
#     train_data['data'], train_data['label'], dev_data['data'])
x = 0
x -= math.log(1, 2) * 1
st.write(x)
