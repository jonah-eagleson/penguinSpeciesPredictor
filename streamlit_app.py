import streamlit as st
import pandas as pd
import sklearn
import numpy

st.title('Penguin Species Predictor')

st.info("This app utilizes a machine learning model to predict penguin species")

with st.expander('Data'):
    st.write('**RAW data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    df
