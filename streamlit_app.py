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

    st.write('**X**')
    X = df.drop('species', axis=1)
    X

    st.write('**y**')
    y = df.species
    y

with st.expander("Data Visualization"):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data prep

with st.sidebar:
    st.header("Input Features")
    island = st.selectbox("Island", ("Biscoe", "Dream", "Torgerson"))
    gender = st.selectbox('Gender', ('Male', 'Female'))
    bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)