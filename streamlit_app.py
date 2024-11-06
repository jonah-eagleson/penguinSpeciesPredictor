import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('Penguin Species Predictor')

st.info("This app utilizes a machine learning model to predict penguin species")

with st.expander('Data'):
    st.write('**RAW data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.species
    y_raw

with st.expander("Data Visualization"):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data prep
with st.sidebar:
    st.header("Input Features")
    island = st.selectbox("Island", ("Biscoe", "Dream", "Torgerson"))
    bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Sex', ('male', 'female'))


# Create Datafram for input
    data = {'island': island, 
            'bill_length_mm': bill_length_mm, 
            'bill_depth_mm': bill_depth_mm, 
            'flipper_length_mm': flipper_length_mm, 
            'body_mass_g': body_mass_g, 
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
    st.write('**Input Penguin**')
    input_df
    st.write("**Combined Penguins Data**")
    input_penguins

# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0, 
                 'Chinstrap': 1, 
                 'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander("Data Preperation"):
    st.write('**Encoded Input Penguin (X)**')
    input_row
    st.write("**Encoded y**")
    y

# Model Training
clf = RandomForestClassifier()
clf.fit(X, y)

# Apply model to make predictions
prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)

df_prediction_prob = pd.DataFrame(prediction_prob)
df_prediction_prob.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_prob.rename(columns={0: 'Adelie', 
                                1: 'Chinstrap', 
                                2: 'Gentoo'})


# Display prediction
df_prediction_prob

st.subheader('Predicted Species')
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguin_species[prediction][0]))