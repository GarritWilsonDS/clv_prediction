from modules.modules import clean_dataframe

import streamlit as st

st.markdown("# Customer Lifetime Value Prediction")

st.text('')
st.text('')
st.text('')

st.text("Please upload transaction data here:")

st.text('')
st.text('')
st.text('')

dataframe = st.file_uploader("Transaction Data")

#st.text(f'Current length of dataframe: {dataframe.shape[0]}')

## clean dataframe

df = clean_dataframe(dataframe)

st.text('')
st.text('')
st.text('')
st.text(f'New length of dataframe: {df.shape[0]}')


## ...