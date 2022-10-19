from modules.modules import clean_dataframe

import streamlit as st
import pandas as pd

st.markdown("# Customer Lifetime Value Prediction")

st.text('')
st.text('')
st.text('')

st.text("Please upload transaction data here:")

st.text('')
st.text('')
st.text('')

csv_data = st.file_uploader("Transaction Data")

## clean dataframe

if csv_data:
    
    dataframe = pd.read_csv(csv_data, encoding= 'unicode_escape')
    
    original_len = dataframe.shape[0]
    
    st.text(f"{type(dataframe)}")
    
    st.dataframe(data=dataframe.head())
        
    df = clean_dataframe(dataframe)

    st.text('')
    st.text('')
    st.text('')
    st.text(f'New length of dataframe: {df.shape[0]}.')


    ## ...