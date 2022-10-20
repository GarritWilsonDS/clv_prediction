from modules.modules import clean_dataframe, clustering
from datetime import datetime, timedelta, date

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

    ## segmenting data into 3 and 6 month dataframes.
    ## 3 Months of data will be used to forecast CLV over the following 6 months.
    data_3m = df[(df.InvoiceDate.dt.date < date(2011,6,1)) & (df.InvoiceDate.dt.date >= date(2011,3,1))].reset_index()
    data_6m = df[(df.InvoiceDate.dt.date >= date(2011,6,1)) & (df.InvoiceDate.dt.date < date(2011,12,1))].reset_index()
    
    user_df = pd.DataFrame(data_3m.CustomerID.unique(), columns= ["CustomerID"])
    
    ## creating Recency Metric
    recency_df = pd.DataFrame(data_3m.groupby("CustomerID")["InvoiceDate"].max().reset_index())
    recency_df.columns = ["CustomerID", "LatestPurchase"]
    
    recency_df["Recency"] = (data_3m["InvoiceDate"].max() - recency_df["LatestPurchase"]).dt.days
    recency_df.drop("LatestPurchase", axis= 1, inplace= True)
    
    recency_df = clustering(data = recency_df,
                            k = 3,
                            column="Recency")
    
    user_df = pd.merge(recency_df, user_df, on= "CustomerID") 
    
    st.dataframe(data= user_df.head())