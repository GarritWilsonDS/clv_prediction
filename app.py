from modules.modules import clean_dataframe, clustering, order_clusters
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
    
    st.text('')
    st.text('')
    st.text('')
    
    st.markdown("##### Original Dataframe:")
    st.dataframe(data=dataframe.head())
        
    df = clean_dataframe(dataframe)

    st.text('')
    st.text('')
    st.text('')

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
    
    recency_df = clustering(data= recency_df,
                            k= 3,
                            column="Recency")
    
    recency_df = order_clusters(data= recency_df,
                                column= "RecencyCluster",
                                target= "Recency",
                                ascending= False)
    
    user_df = pd.merge(recency_df, user_df, on= "CustomerID") 
    
    ## creating Frequency Metric
    frequency_df = pd.DataFrame(data_3m.groupby("CustomerID")["InvoiceDate"].count().reset_index())
    frequency_df.columns = ["CustomerID", "Frequency"]
    
    frequency_df = clustering(data= frequency_df,
                              k= 5,
                              column= "Frequency")
    
    frequency_df = order_clusters(data= frequency_df,
                                  column= "FrequencyCluster",
                                  target= "Frequency",
                                  ascending= True)
    
    user_df = pd.merge(frequency_df, user_df, on= "CustomerID")
    
    ## creating Revenue Metric
    revenue_df = pd.DataFrame(data_3m.groupby("CustomerID")["Revenue"].sum().reset_index())
    revenue_df.columns = ["CustomerID", "Revenue"]
    
    revenue_df = clustering(data= revenue_df,
                            k= 5,
                            column= "Revenue")
    
    revenue_df = order_clusters(data= revenue_df,
                                column= "RevenueCluster",
                                target= "Revenue",
                                ascending= True)
    
    user_df = pd.merge(revenue_df, user_df, on= "CustomerID")
    
    st.markdown("##### Final Dataframe used for prediction:")
    st.text('''This dataframe contains the predictive variables Recency, Frequency, and Revenue, 
per unique customer.''')
    
    st.dataframe(data= user_df.head())
    