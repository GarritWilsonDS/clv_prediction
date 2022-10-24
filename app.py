from curses import use_default_colors
from modules.modules import (clean_dataframe, clustering, make_scatter3d, 
                             order_clusters, data_prep, modeling, plot_predictions, predict)
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
    
    user_df["OverallScore"] = user_df["RecencyCluster"] + user_df["FrequencyCluster"] + user_df["RevenueCluster"]
    
    
    ## plotting customer segmentation in 3d plotly scatterplot
    plot = make_scatter3d(user_df)
    
    st.text('')
    st.text('')
    st.text('')
    st.plotly_chart(plot, sharing= "streamlit")
    
    ## calculating 6-months lifetime value
    lv_df = pd.DataFrame(data_6m.groupby("CustomerID")["Revenue"].sum().reset_index())
    lv_df.columns = ["CustomerID", "LifetimeValue"]
    
    ## merge predictive dataframe and 6-months lifetime value dataframe
    df_scaled = pd.merge(user_df, lv_df, on= "CustomerID", how= "left")
    df_scaled.dropna(inplace=True)
    
    ## data preparation (scaling, encoding)
    df_prep = data_prep(df_scaled)
    
    ## modeling and prediction
    dataframe, data_split, _ = modeling(df_scaled)
    
    X_train, X_test, y_train, y_test = data_split[0], data_split[1], data_split[2], data_split[3]
    
    model, rmse = _
    
    ## making predictions on test data
    df_pred = predict(X_test, y_test, model).reset_index()
    
    plot_predictions(df_pred)
    
    st.text('')
    st.text('')
    st.text('')
    
    st.markdown("##### Plotting Actual vs. Predicted LTV for 50 Random Customers")
    st.image("imgs/plotted_predictions.png")
    st.text(f'Root Mean Squared Error: {rmse}')
    
    st.text('')
    st.text('')
    st.text('')
    st.text('Comparing the RMSE to the distribution of LTV over the all customers.')
    st.dataframe(df_prep["LifetimeValue"].describe())
    
    
    