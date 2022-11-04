from modules.cleaning import clean_dataframe
from modules.features import clustering, segment_data, order_clusters, create_features_and_target
from modules.plotting import make_scatter3d, plot_predictions
from modules.ml import data_prep, modeling, predict
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.markdown("# Customer Lifetime Value Prediction")

st.text('')
st.text('')
st.text('')

st.markdown("Please upload transaction data here:")
st.markdown("Please make sure the dataframe contains the following columns: ")
st.markdown("'CustomerID', 'Revenue' & 'InvoiceDate'")


st.text('')
st.text('')
st.text('')

csv_data = st.file_uploader("Transaction Data")

## clean dataframe

if csv_data:
    
    data = pd.read_csv(csv_data, encoding= 'unicode_escape')
    data = data.loc[data['InvoiceDate'] < '2011-12-01']
    
    st.text('')
    st.text('')
    st.text('')
    
    st.markdown("##### Original Dataframe:")
    st.dataframe(data=data.head())
        
    df = clean_dataframe(data)

    st.text('')
    st.text('')
    st.text('')

    ## segmenting data into 3 - month bins and selecting all but last one
    tmp = segment_data(df, clv_freq='3M')
    
    date = pd.to_datetime(sorted(tmp['InvoiceDate'].unique(), reverse=True)[1]).date()
    
    dataframe = df.loc[df["InvoiceDate"].dt.date <= date]
    
    
    user_df = pd.DataFrame(dataframe.CustomerID.unique(), columns= ["CustomerID"])
    
    ## creating Recency Metric
    recency_df = pd.DataFrame(dataframe.groupby("CustomerID")["InvoiceDate"].max().reset_index())
    recency_df.columns = ["CustomerID", "LatestPurchase"]
    
    recency_df["Recency"] = (dataframe["InvoiceDate"].max() - recency_df["LatestPurchase"]).dt.days
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
    frequency_df = pd.DataFrame(dataframe.groupby("CustomerID")["InvoiceDate"].count().reset_index())
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
    revenue_df = pd.DataFrame(dataframe.groupby("CustomerID")["Revenue"].sum().reset_index())
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
    st.markdown("##### Plotting the customer segmentation:")
    st.plotly_chart(plot, sharing= "streamlit")

    ## Segmenting data into 3-months bins and create features 
    df_data = segment_data(df, clv_freq='3M')
    
    st.dataframe(data=df_data.head())
    st.markdown('### ACHTUNG: hier stimmt was nicht, M_4 und M_5 fehlen.')
    
    df_final = create_features_and_target(df_data)
    
    st.dataframe(data=df_final.head())
    
    ## merging dataframes
    final_dataframe = pd.merge(df_final, user_df, left_on= "CustomerID", right_on= "CustomerID", how="left")
    
#    st.dataframe(data=final_dataframe.head())

    
    ## data preparation (scaling, encoding)
    df_prep = data_prep(final_dataframe)
    
    ## modeling and prediction
    dataframe, data_split, _ = modeling(df_prep)
    
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
    
    
    
    