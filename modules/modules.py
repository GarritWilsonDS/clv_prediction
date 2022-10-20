import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import datetime, timedelta, date
from sklearn.cluster import KMeans

def clean_dataframe(df):
    '''Function to clean dataframe from missing data, outliers, duplicates,
    and to change data types as necessary.'''
    
    
    ## dropping outliers and negative values in Quantity
    idx_neg = df.loc[df.Quantity < 0].index
    
    df.drop(idx_neg, inplace= True)
    
    
    ## removing outliers
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    
    for col in num_cols:
        
        mask = df[col] > df[col].quantile(0.99)

        df.drop(df[mask].index, inplace=True)
        
    ## changing datatype for InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
    
    return df

def clustering(data=None, k=None, column=None):
    '''This function clusters data of a given column,
    and returns the dataframe with the cluster predictions.'''
    
    kmeans = KMeans(n_clusters = k,
                    max_iter= 1000)
    
    kmeans.fit(data[[column]])
    
    new_column = column + "Cluster"
    
    data[new_column] = kmeans.predict(data[[column]])
    
    return data

def order_clusters(data=None, column=None, target=None, ascending=None):
    '''This function orders the clusters of a given dataframe,
    so that cluster names are not a nominal variable but ordinal.'''
    
    new_column = "new_" + column 
    
    df = data.groupby(column)[target].mean().reset_index()
    
    df = df.sort_values(by=target, ascending=ascending)
    
    df["index"] = df.index
    
    df_final = pd.merge(data, df[[column, "index"]], on=column)
    
    df_final.drop([column], axis=1, inplace=True)
    
    df_final = df_final.rename(columns={"index": column})
    
    return df_final
