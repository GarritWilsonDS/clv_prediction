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

def clustering(k=None, data=None, column=None):
    '''This function clusters data of a given column,
    and returns the dataframe with the cluster predictions.'''
    
    kmeans = KMeans(n_clusters = k,
                    max_iter= 1000)
    
    kmeans.fit(data[[column]])
    
    new_column = column + "Cluster"
    
    data[new_column] = kmeans.predict(data[[column]])
    
    return data