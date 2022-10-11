from tkinter.tix import InputOnly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from datetime import datetime, timedelta, date

def clean_dataframe(df):
    '''Function to clean dataframe from missing data, outliers, duplicates,
    and to change data types as necessary.'''
    
    
    ## dropping outliers and negative values in Quantity
    idx_neg = df.loc[df.Quantity < 0].index
    
    df.drop(idx_neg, inplace= True)
    
    idx_outliers = df.sort_values(by="Quantity", ascending= False).head(2).index
    
    df.drop(idx_outliers, inplace= True)
    
    ## changing datatype for InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
    
    return df