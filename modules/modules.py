import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost as xgb

from datetime import datetime, timedelta, date
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate

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
    
    ## creating revenue column
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    
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

def data_prep(df):
    '''This function applies scaling and encoding to features, 
    for the step of modeling and predicting.'''
    
    for col in ["Recency", "Frequency", "Revenue"]:
    
        scaler = RobustScaler()
        
        scaler.fit(df[[col]])
        
        df[col] = scaler.transform(df[[col]])
    
    return df

def modeling(df):
    '''This function fits an XGBRegressor algorithm to the data,
    and predicts the lifetime value per Customer.'''
    
    X = df.drop(["CustomerID", "6m_Revenue", "LTVCluster"], axis= 1)
    y= df["6m_Revenue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
    
    xgb_regressor = xgb.XGBRegressor()
    
    xgb_regressor.fit(X_train, y_train)
    
    cv_results = cross_validate(xgb_regressor,
                               X= X_train,
                               y= y_train,
                               cv= 5,
                               scoring= ["neg_mean_squared_error"])
    
    score = abs(cv_results["test_neg_mean_squared_error"].mean())
    
    print(f'RMSE of XGBRegressor: {math.sqrt(score)}')
    
    return df, (X_train, X_test, y_train, y_test), xgb_regressor
    

def predict(X_test, y_test, model):
    '''This function makes prediction on the test data.'''
    
    predicted_ltv = X_test
    predicted_ltv["Predicted_LTV"] = model.predict(predicted_ltv)

    predicted_ltv["CustomerID"] = list(range(0, len(predicted_ltv)))
    predicted_ltv["Actual_LTV"] = y_test

    predicted_ltv= predicted_ltv[["CustomerID", "Actual_LTV", "Predicted_LTV"]]

    return predicted_ltv
    