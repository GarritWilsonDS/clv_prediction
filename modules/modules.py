import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost as xgb
import plotly.express as px

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

def make_scatter3d(dataframe):
    
    df = dataframe.copy()
    
    
    for col in ["Frequency", "Recency", "Revenue"]:

        mask = df[col] > df[col].quantile(0.99)

        df.drop(df[mask].index, inplace=True)

    ## plotting with plotly
    fig = px.scatter_3d(df, x='Frequency', y='Recency', z='Revenue', color= "OverallScore",
                    size_max=2, opacity= 0.6)

    return fig

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
    
    X = df.drop(["CustomerID", "LifetimeValue"], axis= 1)
    y= df["LifetimeValue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
    
    xgb_regressor = xgb.XGBRegressor()
    
    xgb_regressor.fit(X_train, y_train)
    
    cv_results = cross_validate(xgb_regressor,
                               X= X_train,
                               y= y_train,
                               cv= 5,
                               scoring= ["neg_mean_squared_error"])
    
    mse = abs(cv_results["test_neg_mean_squared_error"].mean())
    
    rmse = round(math.sqrt(mse), 2)
    
    return df, (X_train, X_test, y_train, y_test), (xgb_regressor, rmse)
    

def predict(X_test, y_test, model):
    '''This function makes prediction on the test data.'''
    
    predicted_ltv = X_test
    predicted_ltv["Predicted_LTV"] = model.predict(predicted_ltv)

    predicted_ltv["CustomerID"] = list(range(0, len(predicted_ltv)))
    predicted_ltv["Actual_LTV"] = y_test

    predicted_ltv= predicted_ltv[["CustomerID", "Actual_LTV", "Predicted_LTV"]]

    return predicted_ltv

def plot_predictions(df):
    '''This function takes a random sample of customers from the predicted test data,
    and plots y_pred vs. y_true.'''
    
    sample = df.sample(n=50)
    
    X= sample["CustomerID"]
    Y1 = sample["Predicted_LTV"]
    Y2 = sample["Actual_LTV"]

    X_axis = np.arange(len(X))

    plt.figure(figsize=(20,10))
    plt.bar(X_axis - 0.2, Y1, 0.4, label = 'Predicted LTV')
    plt.bar(X_axis + 0.2, Y2, 0.4, label = 'Real LTV')

    plt.title("Actual vs. Predicted Lifetime Value for 100 Random Customers", size= 20)
    plt.ylabel('LTV', size= 15)
    plt.legend()
    
    plt.savefig("imgs/plotted_predictions.png")
    