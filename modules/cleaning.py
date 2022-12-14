import pandas as pd

def clean_dataframe(df):
    '''Function to clean dataframe from missing data, outliers, duplicates,
    and to change data types as necessary.'''
    
    if "Revenue" in df.columns:
        
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    else:
            
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