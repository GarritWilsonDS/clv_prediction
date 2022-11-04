import pandas as pd

from sklearn.cluster import KMeans


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

def segment_data(df, clv_freq):
    '''This function slices the dataframe into chunks of a select timeframe.
    This is done so that previous timeframes can be used to predict CLV for a later timeframe.
    Ex.: Slicing into 3 months timeframes, to predict the last chunk based on all previous ones.'''
    
    def groupby_mean(x):
        return x.mean()

    def groupby_count(x):
        return x.count()

    def purchase_duration(x):
        return (x.max() - x.min()).days

    def avg_frequency(x):
        return (x.max() - x.min()).days / x.count()

    groupby_mean.__name__ = 'avg'
    groupby_count.__name__ = 'count'
    purchase_duration.__name__ = 'purchase_duration'
    avg_frequency.__name__ = 'purchase_frequency'
    
    df_orders = df.groupby(['CustomerID', 'InvoiceNo']).agg({'Revenue': sum, 'InvoiceDate': max})

    df_data = df_orders.reset_index().groupby([
                'CustomerID',
                pd.Grouper(key='InvoiceDate', freq=clv_freq)
                ]).agg({'Revenue': [sum, groupby_mean, groupby_count],})

    df_data.columns = ['_'.join(col).lower() for col in df_data.columns]
    
    df_data.reset_index(inplace= True)
    
    map_date_month = {str(x)[:10]: 'M_%s' % (i+1) for i, x in enumerate(
                    sorted(df_data.reset_index()['InvoiceDate'].unique(), reverse=True))}
    
    df_data["M"] = df_data["InvoiceDate"].apply(lambda x: map_date_month[str(x)[:10]])
    
    return df_data
    
def create_features_and_target(df):
    '''This function takes the monthly aggregated data, and creates features
    as inputs for our regression from it.'''
    
    ## create features
    df_features = pd.pivot_table(
                    df.loc[df["M"] != "M_1"],
                    values= ["revenue_sum", "revenue_avg", "revenue_count"],
                    columns = "M",
                    index= "CustomerID")
    
    df_features.columns = ['_'.join(col) for col in df_features.columns]
    
    df_features.reset_index(level=0, inplace= True)
    
    df_features.fillna(0, inplace=True)
    
    ## create target
    df_target = df.loc[df["M"] == "M_1"][["CustomerID", "revenue_sum"]]

    df_target.columns = ["CustomerID", "CLV"]
    
    ## creating final dataframe by merging
    df_final = pd.merge(df_features, df_target, on= "CustomerID", how= "left")
    
    df_final.fillna(0, inplace=True)
    
    return df_final