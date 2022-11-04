import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def make_scatter3d(dataframe):
    
    df = dataframe.copy()
    
    
    for col in ["Frequency", "Recency", "Revenue"]:

        mask = df[col] > df[col].quantile(0.99)

        df.drop(df[mask].index, inplace=True)

    ## plotting with plotly
    fig = px.scatter_3d(df, x='Frequency', y='Recency', z='Revenue', color= "OverallScore",
                    size_max=2, opacity= 0.6)

    return fig

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