import math
import xgboost as xgb

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate

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
    
    X = df.drop(["CustomerID", "CLV"], axis= 1)
    y= df["CLV"]
    
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