import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from wrangle import *
from explore import *
import env
import os
pd.options.mode.chained_assignment = None

def baseline_mean(df, target):
    df = df.dropna()
    baseline = df[target].mean()

    return baseline


def rmse_metric(y_validate, target):
    baseline = baseline_mean(y_validate. target)
    metric_df = ([
    {
        'model': 'baseline_mean',
        'rmse': mean_squared_error(y_validate[target], baseline),
        'r^2': explained_variance_score(y_validate[target], baseline)
    
    }
    ])
    return metric_df

def model_metrics(model, 
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val, 
                  scores):
    '''
    
    '''
    scores = scores #this makes a returnable variable called "scores" from the argument of the function of the same name.

    model.fit(X_train, y_train['tax_val'])
    in_sample_pred = model.predict(X_train)
    out_sample_pred = model.predict(X_val)
    model_name = model
    y_train[model_name] = in_sample_pred
    y_val[model_name] = out_sample_pred
    print(y_val.shape)
    print(out_sample_pred.shape)
    rmse_val = mean_squared_error(y_val['tax_val'], out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(y_val['tax_val'], out_sample_pred)
    a = ({
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    })
    scores.append(a)
    return scores
def scaling_min_max(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled

def cluster_dummies(X_train):
    
    X_train = X_train.astype({'area_dollar_clusters': 'category', 'area_bedbath_clusters': 'category', "acre_bath_value": "category", "areas_valuepsqft": "category"})
    dummy_df = pd.get_dummies(X_train[['area_dollar_clusters','area_bedbath_clusters', "acre_bath_value", "areas_valuepsqft"]], dummy_na=False, drop_first=[True, True])
    X_train = pd.concat([X_train, dummy_df], axis=1)

    return X_train