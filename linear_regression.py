### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



def simulate_data(nobs):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    exp_param = 9000
    poisson_param = 15
    
    b = np.array([1,2])
    x1 = np.random.exponential(scale=exp_param,size=nobs)
    x2 = np.random.poisson(lam=poisson_param,size=nobs)

    X = np.vstack((x1,x2)).T

    epsilon = np.random.normal(0,1,nobs)

    y = np.dot(X,b) + epsilon

    data = {
        'X': X,
        'y': y,
        'epsilon':epsilon,
        'beta': b

    }
    
    return data



def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    N = 1000
    data = simulate_data(N)
    X = data['X']
    y = data['y']
    beta = data['beta']
    epsilon = data['epsilon']

    ## Statsmodels regression
    model_sm = sm.OLS(y,X)
    results_sm = model_sm.fit()
    params_sm = results_sm.params


    # Sklearn regression
    method_skl_ols = LinearRegression()
    results_skl = method_skl_ols.fit(X,y)
    params_skl = method_skl_ols.coef_

    results = pd.DataFrame(
        {
            'statsmodels': params_sm,
            'sklearn': params_skl
        }
    )

    return results

    


def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###