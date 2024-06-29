"""Machine learning functions"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression(df, fit_intercept=True, positive=False):
    """Linear regression model"""
    regressor = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    regressor.fit(df["target"], df.drop(columns=["target"]))
    return
