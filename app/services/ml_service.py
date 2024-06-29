"""Machine learning functions"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression(df: pd.DataFrame, independent_variable, dependent_variables, fit_intercept=True, positive=False):
    """Linear regression model"""

    df = df.copy()

    # Setup the model
    regressor = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    results = {}

    # Extract independent variable
    X = df[independent_variable].values.reshape(-1, 1)

    # Fit the model to each dependent variable individually because was not able to get individual scores when fitting all at once
    for dependent_variable in dependent_variables:
        y = df[dependent_variable].values.reshape(-1, 1)
        regressor.fit(X, y)
        results[dependent_variable] = [float(regressor.coef_), float(regressor.intercept_), float(regressor.score(X, y))]

    return results
