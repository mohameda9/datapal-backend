"""Machine learning functions"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression(df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      indep_var,
                      dep_vars,
                      fit_intercept=True,
                      positive=False):
    """Linear regression model"""

    # Setup the model
    regressor = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    results = {}  # "dep_var": [coefficient, intercept, train score, test score]

    # Extract independent variable
    X_train = df_train[indep_var].values.reshape(-1, 1)

    if df_test is not None:
        X_test = df_test[indep_var].values.reshape(-1, 1)

    # Fit the model to each dependent variable individually because was not able to get individual scores when fitting all at once
    for dep_var in dep_vars:
        y_train = df_train[dep_var].values.reshape(-1, 1)
        regressor.fit(X_train, y_train)

        if df_test is not None:
            y_test = df_test[dep_var].values.reshape(-1, 1)
            results[dep_var] = [float(regressor.coef_), float(regressor.intercept_), float(regressor.score(X_train, y_train)), float(regressor.score(X_test, y_test))]
        else:
            results[dep_var] = [float(regressor.coef_), float(regressor.intercept_), float(regressor.score(X_train, y_train)), None]

    return results
