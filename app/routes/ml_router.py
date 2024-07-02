"""API routers for machine learning functions"""

from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import ml_service as funs
from app.routes.common_router_functions import Data, convert_to_df
from sklearn.model_selection import train_test_split

router = APIRouter()


class model_params(BaseModel):
    data: Data
    indep_var: str
    dep_vars: List[str]
    train_size: float = 0.8
    split_seed: int = -1


def split_df(model_params: model_params):
    """Split the data into training and testing sets"""

    df = convert_to_df(model_params.data)

    if model_params.train_size <= 0.0 or model_params.train_size > 1.0:
        return {"message": "Train size must be in (0, 1]"}
    elif model_params.train_size == 1.0:
        df_train = df
        df_test = None
    elif model_params.train_size * len(df) < 2:
        return {"message": "Train size too small"}
    elif (1 - model_params.train_size) * len(df) < 2:
        return {"message": "Train size too large"}
    else:
        if model_params.split_seed == -1:
            random_state = None
        elif model_params.split_seed < 0:
            return {"message": "Split seed must be non-negative"}
        else:
            random_state = model_params.split_seed
        df_train, df_test = train_test_split(df, train_size=model_params.train_size, random_state=random_state)

    return df_train, df_test


@router.post("/linear_regression")
async def linear_regression(model_params: model_params,
                            fit_intercept: bool = True,
                            positive: bool = False):
    """Linear regression model router"""

    try:
        df_train, df_test = split_df(model_params)
    except ValueError:
        return split_df(model_params)

    results = funs.linear_regression(df_train, df_test,
                                     model_params.indep_var,
                                     model_params.dep_vars,
                                     fit_intercept,
                                     positive)

    return {"message": "Data received", "data": results}
