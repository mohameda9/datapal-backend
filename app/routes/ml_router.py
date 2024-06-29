"""API routers for machine learning functions"""

from typing import Dict, List
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from app.services import ml_service as funs
from app.routes.common_router_functions import Data, convert_to_df

router = APIRouter()


class column_names(BaseModel):
    column_names: List[str]


@router.post("/linear_regression")
async def linear_regression(data: Data,
                            independent_variable: str,
                            dependent_variables: column_names,
                            fit_intercept: bool = True,
                            positive: bool = False):
    """Linear regression model router"""

    df = convert_to_df(data)
    results = funs.linear_regression(df,
                                     independent_variable,
                                     dependent_variables.column_names,
                                     fit_intercept,
                                     positive)

    return {"message": "Data received", "data": results}
