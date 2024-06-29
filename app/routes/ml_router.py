"""API routers for machine learning functions"""

from typing import Dict, List
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from app.services import ml_service as funs
from app.routes.common_router_functions import convert_to_df

router = APIRouter()


@router.get("/hello_ml")
async def hello_ml(x):
    """Test router"""
    print("Hello ML:", x)
    return x


@router.post("/linear_regression")
async def linear_regression():
    """Linear regression model"""
    funs.linear_regression()
    return None
