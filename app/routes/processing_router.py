"""API routers for feature engineering functions"""

from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import processing_service as funs
from app.routes.common_router_functions import Data, convert_to_df

router = APIRouter()


@router.get("/hello_processing")
async def hello_processing(x):
    """Test router"""
    print("Hello processing:", x)
    return x


class columnCreationInput(BaseModel):
    columnCreationInput: Dict


class OneHotDefs(BaseModel):
    """One-hot encoding definitions"""
    OneHotDefs: Dict


@router.post("/onehotencoding")
async def one_hot_encoding(data: Data, column_name):
    """One-hot encode categorical data"""

    # column_defs = column_defs.OneHotDefs
    # print(column_defs)

    df = convert_to_df(data)
    df = funs.one_hot_encoding(df, column_name)

    return {"message": "Data received", "data": df.to_json(orient="records")}


@router.post("/columnCreation")
async def createNewColumn(data: Data, columnCreationInput: Dict[str, Any]):
    # Extract relevant informatioon
    print(columnCreationInput)

    df = convert_to_df(data)

    columnCreationInput = funs.preprocess_column_creation_input(columnCreationInput)

    df = funs.newColumn_from_condition(columnCreationInput, df)
    print(df)
    df.fillna('', inplace = True)

    return {"message": "Data received", "data": df.to_json(orient='records')}


@router.post("/scale")
async def scale_column(data: Data, column_name, method,
                       new_min: int = 0, new_max: int = 1):
    """Scale a column in a dataset"""

    # column_defs = column_defs.OneHotDefs
    # print(column_defs)
    df = convert_to_df(data)

    if method == "normalize":
        df = funs.normalize_column(df, column_name,
                                   new_min=new_min, new_max=new_max)
    elif method == "standardize":
        pass
    else:
        raise f"{method} not a valid method"

    return {"message": "Data received", "data": df.to_json(orient="records")}
