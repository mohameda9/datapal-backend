"""API routers for feature engineering functions"""

from typing import List, Dict, Any
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

class WorkflowAction(BaseModel):
    title: str
    description: str
    data: Dict[str, Any]

class columnCreationInput(BaseModel):
    columnCreationInput: Dict


class OneHotDefs(BaseModel):
    """One-hot encoding definitions"""
    OneHotDefs: Dict

class Workflow(BaseModel):
    workflow:list



@router.post("/columnCreation")
async def createNewColumn(data: Data, columnCreationInput: Dict[str, Any]):
    # Extract relevant informatioon



    df = convert_to_df(data)

    columnCreationInput = funs.preprocess_column_creation_input(columnCreationInput)

    df = funs.newColumn_from_condition(columnCreationInput, df)
    #df.fillna('', inplace = True)

    return {"message": "Data received", "data": df.to_json(orient='records')}


@router.post("/executeWorkflow")
async def aa(data: Data, workflow: List[Dict]):
    # Extract relevant informatioon
    print(workflow)

    df = convert_to_df(data)

    for action in workflow:
        if action.get("executed", False):
            print("aaaa")
            continue
        print(action)
        if action["title"] == "One Hot Encoding":
            df = funs.one_hot_encoding(df, action["APIdata"]["columnName"])
        elif action["title"] =="Normalize Column":
            df = funs.normalize_column(df, action["APIdata"]["columnName"],
                                   new_min=action["APIdata"]["newMin"], new_max=action["APIdata"]["newMax"])
        elif action["title"] =="Create New Column":
            columnCreationInput = funs.preprocess_column_creation_input(action["APIdata"])
            df = funs.newColumn_from_condition(columnCreationInput, df)
        print(df)



    return {"message": "Data received", "data": df.to_json(orient='records')}



 







