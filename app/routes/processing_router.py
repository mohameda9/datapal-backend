"""API routers for feature engineering functions"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import processing_service as funs
from app.routes.common_router_functions import Data, convert_to_df
from sklearn.model_selection import train_test_split
import pandas as pd
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
async def executeWorkflow(data: Data, workflow: List[Dict], testData: Optional[Data]=[]):
    # Extract relevant information
    print(workflow)

    print(testData)


    df = convert_to_df(data)
    if testData.data !=[]:
        test_df = convert_to_df(testData)
    else:
        test_df = None

    for action in workflow:
        print(action)
        if action.get("executed", False):
            print("already executed")
            continue
        if action["title"] == "One Hot Encoding":
            df = funs.one_hot_encoding(df, action["APIdata"]["columnName"])

        elif action["title"] == "Create New Column":
            columnCreationInput = funs.preprocess_column_creation_input(action["APIdata"])
            df = funs.newColumn_from_condition(columnCreationInput, df)

        elif action["title"] == "Partition Data":
            train_size = action["APIdata"]["splitSize"]
            if train_size == 1 or train_size == 0:
                continue
            df, test_df = train_test_split(df, test_size=1 - train_size, random_state=42)
            print(df.shape, test_df.shape)

        elif action["title"] == "Categorical Labeling":
            update_dict = action["APIdata"]
            columnToUpdate = action["APIdata"]["selectedColumn"]
            newColumnName = action["APIdata"]["newColumnName"]
            clean_mapping = {category: pair['value'] for pair in update_dict['categoryValuePairs'] for category in pair['selectedCategories']}
            default_value = update_dict['defaultValue']
            if default_value is None:
                df[newColumnName] = df[columnToUpdate].map(clean_mapping).fillna(df[columnToUpdate])
            else:
                df[newColumnName] = df[columnToUpdate].map(clean_mapping).fillna(default_value)
            print(df)

        elif action["title"] == "Scale Column":
            column_name = action["APIdata"]["columnName"]
            method = action["APIdata"]["method"]
            fit_on_train = action["APIdata"].get("fitOnTrain", True)
            kwargs = {k: v for k, v in action["APIdata"].items() if k not in ["columnName", "method", "fitOnTrain"]}
            df, test_df = funs.scaleColumn(df, test_df, column_name, method, fit_on_train, **kwargs)
        
        if test_df is None:
            testData_toSend = pd.DataFrame({}).to_json(orient='records')
        else: testData_toSend = test_df.to_json(orient='records')
    return {"message": "Data received", "data": df.to_json(orient='records'), "testData":testData_toSend}



