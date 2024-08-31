"""API routers for feature engineering functions"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import processing_service as funs
from app.routes.common_router_functions import Data, convert_to_df
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
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


@router.post("/getColumnDescriptiveStats")
async def getColumnDescriptiveStats(data: Data, column: str):
    df = convert_to_df(data)
    return funs.calculateColumnStats(df, column)
    


@router.post("/columnCreation")
async def createNewColumn(data: Data, columnCreationInput: Dict[str, Any]):
    # Extract relevant informatioon



    df = convert_to_df(data)

    columnCreationInput = funs.preprocess_column_creation_input(columnCreationInput)

    df = funs.newColumn_from_condition(columnCreationInput, df)
    #df.fillna('', inplace = True)

    return {"message": "Data received", "data": df.to_json(orient='records')}

@router.post("/executeWorkflow")
async def executeWorkflow(data: Data, workflow: List[Dict], testData: Optional[Data] = []):
    df = convert_to_df(data)
    test_df = convert_to_df(testData) if testData.data else None
    response_data = {
        "workflow_results": [],
        "column_types": {}
    }

    for action in workflow:
        workflow_result = {}

        try:
            if action["APIdata"].get("executed", False):
                response_data["workflow_results"].append(workflow_result)
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
            elif action["title"] == "Scale Column":
                column_name = action["APIdata"]["columnName"]
                method = action["APIdata"]["method"]
                fit_on_train = action["APIdata"].get("fitOnTrain", True)
                kwargs = {k: v for k, v in action["APIdata"].items() if k not in ["columnName", "method", "fitOnTrain"]}
                df, test_df = funs.scaleColumn(df, test_df, column_name, method, fit_on_train, **kwargs)
            elif action["title"] == "Handle Missing Values":
                column = action["APIdata"]["column"]
                method = action["APIdata"]["method"]
                value = action["APIdata"].get("value", None)
                group_by = action["APIdata"].get("group_by", None)
                consider_nan_as_category = action["APIdata"].get("consider_nan_as_category", False)
                interpolate_col = action["APIdata"].get("interpolate_col", None)
                fit_to_train = action["APIdata"].get("fit_to_train", True)

                df, test_df = funs.handle_missing_values(df, column, method, value, group_by, interpolate_col, consider_nan_as_category, fit_to_train, test_df)
            elif action["title"] == "Describe Data":
                describe_result = {}
                for column in df.columns:
                    column_description = funs.calculateColumnStats(df, column)
                    describe_result[column] = column_description
                workflow_result["description_metadata"] = describe_result
            elif action["title"] == "Discretize Column":
                column_name = action["APIdata"]["selectedColumn"]
                intervals = action["APIdata"]["intervals"]
                df = funs.discretize_column(df, column_name, intervals)

            workflow_result["executed"] = True
        except Exception as e:
            workflow_result["error"] = str(e)

        response_data["workflow_results"].append(workflow_result)

    # Determine the column types after all workflows have been applied
    response_data["column_types"] = funs.determine_column_type(df)

    if test_df is None:
        testData_toSend = pd.DataFrame({}).to_json(orient='records')
    else:
        testData_toSend = test_df.to_json(orient='records')

    response_data["data"] = df.to_json(orient='records')
    response_data["testData"] = testData_toSend

    print(response_data["workflow_results"])

    return response_data
