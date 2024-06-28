import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
from app.services import functions as funs


class DataRow(BaseModel):
    columns: List

class Data(BaseModel):
    data: List[DataRow]


def convert_to_df(data:Data):
    # Extract data
    rows = [row.columns for row in data.data]

    # Convert to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming the first row contains headers
    print(df)
    return df
