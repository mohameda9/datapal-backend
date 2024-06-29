"""Common functions for API routers"""

from typing import Dict, List
from pydantic import BaseModel
import pandas as pd


class DataRow(BaseModel):
    """Row from a dataset"""
    columns: List


class Data(BaseModel):
    """Dataset table"""
    data: List[DataRow]


def convert_to_df(data: Data):
    """Convert dataset to a Pandas dataframe"""
    # Extract data
    rows = [row.columns for row in data.data]

<<<<<<< HEAD
    # Convert to dataframe (assuming the first row contains headers)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    print(df)
=======
    # Convert to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming the first row contains headers


>>>>>>> 8ec7a50 (updated column creation)
    return df
