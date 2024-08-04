"""Common functions for API routers"""

from typing import List
from pydantic import BaseModel
import pandas as pd,re, numpy as np


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

    # Convert to dataframe (assuming the first row contains headers)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace({None: np.nan})

    print(df)
    print(df.dtypes)

    return df
