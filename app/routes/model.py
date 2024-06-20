import pandas as pd
from fastapi import APIRouter, File, UploadFile
#from app.client import supabase
import io
from pydantic import BaseModel
from typing import Dict, List, Union
import csv
from io import StringIO, BytesIO
from app.services import functions as funs

router = APIRouter()


class DataRow(BaseModel):
    columns: List[str]

class OneHotDefs(BaseModel):
    OneHotDefs : Dict 

class Data(BaseModel):
    data: List[DataRow]

@router.post("/onehotencoding")
async def one_hotenconding(data: Data,
                           column_name, column_defs:OneHotDefs ):
    
    column_defs = column_defs.OneHotDefs
    print(column_defs)

    df = convert_to_df(data)

    df = funs.onehotEncoding(df, column_name, column_defs)

    print(df)
    return {"message": "Data received", "data": df.to_json(orient='records')}




def convert_to_df(data:Data):
    # Extract data
    rows = [row.columns for row in data.data]

    # Convert to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming the first row contains headers
    print(df)
    return df