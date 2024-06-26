import pandas as pd
from fastapi import APIRouter, File, UploadFile
#from app.client import supabase
import io
from pydantic import BaseModel
from typing import Dict, List, Union
import csv
from io import StringIO, BytesIO
from app.services import functions as funs
from fastapi.middleware.cors import CORSMiddleware

router = APIRouter()




class DataRow(BaseModel):
    columns: List

class OneHotDefs(BaseModel):
    OneHotDefs : Dict 

class Data(BaseModel):
    data: List[DataRow]

@router.post("/onehotencoding")
async def one_hotenconding(data: Data,
                           column_name):
    
    # column_defs = column_defs.OneHotDefs
    # print(column_defs)

    df = convert_to_df(data)

    df = funs.onehotEncoding(df, column_name)

    print(df.to_json(orient='records'))

    return {"message": "Data received", "data": df.to_json(orient='records')}


@router.get("/test")
async def test(k):
    print(k)
    return k
    

def convert_to_df(data:Data):
    # Extract data
    rows = [row.columns for row in data.data]

    # Convert to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming the first row contains headers
    print(df)
    df.head().to_csv("x.csv")
    return df