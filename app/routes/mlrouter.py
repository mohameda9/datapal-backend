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



class modelSettings(BaseModel):
    a: List[str]

@router.post("/newtest")
def test(settings:modelSettings, k):
    print(settings.a)
    return k







