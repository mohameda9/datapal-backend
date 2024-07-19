from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import processing_service as funs
from app.routes.common_router_functions import Data, convert_to_df
from app.services.stat_tests import *

router = APIRouter()

@router.post("/stat")
async def stat_analysis(data: Data, statConfig: dict):
    df = convert_to_df(data)
    result = handle_request(df, statConfig)
    print(result)
    return result


@router.post("/goodFit")
async def stat_analysis(data: Data, goodFitTestConfig: dict):
    df = convert_to_df(data)

    '''
    to complete
    
    '''
    return 
