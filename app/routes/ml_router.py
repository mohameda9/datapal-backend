from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from app.routes.common_router_functions import Data, convert_to_df
from app.services.clustering import run_clustering
import pandas as pd

router = APIRouter()

class ClusteringConfig(BaseModel):
    data: Data
    columns: list
    statConfig: Dict[str, Any]

@router.post("/clustering")
async def clustering_analysis(config: ClusteringConfig):
    df = convert_to_df(config.data)
    df = df[config.columns]  # Filter columns
    result = run_clustering(df, config.statConfig)
    return result
