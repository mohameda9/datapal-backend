from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from bson import ObjectId
import motor.motor_asyncio

router = APIRouter()

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')
db = client.datapal

class DataInstance(BaseModel):
    id: str = None
    name: str
    data: list
    numdisplayedRows: int
    totalRows: int
    dataTypes: dict
    workflow: list
    isCollapsed: bool

def fix_id(instance):
    instance["id"] = str(instance["_id"])
    del instance["_id"]
    return instance

@router.get("/api/instances", response_model=List[DataInstance])
async def get_instances():
    instances = await db.instances.find().to_list(1000)
    return [fix_id(instance) for instance in instances]

@router.post("/api/instances", response_model=DataInstance)
async def create_instance(instance: DataInstance):
    instance_dict = instance.dict()
    instance_dict["_id"] = str(ObjectId())
    await db.instances.insert_one(instance_dict)
    return fix_id(instance_dict)

@router.put("/api/instances/{instance_id}", response_model=DataInstance)
async def update_instance(instance_id: str, instance: DataInstance):
    updated_instance = await db.instances.find_one_and_update(
        {"_id": ObjectId(instance_id)},
        {"$set": instance.dict(exclude={"id"})},
        return_document=True
    )
    if updated_instance:
        return fix_id(updated_instance)
    else:
        raise HTTPException(status_code=404, detail="Instance not found")

@router.delete("/api/instances/{instance_id}")
async def delete_instance(instance_id: str):
    delete_result = await db.instances.delete_one({"_id": ObjectId(instance_id)})
    if delete_result.deleted_count == 1:
        return {"message": "Instance deleted"}
    else:
        raise HTTPException(status_code=404, detail="Instance not found")

@router.delete("/api/instances")
async def delete_all_instances():
    delete_result = await db.instances.delete_many({})
    return {"message": f"Deleted {delete_result.deleted_count} instances"}
