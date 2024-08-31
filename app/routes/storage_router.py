from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.firebase_init import db, bucket
from io import StringIO
import csv
import pandas as pd

router = APIRouter()

class DataInstance(BaseModel):
    id: Optional[int] = None
    name: str
    data: list
    testData: list = []
    numdisplayedRows: int
    totalRows: int
    dataTypes: dict
    workflow: list
    isCollapsed: bool


class Project(BaseModel):
    name: str
    user_id: str



def convert_to_csv(data):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()

def clean_dict(d):
    """ Recursively clean dictionary by removing keys with empty string or None values """
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if k and v is not None}
    elif isinstance(d, list):
        return [clean_dict(i) for i in d]
    else:
        return d

@router.get("/loadProject/{user_id}/{project_id}", response_model=List[DataInstance])
async def get_instances(user_id: str, project_id: str):
    print('loading')
    instances_ref = db.collection('users').document(user_id).collection('projects').document(project_id).collection('datainstances')
    instances = instances_ref.stream()
    result = []
    for instance in instances:
        instance_data = instance.to_dict()
        data_blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_data.csv")
        testdata_blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_testData.csv")

        if data_blob.exists():
            data_csv = data_blob.download_as_text()
            instance_data['data'] = convert_data_types(data_csv, instance_data.get('dataTypes', {}))

        if testdata_blob.exists():
            testdata_csv = testdata_blob.download_as_text()
            instance_data['testData'] = convert_data_types(testdata_csv, instance_data.get('dataTypes', {}))
        
        result.append(instance_data)
    return result

def convert_data_types(csv_text, data_types):
    df = pd.read_csv(StringIO(csv_text))
    
    for column, col_type in data_types.items():
        if col_type == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif col_type == 'numeric binary':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif col_type in ['categorical', 'categorical binary']:
            df[column] = df[column].astype(str)
            df[column] = df[column].replace('nan', pd.NA)

    
    # Replace NaNs with None and convert to list of lists
    df = df.where(pd.notnull(df), None)
    return [df.columns.tolist()] + df.values.tolist()




@router.post("/saveProject/{user_id}/{project_id}")
async def saveProject(user_id: str, project_id: str, instances: List[DataInstance]):
    created_count = 0
    for idx, instance in enumerate(instances):
        instance_dict = instance.dict()
        instance_dict['id'] = idx

        # Convert data to CSV and upload to Firebase Storage
        csv_data = convert_to_csv(instance_dict.pop('data'))
        blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{idx}_data.csv")
        blob.upload_from_string(csv_data, content_type="text/csv")

        print("savedFile")

        if instance_dict['testData']:
            csv_test_data = convert_to_csv(instance_dict.pop('testData'))
            test_blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{idx}_testData.csv")
            test_blob.upload_from_string(csv_test_data, content_type="text/csv")
        print("savedTestFile")

        # Ensure all keys in instance_dict are valid Firestore keys
        valid_instance_dict = clean_dict(instance_dict)
        valid_instance_dict.pop('data', None)
        valid_instance_dict.pop('testData', None)

        instance_ref = db.collection('users').document(user_id).collection('projects').document(project_id).collection('datainstances').document(str(idx))
        instance_ref.set(valid_instance_dict)

        created_count += 1
    
    return {"message": f"{created_count} instances created successfully"}

@router.delete("/fetchfirebase/{user_id}/{project_id}")
async def delete_all_instances(user_id: str, project_id: str):
    instances = db.collection('users').document(user_id).collection('projects').document(project_id).collection('datainstances').stream()
    for instance in instances:
        db.collection('users').document(user_id).collection('projects').document(project_id).collection('datainstances').document(instance.id).delete()
        blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_data.csv")
        blob.delete()

        test_blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_testData.csv")
        if test_blob.exists():
            test_blob.delete()

    return {"message": "All instances deleted"}



@router.post("/create_project")
async def create_project(project: Project):
    print("attempting to create project:", project)

    user_projects_path = f"users/{project.user_id}/projects/"
    project_path = f"{user_projects_path}{project.name}/"

    # Check if the project already exists
    blobs = list(bucket.list_blobs(prefix=user_projects_path))
    existing_projects = set(blob.name.split('/')[3] for blob in blobs if blob.name.split('/')[2] == 'projects')

    if project.name in existing_projects:
        raise HTTPException(status_code=400, detail="Project already exists")

    # Create a placeholder file to create the project folder
    blob = bucket.blob(f"{project_path}placeholder.txt")
    blob.upload_from_string("This is a placeholder file to create the project folder.")

    return {"message": "Project created successfully"}



@router.get("/fetchfirebase/{user_id}")
async def get_user_projects(user_id: str):
    user_projects_path = f"users/{user_id}/projects/"
    blobs = list(bucket.list_blobs(prefix=user_projects_path))
    
    project_names = set()
    for blob in blobs:
        parts = blob.name.split('/')
        if len(parts) > 3 and parts[2] == 'projects':
            project_names.add(parts[3])
    
    projects = []
    for project_name in project_names:
        project_blobs = list(bucket.list_blobs(prefix=f"users/{user_id}/projects/{project_name}/"))
        if project_blobs:
            last_modified = max(blob.updated for blob in project_blobs)
            projects.append({
                'name': project_name,
                'last_modified': last_modified.isoformat()
            })
    
    return {"projects": projects}


@router.delete("/delete_project/{user_id}/{project_id}")
async def delete_project(user_id: str, project_id: str):
    project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
    
    # Delete all data instances within the project
    instances = project_ref.collection('datainstances').stream()
    for instance in instances:
        instance.reference.delete()
        blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_data.csv")
        if blob.exists():
            blob.delete()
        
        test_blob = bucket.blob(f"users/{user_id}/projects/{project_id}/datainstances/{instance.id}_testData.csv")
        if test_blob.exists():
            test_blob.delete()
    
    # Delete the project document
    project_ref.delete()
    
    # Delete project folder in storage
    blobs = bucket.list_blobs(prefix=f"users/{user_id}/projects/{project_id}/")
    for blob in blobs:
        blob.delete()
    
    return {"message": "Project deleted successfully"}