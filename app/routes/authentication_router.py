from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from firebase_admin import auth
from app.firebase_init import db

router = APIRouter()

class User(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def signup(user: User):
    try:
        user_record = auth.create_user(
            email=user.email,
            password=user.password
        )
        return {"id": user_record.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(user: User):
    print(user)
    try:
        user_record = auth.get_user_by_email(user.email)
        print(user_record)
        return {"user": {"id": user_record.uid, "email": user_record.email}}
    except auth.UserNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid email")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/logout")
async def logout():
    return {"message": "Logged out"}

@router.post("/verify_token")
async def verify_token(token: str):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
