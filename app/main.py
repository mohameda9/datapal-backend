"""Main backend file. Initialize API and routes."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import processing_router, ml_router, stats_router, storage_router

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3000/"
]
print(origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(processing_router.router)
app.include_router(ml_router.router)
app.include_router(stats_router.router)
app.include_router(storage_router.router)


# TODO: save model to db, make it reusable: app.post("/training-environment")
