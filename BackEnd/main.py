from fastapi import FastAPI
from pydantic import BaseModel
from router import router
from google.cloud import firestore

app = FastAPI()
app.include_router(router)

