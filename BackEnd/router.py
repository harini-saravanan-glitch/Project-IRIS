from fastapi import APIRouter
from pydantic import BaseModel
import json
from google.cloud import firestore

# Initialize Firestore client
db = firestore.Client.from_service_account_json(r'serviceAccountKey.json')
router = APIRouter()

class wishList(BaseModel):
    email:str
    Full_Name : str

@router.post('/wishlist')
@router.post('/wishlist')
def add_wishlist(body: wishList):
    doc_ref = db.collection('wishlist').document(body.email)
    doc = doc_ref.get()
    if doc.exists:
        return {"message": "Wishlist already exists for this email", "data": doc.to_dict()}
    data = body.dict()
    doc_ref.set(data)
    return {"message": "Wishlist added successfully", "data": data}

