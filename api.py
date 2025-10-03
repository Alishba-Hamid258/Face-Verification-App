from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import face_recognition
import numpy as np
import pickle
from pymongo import MongoClient
from datetime import datetime
from typing import List
from PIL import Image
import io
import time
import cv2
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION, CACHE_DURATION

app = FastAPI()

# MongoDB client (initialized on startup)
client = None
db = None
collection = None

# Cached embeddings
cached_encodings = None
cached_names = None
last_cache_update = 0

@app.on_event("startup")
async def startup_event():
    global client, db, collection
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        print("MongoDB connection opened")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global client
    if client is not None:
        client.close()
        print("MongoDB connection closed")

def load_embeddings():
    global cached_encodings, cached_names, last_cache_update
    current_time = time.time()
    if cached_encodings is None or cached_names is None or (current_time - last_cache_update) > CACHE_DURATION:
        cached_encodings = []
        cached_names = []
        for doc in collection.find():
            if 'face_embedding' in doc:
                cached_names.append(doc['name'])
                cached_encodings.append(pickle.loads(bytes.fromhex(doc['face_embedding'])))
        last_cache_update = current_time
    return cached_encodings, cached_names

@app.post("/add-politician")
async def add_politician(
    name: str = Form(...),
    description: str = Form(...),
    party: str = Form(...),
    images: List[UploadFile] = File(...),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    
    encodings = []
    image_sources = []
    for image in images:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        rgb_image = np.array(pil_image)
        rgb_image = cv2.resize(rgb_image, (160, 120))
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            face_locations = face_recognition.face_locations(rgb_image, model="cnn")
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                encodings.append(face_encodings[0])
                image_sources.append(image.filename)
    
    if not encodings:
        raise HTTPException(status_code=400, detail="No faces detected in any uploaded images.")

    avg_encoding = np.mean(encodings, axis=0)
    collection.update_one(
        {'name': name},
        {'$set': {
            'face_embedding': pickle.dumps(avg_encoding).hex(),
            'details.image_sources': image_sources,
            'details.image_count': len(encodings),
            'details.updated_at': datetime.now().strftime('%Y-%m-%d'),
            'description': description,
            'party': party
        }},
        upsert=True
    )
    return {"status": "success", "message": f"Added {name} with {len(encodings)} images."}

@app.post("/edit-politician")
async def edit_politician(
    old_name: str = Form(...),
    new_name: str = Form(...),
    new_description: str = Form(...),
    new_party: str = Form(...),
    images: List[UploadFile] = File(None),
):
    if images:
        encodings = []
        image_sources = []
        for image in images:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
            rgb_image = np.array(pil_image)
            rgb_image = cv2.resize(rgb_image, (160, 120))
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            if not face_locations:
                face_locations = face_recognition.face_locations(rgb_image, model="cnn")
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    image_sources.append(image.filename)
        
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            embedding_hex = pickle.dumps(avg_encoding).hex()
        else:
            embedding_hex = None
    else:
        embedding_hex = None

    update_fields = {
        'name': new_name,
        'description': new_description,
        'party': new_party,
        'details.updated_at': datetime.now().strftime('%Y-%m-%d')
    }
    if embedding_hex:
        update_fields['face_embedding'] = embedding_hex
        update_fields['details.image_sources'] = image_sources
        update_fields['details.image_count'] = len(encodings)
    
    result = collection.update_one(
        {'name': old_name},
        {'$set': update_fields}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Politician not found.")
    return {"status": "success", "message": f"Edited {old_name} to {new_name}."}

@app.post("/verify-image")
async def verify_image(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        rgb_image = np.array(pil_image)
        rgb_image = cv2.resize(rgb_image, (160, 120))
        print(f"Resized image shape: {rgb_image.shape}")

        known_encodings, known_names = load_embeddings()
        print(f"Loaded {len(known_encodings)} embeddings")

        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            face_locations = face_recognition.face_locations(rgb_image, model="cnn")
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected")

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face encoding detected")
        face_encoding = face_encodings[0]

        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                doc = collection.find_one({'name': name})
                print(f"Verification took {time.time() - start_time:.2f} seconds")
                return {
                    "matched": True,
                    "name": name,
                    "description": doc.get('description'),
                    "party": doc.get('party'),
                    "distance": face_distances[best_match_index]
                }
        print(f"Verification took {time.time() - start_time:.2f} seconds")
        return {"matched": False, "name": "Unknown", "distance": min(face_distances) if face_distances.size else None}
    except Exception as e:
        print(f"Error in verify_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/politicians")
async def get_politicians():
    known_encodings, known_names = load_embeddings()
    return {"politicians": known_names}

@app.post("/delete-politician")
async def delete_politician(name: str = Form(...)):
    result = collection.delete_one({'name': name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Politician not found.")
    return {"status": "success", "message": f"Deleted {name}."}

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)