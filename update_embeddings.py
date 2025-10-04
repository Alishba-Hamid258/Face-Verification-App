import face_recognition
import os
import numpy as np
import pickle
from pymongo import MongoClient
from datetime import datetime
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION

# Absolute path to dataset
dataset_path = r"C:\Users\USER\Projects\FaceVerificationAppPoliticians\dataset"

# Name mapping to match folder names with MongoDB names and details
name_mapping = {
    "asif_ali_zardari": {"name": "Asif Ali Zardari", "description": "Former President of Pakistan", "party": "Pakistan Peoples Party (PPP)"},
    "bilawal_bhutto_zardari": {"name": "Bilawal Bhutto Zardari", "description": "Chairman of PPP", "party": "Pakistan Peoples Party (PPP)"},
    "imran_khan": {"name": "Imran Khan", "description": "Former Prime Minister and cricket legend", "party": "Pakistan Tehreek-e-Insaf (PTI)"},
    "maryam_nawaz": {"name": "Maryam Nawaz", "description": "Politician and leader of PML-N", "party": "Pakistan Muslim League (N)"},
    "nawaz_sharif": {"name": "Nawaz Sharif", "description": "Former Prime Minister", "party": "Pakistan Muslim League (N)"},
    "Shehbaz Sharif": {"name": "Shehbaz Sharif", "description": "Prime Minister of Pakistan", "party": "Pakistan Muslim League (N)"},
    "erdogan": {"name": "Recep Tayyip Erdogan", "description": "President of Turkey", "party": "Justice and Development Party (AKP)"}
}

# Verify dataset folder exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset folder not found at {dataset_path}")
    exit(1)

# Connect to MongoDB
try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION]
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Process each politician's folder
updates_performed = 0
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        encodings = []
        image_sources = []
        print(f"Processing {person_name}...")
        for image_file in os.listdir(person_folder):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Support more image formats
                image_path = os.path.join(person_folder, image_file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    if face_locations:
                        face_enc = face_recognition.face_encodings(image, face_locations)[0]
                        encodings.append(face_enc)
                        image_sources.append(image_path)
                        print(f"Face detected in {image_path}")
                    else:
                        print(f"No face detected in {image_path} with HOG. Trying CNN...")
                        face_locations_cnn = face_recognition.face_locations(image, model="cnn")
                        if face_locations_cnn:
                            face_enc = face_recognition.face_encodings(image, face_locations_cnn)[0]
                            encodings.append(face_enc)
                            image_sources.append(image_path)
                            print(f"Face detected with CNN in {image_path}")
                        else:
                            print(f"No face detected with CNN in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            db_info = name_mapping.get(person_name)
            if db_info:
                db_name = db_info["name"]
                description = db_info["description"]
                party = db_info["party"]
                try:
                    result = collection.update_one(
                        {'name': db_name},
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
                    if result.modified_count > 0 or result.upserted_id:
                        updates_performed += 1
                        print(f"Successfully updated/inserted document for {db_name}")
                    else:
                        print(f"No change for {db_name}. Document may already have this embedding.")
                except Exception as e:
                    print(f"Error updating MongoDB for {db_name}: {e}")
            else:
                print(f"No name mapping for {person_name}. Skipping MongoDB update.")
        else:
            print(f"No valid encodings for {person_name}. Ensure images contain clear faces.")
    else:
        print(f"Skipping {person_name}: Not a directory")

# Print summary
print(f"Total updates performed: {updates_performed}")
try:
    print(f"Current MongoDB names: {list(collection.distinct('name'))}")
except Exception as e:
    print(f"Error retrieving distinct names: {e}")

# Close MongoDB connection
client.close()
print("MongoDB connection closed")