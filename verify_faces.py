import cv2
import face_recognition
import numpy as np
import pickle
from pymongo import MongoClient
import time
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION

# Connect to MongoDB and load encodings
try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION]
    known_encodings = []
    known_names = []
    for doc in collection.find():
        if 'face_embedding' in doc:
            known_names.append(doc['name'])
            known_encodings.append(pickle.loads(bytes.fromhex(doc['face_embedding'])))
    if not known_encodings:
        print("No embeddings found in MongoDB. Run update_embeddings.py first.")
        client.close()
        exit(1)
except Exception as e:
    print(f"Error connecting to/loading MongoDB: {e}")
    client.close()
    exit(1)

# Try laptop camera indices
camera_indices = [0, 1]
video_capture = None
for index in camera_indices:
    video_capture = cv2.VideoCapture(index)
    if video_capture.isOpened():
        print(f"Laptop camera opened successfully at index {index}")
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        break
    video_capture.release()
if video_capture is None or not video_capture.isOpened():
    print("Error: Could not open laptop camera. Check privacy settings, drivers, or conflicts.")
    print("Steps: 1) Settings > Privacy > Camera > Enable access. 2) Update drivers. 3) Restart laptop.")
    client.close()
    exit(1)

print(f"Starting face verification with {len(known_names)} known faces: {known_names}")
start_time = time.time()
frame_count = 0
max_runtime = 300

while time.time() - start_time < max_runtime:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame. Restarting capture...")
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
        if not face_locations:
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn", number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        print(f"Error in face detection: {e}")
        continue

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                print(f"Matched {name} with distance: {face_distances[best_match_index]:.2f}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
client.close()
print("MongoDB connection closed")
print("Verification session ended")