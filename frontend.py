import streamlit as st
import numpy as np
import time
import pickle
from datetime import datetime
import io
from PIL import Image
import sys
import os
import subprocess
import streamlit as st

# 1. Inject the Streamlit Cloud venv directly into sys.path
venv_site_packages = "/home/adminuser/venv/lib/python3.11/site-packages"
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# 2. Setup dynamic fallback directory
deps_dir = "/tmp/deps"
os.makedirs(deps_dir, exist_ok=True)
if deps_dir not in sys.path:
    sys.path.insert(0, deps_dir)

# 3. Check and install missing packages dynamically
def install_fallback_dependencies():
    packages_to_install = []
    
    try:
        import pymongo
    except ImportError:
        packages_to_install.extend(["pymongo==4.7.3", "dnspython==2.6.1"])

    try:
        import dlib
    except ImportError:
        packages_to_install.append("dlib-bin==19.24.6")

    try:
        import face_recognition_models
    except ImportError:
        packages_to_install.append("face_recognition_models>=0.3.0")
        
    try:
        import click
    except ImportError:
        packages_to_install.append("Click>=6.0")

    if packages_to_install:
        st.warning(f"Installing missing core dependencies: {', '.join(packages_to_install)}. Please wait...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", "-t", deps_dir] + packages_to_install)
        except Exception as e:
            st.error(f"Failed to install dependencies: {e}")
            st.stop()

    try:
        import face_recognition
    except ImportError:
        st.warning("Installing face_recognition wrapper...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", "-t", deps_dir, "--no-deps", "face_recognition"])
        except Exception as e:
            st.error(f"Failed to install face_recognition: {e}")
            st.stop()

install_fallback_dependencies()

try:
    from pymongo import MongoClient
    import face_recognition
except Exception as e:
    st.error(f"Final import check failed: {e}")
    st.stop()

import numpy as np
import time
import pickle
from datetime import datetime
import io
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION, CACHE_DURATION

st.title("Face Verification System")

# --- Database & Caching ---
@st.cache_resource
def get_database():
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        return client, collection
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None, None

client, collection = get_database()

if collection is None:
    st.stop()

@st.cache_data(ttl=CACHE_DURATION)
def load_embeddings():
    cached_encodings = []
    cached_names = []
    try:
        for doc in collection.find():
            if 'face_embedding' in doc:
                cached_names.append(doc['name'])
                cached_encodings.append(pickle.loads(bytes.fromhex(doc['face_embedding'])))
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
    return cached_encodings, cached_names

# --- Helper Functions ---
def resize_with_aspect_ratio(image, width=None, height=None):
    if width is None and height is None:
        return image
    
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        
    resized_pil = pil_image.resize(dim, Image.Resampling.LANCZOS)
    return np.array(resized_pil)

def process_uploaded_image(uploaded_file):
    contents = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    rgb_image = np.array(pil_image)
    return verify_face_in_image(rgb_image)

def verify_face_in_image(rgb_image):
    rgb_image = resize_with_aspect_ratio(rgb_image, width=320)
    known_encodings, known_names = load_embeddings()
    
    if not known_encodings:
        return {"matched": False, "name": "Unknown", "distance": None, "error": "No known faces in database."}

    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    if not face_locations:
        face_locations = face_recognition.face_locations(rgb_image, model="cnn")
    if not face_locations:
        return {"matched": False, "name": "Unknown", "distance": None, "error": "No face detected"}

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if not face_encodings:
        return {"matched": False, "name": "Unknown", "distance": None, "error": "No face encoding detected"}
    
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    if any(matches):
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            doc = collection.find_one({'name': name})
            return {
                "matched": True,
                "name": name,
                "description": doc.get('description'),
                "party": doc.get('party'),
                "distance": face_distances[best_match_index]
            }
    return {"matched": False, "name": "Unknown", "distance": min(face_distances) if face_distances.size else None}


# --- UI ---
known_encodings, known_names = load_embeddings()
st.write(f"Known faces: {known_names}")

st.subheader("Upload Image for Face Verification")
uploaded_file = st.file_uploader("Choose an image (.jpg, .png)", type=["jpg", "png"])
if uploaded_file is not None:
    result = process_uploaded_image(uploaded_file)
    if result.get("error"):
        st.error(result["error"])
    elif result.get("matched"):
        st.success(f"Matched: {result['name']} (Distance: {result['distance']:.2f})")
        st.write(f"Description: {result['description']}")
        st.write(f"Party: {result['party']}")
    else:
        st.warning("No match found.")

# --- Admin Login Sidebar ---
st.sidebar.title("Admin Login")
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if not st.session_state.admin_logged_in:
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Login")
        if login_submitted:
            if username == "admin" and password == "secret123":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
else:
    st.sidebar.success("Logged in as admin")
    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()

# --- Admin Controls ---
if st.session_state.admin_logged_in:
    st.subheader("Add a New Person")
    with st.form(key="add_person_form"):
        new_name = st.text_input("Name")
        new_description = st.text_input("Description")
        new_party = st.text_input("Party")
        new_images = st.file_uploader("Upload Images for New Person", type=["jpg", "png"], accept_multiple_files=True)
        submit_button = st.form_submit_button(label="Add Person")

        if submit_button and new_images and new_name and new_description and new_party:
            encodings = []
            image_sources = []
            for image in new_images:
                contents = image.read()
                pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
                rgb_image = np.array(pil_image)
                rgb_image = resize_with_aspect_ratio(rgb_image, width=320)
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                if not face_locations:
                    face_locations = face_recognition.face_locations(rgb_image, model="cnn")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if face_encodings:
                        encodings.append(face_encodings[0])
                        image_sources.append(image.name)
            
            if not encodings:
                st.error("No faces detected in any uploaded images.")
            else:
                avg_encoding = np.mean(encodings, axis=0)
                collection.update_one(
                    {'name': new_name},
                    {'$set': {
                        'face_embedding': pickle.dumps(avg_encoding).hex(),
                        'details.image_sources': image_sources,
                        'details.image_count': len(encodings),
                        'details.updated_at': datetime.now().strftime('%Y-%m-%d'),
                        'description': new_description,
                        'party': new_party
                    }},
                    upsert=True
                )
                st.success(f"Added {new_name} with {len(encodings)} images.")
                load_embeddings.clear() # Clear cache
                time.sleep(1)
                st.rerun()

    st.subheader("Edit an Existing Person")
    with st.form(key="edit_person_form"):
        old_name = st.selectbox("Select Person to Edit", known_names)
        edit_new_name = st.text_input("New Name", value=old_name if old_name else "")
        edit_description = st.text_input("New Description")
        edit_party = st.text_input("New Party")
        edit_images = st.file_uploader("Upload New Images (Optional)", type=["jpg", "png"], accept_multiple_files=True)
        submit_edit = st.form_submit_button(label="Edit Person")

        if submit_edit and old_name:
            encodings = []
            image_sources = []
            if edit_images:
                for image in edit_images:
                    contents = image.read()
                    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
                    rgb_image = np.array(pil_image)
                    rgb_image = resize_with_aspect_ratio(rgb_image, width=320)
                    face_locations = face_recognition.face_locations(rgb_image, model="hog")
                    if not face_locations:
                        face_locations = face_recognition.face_locations(rgb_image, model="cnn")
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        if face_encodings:
                            encodings.append(face_encodings[0])
                            image_sources.append(image.name)
            
            update_fields = {}
            if edit_new_name: update_fields['name'] = edit_new_name
            if edit_description: update_fields['description'] = edit_description
            if edit_party: update_fields['party'] = edit_party
            update_fields['details.updated_at'] = datetime.now().strftime('%Y-%m-%d')
            
            if encodings:
                avg_encoding = np.mean(encodings, axis=0)
                update_fields['face_embedding'] = pickle.dumps(avg_encoding).hex()
                update_fields['details.image_sources'] = image_sources
                update_fields['details.image_count'] = len(encodings)
            
            result = collection.update_one({'name': old_name}, {'$set': update_fields})
            if result.matched_count == 0:
                st.error("Politician not found.")
            else:
                st.success(f"Edited {old_name}.")
                load_embeddings.clear()
                time.sleep(1)
                st.rerun()

    st.subheader("Delete a Person")
    with st.form(key="delete_person_form"):
        delete_name = st.selectbox("Select Person to Delete", known_names)
        submit_delete = st.form_submit_button(label="Delete Person")

        if submit_delete and delete_name:
            result = collection.delete_one({'name': delete_name})
            if result.deleted_count == 0:
                st.error("Politician not found.")
            else:
                st.success(f"Deleted {delete_name}.")
                load_embeddings.clear()
                time.sleep(1)
                st.rerun()
else:
    st.info("🔒 Please log in via the sidebar to Add, Edit, or Delete people in the database.")


# --- WebRTC Camera Controls ---
st.subheader("Live Camera Verification")
st.write("Click 'Start' to begin live verification using your camera.")

class FaceProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        self.frame_count += 1
        
        # Process every 15 frames to avoid lagging
        if self.frame_count % 15 == 0:
            self.last_result = verify_face_in_image(img)
            
        if self.last_result:
            from PIL import ImageDraw
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            
            if self.last_result.get("matched"):
                name = self.last_result["name"]
                distance = self.last_result["distance"]
                party = self.last_result.get("party", "")
                draw.text((10, 30), f"Match: {name} ({distance:.2f})", fill=(0, 255, 0))
                draw.text((10, 60), f"Party: {party}", fill=(0, 255, 0))
            elif not self.last_result.get("error"):
                draw.text((10, 30), "No Match", fill=(255, 0, 0))
                
            img = np.array(pil_img)
            
        return av.VideoFrame.from_ndarray(img, format="rgb24")

webrtc_streamer(
    key="face-verification",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)