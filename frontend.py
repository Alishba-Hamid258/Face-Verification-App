import streamlit as st
import os
import sys
import subprocess
import importlib
import time

# Ensure /tmp/deps is in path
deps_dir = "/tmp/deps"
os.makedirs(deps_dir, exist_ok=True)
if deps_dir not in sys.path:
    sys.path.insert(0, deps_dir)

# 1. Instant UI feedback
st.set_page_config(page_title="Face Verification", layout="wide")
st.title("Face Verification System")
status = st.empty()
status.info("🔍 Initializing System...")
print("[DEBUG] App started. Starting system check...")

# 2. Check for AI Engine
try:
    import face_recognition
    import face_recognition_models
    print("[DEBUG] AI Engine already loaded.")
except ImportError:
    print("[DEBUG] AI Engine missing. Starting installation...")
    status.warning("📦 Installing AI components... (This takes 60 seconds)")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", 
            "-t", deps_dir, "--no-deps", 
            "face_recognition", "face-recognition-models"
        ])
        importlib.invalidate_caches()
        print("[DEBUG] Installation successful. Rebooting app...")
        st.success("✅ Setup complete! Reloading...")
        time.sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Setup failed: {e}")
        print(f"[ERROR] Setup failed: {e}")
        st.stop()

# 3. Load other heavy libraries AFTER initialization
status.info("🔗 Loading libraries...")
import numpy as np
import pickle
import av
from datetime import datetime
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from config import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION, CACHE_DURATION

# 4. Database Connection
status.info("🔗 Connecting to Database...")
print("[DEBUG] Connecting to MongoDB...")

@st.cache_resource
def get_database():
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client[MONGODB_DB_NAME]
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        print(f"[ERROR] Database connection failed: {e}")
        return None

db = get_database()
if db is None:
    st.info("Check your MongoDB Atlas IP whitelist (0.0.0.0/0).")
    st.stop()

status.empty() # Clear the status message
print("[DEBUG] System ready.")

# --- Logic ---
@st.cache_data(ttl=CACHE_DURATION)
def load_embeddings():
    try:
        collection = db[MONGODB_COLLECTION]
        data = list(collection.find({}, {"name": 1, "embedding": 1}))
        names = [d["name"] for d in data]
        embeddings = [pickle.loads(d["embedding"]) for d in data]
        return names, embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return [], []

def get_face_embeddings(image_np):
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return None
    return face_recognition.face_encodings(image_np, face_locations)[0]

def verify_face(target_embedding, known_embeddings, tolerance=0.6):
    if not known_embeddings:
        return None, 0.0
    distances = face_recognition.face_distance(known_embeddings, target_embedding)
    min_distance_idx = np.argmin(distances)
    if distances[min_distance_idx] <= tolerance:
        return min_distance_idx, distances[min_distance_idx]
    return None, distances[min_distance_idx]

# --- Main UI ---
def main():
    # Sidebar Admin Login
    with st.sidebar:
        st.header("Admin Access")
        admin_user = st.text_input("Username")
        admin_pass = st.text_input("Password", type="password")
        is_admin = admin_user == "admin" and admin_pass == "secret123"

    # Main Tabs
    tab1, tab2 = st.tabs(["Verification", "Face Database"])

    with tab1:
        st.subheader("Live Verification")
        names, embeddings = load_embeddings()
        
        if not names:
            st.warning("Database is empty. Please add faces in the Database tab.")
        
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        webrtc_ctx = webrtc_streamer(
            key="verification",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_receiver:
            try:
                frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
                img = frame.to_ndarray(format="rgb24")
                
                target_emb = get_face_embeddings(img)
                if target_emb is not None:
                    idx, dist = verify_face(target_emb, embeddings)
                    if idx is not None:
                        st.success(f"Verified: {names[idx]} (Confidence: {1-dist:.2%})")
                    else:
                        st.error("Face not recognized")
                else:
                    st.info("No face detected in frame")
            except Exception:
                pass

    with tab2:
        if not is_admin:
            st.info("Please login as admin to manage faces")
        else:
            st.subheader("Manage Known Faces")
            collection = db[MONGODB_COLLECTION]
            
            # Add New Face
            with st.expander("Add New Face"):
                new_name = st.text_input("Name")
                uploaded_file = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'])
                
                if st.button("Add to Database"):
                    if new_name and uploaded_file:
                        image = Image.open(uploaded_file).convert('RGB')
                        img_np = np.array(image)
                        emb = get_face_embeddings(img_np)
                        
                        if emb is not None:
                            try:
                                collection.insert_one({
                                    "name": new_name,
                                    "embedding": pickle.dumps(emb),
                                    "created_at": datetime.now()
                                })
                                st.success(f"Added {new_name} to database!")
                                st.cache_data.clear()
                            except Exception as e:
                                st.error(f"Save failed: {e}")
                        else:
                            st.error("Could not find a face in the uploaded image")
                    else:
                        st.warning("Please provide both name and image")

            # Database View
            st.divider()
            names, _ = load_embeddings()
            for name in names:
                col1, col2 = st.columns([4, 1])
                col1.write(name)
                if col2.button("Delete", key=f"del_{name}"):
                    collection.delete_one({"name": name})
                    st.success(f"Deleted {name}")
                    st.cache_data.clear()
                    st.rerun()

if __name__ == "__main__":
    main()