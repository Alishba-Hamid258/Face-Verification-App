import streamlit as st
import requests
import cv2
import numpy as np
import time
import queue
import threading
from config import API_PORT

API_URL = f"http://127.0.0.1:{API_PORT}"

st.title("Face Verification System")

try:
    response = requests.get(f"{API_URL}/politicians", timeout=10)
    known_names = response.json()["politicians"]
    st.write(f"Known faces: {known_names}")
except Exception as e:
    st.error(f"Error loading known faces: {e}")
    st.stop()

st.subheader("Upload Image for Face Verification")
uploaded_file = st.file_uploader("Choose an image (.jpg, .png)", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        response = requests.post(f"{API_URL}/verify-image", files={"file": uploaded_file}, timeout=10)
        result = response.json()
        if result["matched"]:
            st.success(f"Matched: {result['name']} (Distance: {result['distance']:.2f})")
            st.write(f"Description: {result['description']}")
            st.write(f"Party: {result['party']}")
        else:
            st.warning("No match found.")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# --- Admin Login Sidebar ---
st.sidebar.title("Admin Login")
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
    st.session_state.admin_username = ""
    st.session_state.admin_password = ""

if not st.session_state.admin_logged_in:
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Login")
        if login_submitted:
            st.session_state.admin_username = username
            st.session_state.admin_password = password
            st.session_state.admin_logged_in = True
            st.rerun()
else:
    st.sidebar.success(f"Logged in as {st.session_state.admin_username}")
    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.session_state.admin_username = ""
        st.session_state.admin_password = ""
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
            try:
                files = [("images", image) for image in new_images]
                data = {"name": new_name, "description": new_description, "party": new_party}
                response = requests.post(f"{API_URL}/add-politician", data=data, files=files, auth=(st.session_state.admin_username, st.session_state.admin_password), timeout=10)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(response.json().get("detail", "Authentication Failed"))
            except Exception as e:
                st.error(f"Error adding person: {e}")

    st.subheader("Edit an Existing Person")
    with st.form(key="edit_person_form"):
        old_name = st.selectbox("Select Person to Edit", known_names)
        new_name = st.text_input("New Name", value=old_name if old_name else "")
        new_description = st.text_input("New Description")
        new_party = st.text_input("New Party")
        new_images = st.file_uploader("Upload New Images (Optional)", type=["jpg", "png"], accept_multiple_files=True)
        submit_edit = st.form_submit_button(label="Edit Person")

        if submit_edit and old_name:
            try:
                files = [("images", image) for image in new_images] if new_images else []
                data = {
                    "old_name": old_name,
                    "new_name": new_name,
                    "new_description": new_description,
                    "new_party": new_party,
                }
                response = requests.post(f"{API_URL}/edit-politician", data=data, files=files, auth=(st.session_state.admin_username, st.session_state.admin_password), timeout=10)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(response.json().get("detail", "Authentication Failed"))
            except Exception as e:
                st.error(f"Error editing person: {e}")

    st.subheader("Delete a Person")
    with st.form(key="delete_person_form"):
        delete_name = st.selectbox("Select Person to Delete", known_names)
        submit_delete = st.form_submit_button(label="Delete Person")

        if submit_delete and delete_name:
            try:
                data = {"name": delete_name}
                response = requests.post(f"{API_URL}/delete-politician", data=data, auth=(st.session_state.admin_username, st.session_state.admin_password), timeout=10)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                    response = requests.get(f"{API_URL}/politicians", timeout=10)
                    st.session_state.known_names = response.json()["politicians"]
                else:
                    st.error(response.json().get("detail", "Authentication Failed"))
            except Exception as e:
                st.error(f"Error deleting person: {e}")
else:
    st.info("🔒 Please log in via the sidebar to Add, Edit, or Delete people in the database.")

@st.cache_resource
def get_camera_manager():
    class CameraManager:
        def __init__(self):
            self.cap = None
            self.thread = None
            self.run_flag = [False]
            self.frame_queue = queue.Queue()

        def capture_frames(self):
            while self.run_flag[0] and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                self.frame_queue.put(frame)
                time.sleep(0.1)

        def start(self):
            if self.cap and self.cap.isOpened():
                return True
            for idx in range(3):
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    break
            else:
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.run_flag[0] = True
            self.thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.thread.start()
            return True

        def stop(self):
            self.run_flag[0] = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.thread:
                self.thread.join(timeout=1.0)
            self.cap = None
            self.thread = None
            while not self.frame_queue.empty():
                self.frame_queue.get()

    return CameraManager()

cam_manager = get_camera_manager()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

st.subheader("Camera Controls")

if not st.session_state.camera_running:
    if st.button("Start Camera"):
        if cam_manager.start():
            st.session_state.camera_running = True
            st.rerun()
        else:
            st.error("Could not open camera. Check if another app is using it or enable camera permissions in Windows.")
else:
    if st.button("Stop Camera"):
        st.session_state.camera_running = False
        cam_manager.stop()
        st.success("Camera stopped.")
        st.rerun()

frame_placeholder = st.empty()
match_placeholder = st.empty()

if st.session_state.camera_running:
    frame_count = 0
    while st.session_state.camera_running:
        if not cam_manager.frame_queue.empty():
            frame = cam_manager.frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            frame_count += 1
            if frame_count % 20 == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                try:
                    response = requests.post(f"{API_URL}/verify-image", files=files, timeout=15)
                    result = response.json()
                    if result["matched"]:
                        match_placeholder.success(f"**Matched:** {result['name']} (Distance: {result['distance']:.2f})\n\n**Party:** {result['party']}\n\n**Description:** {result['description']}")
                    else:
                        match_placeholder.warning("No match found.")
                except requests.exceptions.ReadTimeout:
                    match_placeholder.error("Server timed out. Please wait or check server status.")
                except Exception as e:
                    match_placeholder.error(f"Error sending frame: {e}")
        time.sleep(0.01)