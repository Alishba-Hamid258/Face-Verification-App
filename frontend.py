import streamlit as st
import requests
import cv2
import numpy as np
import time
import queue
import threading
from config import API_PORT

API_URL = f"http://127.0.0.1:{API_PORT}"

st.set_page_config(page_title="Face Verification System", layout="wide")

# --- Load Names ---
def get_known_names():
    try:
        response = requests.get(f"{API_URL}/politicians", timeout=5)
        return response.json()["politicians"]
    except:
        return []

known_names = get_known_names()

# --- State Management ---
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
    st.session_state.admin_username = ""
    st.session_state.admin_password = ""

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# --- Camera Manager ---
@st.cache_resource
def get_camera_manager():
    class CameraManager:
        def __init__(self):
            self.cap = None
            self.thread = None
            self.run_flag = [False]
            self.frame_queue = queue.Queue()
            self.last_result = None
            self.is_verifying = False

        def capture_frames(self):
            while self.run_flag[0] and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                if self.frame_queue.full(): self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
                time.sleep(0.05)

        def start(self):
            if self.cap and self.cap.isOpened(): return True
            for idx in range(3):
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if self.cap.isOpened(): break
            else: return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.run_flag[0] = True
            self.thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.thread.start()
            return True

        def stop(self):
            self.run_flag[0] = False
            if self.cap: self.cap.release()
            self.cap = None
            self.thread = None
            self.last_result = None

    return CameraManager()

cam_manager = get_camera_manager()

# --- Sidebar Menu ---
st.sidebar.title("Face AI System")
menu = st.sidebar.radio("Main Menu", ["🔍 Verification", "⚙️ Admin Panel"])

st.sidebar.divider()
st.sidebar.subheader("📋 Verifiable Persons")

# Manual Refresh Button
if st.sidebar.button("🔄 Refresh List"):
    st.rerun()

if known_names:
    for name in sorted(known_names):
        st.sidebar.write(f"• {name}")
else:
    st.sidebar.info("No persons in database")
st.sidebar.divider()

if menu == "🔍 Verification":
    st.header("Face Verification")
    col_cam, col_up = st.columns([2, 1])
    
    with col_cam:
        st.subheader("Live Camera")
        if not st.session_state.camera_running:
            if st.button("▶️ Start Camera", type="primary"):
                if cam_manager.start():
                    st.session_state.camera_running = True
                    st.rerun()
        else:
            if st.button("🛑 Stop Camera"):
                st.session_state.camera_running = False
                cam_manager.stop()
                st.rerun()

        frame_placeholder = st.empty()
        match_placeholder = st.empty()

        if st.session_state.camera_running:
            def verify_bg(frame_data, cam_mgr):
                try:
                    files = {"file": ("frame.jpg", frame_data, "image/jpeg")}
                    resp = requests.post(f"{API_URL}/verify-image", files=files, timeout=10)
                    if resp.status_code == 200:
                        cam_mgr.last_result = resp.json()
                    else:
                        # Clear old result if no face is detected
                        cam_mgr.last_result = {"matched": False, "error": "No face detected"}
                except:
                    # Clear on connection error too
                    cam_mgr.last_result = {"matched": False, "error": "Connection error"}
                cam_mgr.is_verifying = False

            frame_count = 0
            while st.session_state.camera_running:
                if not cam_manager.frame_queue.empty():
                    frame = cam_manager.frame_queue.get()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                    if cam_manager.last_result:
                        res = cam_manager.last_result
                        if res.get("matched"):
                            match_placeholder.success(f"**MATCHED: {res['name']}**\n\nParty: {res['party']}")
                        elif res.get("error"):
                            match_placeholder.info("Looking for a face...")
                        else:
                            match_placeholder.warning("Scanning... No match in database.")

                    frame_count += 1
                    if frame_count % 15 == 0 and not cam_manager.is_verifying:
                        cam_manager.is_verifying = True
                        _, buffer = cv2.imencode(".jpg", frame)
                        threading.Thread(target=verify_bg, args=(buffer.tobytes(), cam_manager), daemon=True).start()
                time.sleep(0.01)

    with col_up:
        st.subheader("Photo Upload")
        up_file = st.file_uploader("Verify image", type=["jpg", "png"])
        if up_file:
            with st.spinner("Analyzing..."):
                res = requests.post(f"{API_URL}/verify-image", files={"file": up_file}, timeout=30).json()
                if res.get("matched"):
                    st.success(f"Matched: {res['name']}")
                    st.info(f"Party: {res['party']}\n\n{res['description']}")
                else:
                    st.warning("No match found")

elif menu == "⚙️ Admin Panel":
    # Auto-stop camera when entering Admin
    if st.session_state.camera_running:
        st.session_state.camera_running = False
        cam_manager.stop()
        st.rerun()

    st.header("Admin Management")
    if not st.session_state.admin_logged_in:
        with st.form("login"):
            u = st.text_input("Admin Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if u == "admin" and p == "secret123":
                    st.session_state.admin_username, st.session_state.admin_password = u, p
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    else:
        st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"admin_logged_in": False}))
        
        # Admin Forms
        with st.expander("➕ Add Person", expanded=True):
            with st.form("add"):
                n = st.text_input("Name")
                d = st.text_area("Description")
                py = st.text_input("Party")
                imgs = st.file_uploader("Images", type=["jpg", "png"], accept_multiple_files=True)
                if st.form_submit_button("Save"):
                    if not n or not imgs:
                        st.error("⚠️ Name and at least one Image are required!")
                    else:
                        files = [("images", i) for i in imgs]
                        r = requests.post(f"{API_URL}/add-politician", data={"name":n,"description":d,"party":py}, files=files, auth=(st.session_state.admin_username, st.session_state.admin_password))
                        if r.status_code == 200: st.success("Added!"); time.sleep(1); st.rerun()
                        else: st.error("Error: Could not add person. Check if the image has a clear face.")

        with st.expander("📝 Edit Person"):
            target = st.selectbox("Select Person", known_names)
            with st.form("edit"):
                new_n = st.text_input("New Name", value=target)
                new_d = st.text_area("New Description")
                new_p = st.text_input("New Party")
                if st.form_submit_button("Update"):
                    r = requests.post(f"{API_URL}/edit-politician", data={"old_name":target, "new_name":new_n, "new_description":new_d, "new_party":new_p}, auth=(st.session_state.admin_username, st.session_state.admin_password))
                    if r.status_code == 200: st.success("Updated!"); time.sleep(1); st.rerun()

        with st.expander("🗑️ Delete Person"):
            with st.form("delete_form"):
                del_n = st.selectbox("Select Person to Delete", known_names)
                if st.form_submit_button("Delete Permanently", type="primary"):
                    r = requests.post(f"{API_URL}/delete-politician", data={"name":del_n}, auth=(st.session_state.admin_username, st.session_state.admin_password))
                    if r.status_code == 200:
                        st.success("Deleted Successfully!")
                        # Force update the list in memory immediately
                        st.session_state.known_names = get_known_names()
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Error: Could not delete person.")