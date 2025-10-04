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

st.subheader("Add a New Person via GUI")
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
            response = requests.post(f"{API_URL}/add-politician", data=data, files=files, timeout=10)
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(response.json()["detail"])
        except Exception as e:
            st.error(f"Error adding person: {e}")

st.subheader("Edit an Existing Person via GUI")
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
            response = requests.post(f"{API_URL}/edit-politician", data=data, files=files, timeout=10)
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(response.json()["detail"])
        except Exception as e:
            st.error(f"Error editing person: {e}")

st.subheader("Delete a Person via GUI")
with st.form(key="delete_person_form"):
    delete_name = st.selectbox("Select Person to Delete", known_names)
    submit_delete = st.form_submit_button(label="Delete Person")

    if submit_delete and delete_name:
        try:
            data = {"name": delete_name}
            response = requests.post(f"{API_URL}/delete-politician", data=data, timeout=10)
            if response.status_code == 200:
                st.success(response.json()["message"])
                response = requests.get(f"{API_URL}/politicians", timeout=10)
                st.session_state.known_names = response.json()["politicians"]
            else:
                st.error(response.json()["detail"])
        except Exception as e:
            st.error(f"Error deleting person: {e}")

cap = None
camera_running = False
frame_queue = queue.Queue()
capture_thread = None
frame_placeholder = st.empty()
match_placeholder = st.empty()

def capture_frames():
    global cap, camera_running
    while camera_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_queue.put(frame)
        time.sleep(0.2)

def start_camera():
    global cap, camera_running, capture_thread
    if camera_running:
        return

    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            st.success(f"Camera opened successfully at index {idx}")
            break
    else:
        st.error("Could not open camera. Check if another app is using it or enable camera permissions in Windows.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    camera_running = True

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    frame_count = 0
    while camera_running:
        if not frame_queue.empty():
            frame = frame_queue.get()
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
                        match_placeholder.text(f"Matched: {result['name']} (Distance: {result['distance']:.2f})")
                        st.write(f"Description: {result['description']}")
                        st.write(f"Party: {result['party']}")
                    else:
                        match_placeholder.text("No match found.")
                except requests.exceptions.ReadTimeout:
                    match_placeholder.text("Server timed out. Please wait or check server status.")
                except Exception as e:
                    match_placeholder.text(f"Error sending frame: {e}")
        time.sleep(0.01)

def stop_camera():
    global camera_running, cap, capture_thread
    camera_running = False
    if cap is not None and cap.isOpened():
        cap.release()
    if capture_thread is not None:
        capture_thread.join(timeout=1.0)
    st.success("Camera stopped.")

st.subheader("Camera Controls")
if st.button("Start Camera"):
    start_camera()

if camera_running:
    if st.button("Stop Camera"):
        stop_camera()