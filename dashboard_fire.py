import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

st.title("Fire Event Detection")

# Sidebar camera selection
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.number_input(
    "Select Camera Index",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    help="0 = Default webcam, 1 = External camera, etc."
)
st.sidebar.info("💡 Tip: Try 0 for built-in webcam, 1 for external USB camera")

# Load model
model = YOLO("Fire_Event_best.pt")

# Custom CSS for flashy alert
st.markdown("""
    <style>
    .flashy-alert {
        font-size: 28px !important;
        font-weight: bold;
        color: white !important;
        background: linear-gradient(90deg, #ff0000, #ff8000, #ffff00, #ff8000, #ff0000);
        background-size: 300% 300%;
        animation: flashy 1.5s infinite alternate;
        padding: 15px !important;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        margin: 10px 0;
    }
    @keyframes flashy {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

# Placeholders
frame_placeholder = st.empty()
message_placeholder = st.empty()
status_placeholder = st.empty()

# Initialize camera
cap = None
try:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        status_placeholder.error(f"❌ Failed to open camera {camera_index}. Try a different index!")
        st.stop()
    else:
        status_placeholder.success(f"✅ Using camera {camera_index}")
except Exception as e:
    status_placeholder.error(f"❌ Camera error: {str(e)}")
    st.stop()

# Store last detection state
last_detection = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.warning("⚠️ Camera disconnected or frame read failed")
            break

        # Run detection
        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Process detections
        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        current_detection = ""

        if names:
            unique_names = list(set(names))
            current_detection = ', '.join(unique_names)
            alert_text = f"🚨 DETECTED: {current_detection} 🚨"
            
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>', 
                unsafe_allow_html=True
            )

            # Update detection state
            last_detection = current_detection
        else:
            message_placeholder.empty()
            last_detection = ""

        # Small delay to reduce CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    if cap is not None:
        cap.release()
    status_placeholder.info("⏹️ Camera released")