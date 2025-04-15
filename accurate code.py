import cv2
import face_recognition
import os
import zipfile
import threading
import time
from flask import Flask, render_template_string

# Global Variables
ignition_status = "OFF"
license_encoding = None
capture_mode = "license"
frame_skip = 3  # Process every 3rd frame
last_seen_time = time.time()
last_mismatch_time = None
face_detected = False
match_found = False
countdown = None  # Countdown for ignition off

# Path to ZIP file
ZIP_FILE_PATH = r"C:/Users/sampa/Downloads/face_recognition-master.zip"
MODEL_DIR = "face_models"
EXTRACTED_DIR = os.path.join(MODEL_DIR, "extracted")

# Flask Web App
app = Flask(__name__)  # Fixed __name__

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ignition Monitor</title>
    <meta http-equiv="refresh" content="1">  <!-- Auto-refresh every 1 second -->
</head>
<body>
    <h1>Ignition Status: {{ status }}</h1>
    {% if message %}
    <p style="color:red; font-size:18px;">{{ message }}</p>
    {% endif %}
    <p>Last Updated: {{ update_time }}</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
                                  status=ignition_status,
                                  message=status_message(),
                                  update_time=time.strftime("%Y-%m-%d %H:%M:%S"))

def run_flask():
    """Run Flask Web Server"""
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def extract_models():
    """Extracts the model ZIP file from the given path."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting model files...")
        with zipfile.ZipFile(ZIP_FILE_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        print("Models extracted successfully.")
    else:
        print("Models already extracted.")

def status_message():
    """Returns the warning message only if ignition is ON."""
    global face_detected, match_found, last_seen_time, last_mismatch_time, countdown
    current_time = time.time()

    if ignition_status == "OFF":
        return ""  # No message if ignition is OFF

    if countdown is not None and countdown > 0:
        return f"IGNITION OFF in {countdown} seconds"

    if not face_detected and current_time - last_seen_time > 1:
        countdown = 10
        return f"PERSON OUT OF FRAME - Ignition in {countdown} seconds"
    if not match_found and last_mismatch_time and current_time - last_mismatch_time > 1:
        countdown = 10
        return f"NOT MATCH - Ignition in {countdown} seconds"

    return ""  # No issue, no message

def camera_processing():
    """Capture license image and verify face using extracted models."""
    global ignition_status, license_encoding, capture_mode, last_seen_time, last_mismatch_time, face_detected, match_found, countdown
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Camera initialized. Press 'c' to capture license, 'q' to exit.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_disp = frame.copy()
        current_time = time.time()

        if capture_mode == "license":
            cv2.putText(frame_disp, "Press 'c' to capture license image", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame, model="hog")

            if faces:
                encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=faces)
                match_found = False

                for (face_loc, face_enc) in zip(faces, encodings):
                    match = face_recognition.compare_faces([license_encoding], face_enc, tolerance=0.45)[0]
                    (top, right, bottom, left) = face_loc

                    if match:
                        match_found = True
                        cv2.rectangle(frame_disp, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame_disp, "Match - Ignition ON", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        ignition_status = "ON"
                        last_seen_time = current_time  # Update last seen time
                        face_detected = True
                        last_mismatch_time = None  # Reset mismatch timer
                        countdown = None  # Reset countdown
                    else:
                        cv2.rectangle(frame_disp, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame_disp, "No Match", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        if last_mismatch_time is None:
                            last_mismatch_time = current_time
                        face_detected = True

                if not match_found and current_time - last_mismatch_time > 1 and countdown is None:
                    countdown = 10  # Start countdown

            else:
                if not face_detected and countdown is None:
                    countdown = 10  # Start countdown if no face detected
                face_detected = False  # No face detected

            # Countdown to turn off ignition
            if countdown is not None and ignition_status == "ON":
                if countdown > 0:
                    countdown -= 1  # Reduce countdown every frame cycle
                else:
                    ignition_status = "OFF"
                    countdown = None  # Reset countdown after turning off

        cv2.imshow("Camera", frame_disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if capture_mode == "license" and key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame, model="hog")
            if faces:
                license_encoding = face_recognition.face_encodings(rgb_frame, known_face_locations=faces)[0]
                print("License image captured! Now entering verification mode.")
                capture_mode = "verify"
            else:
                print("No face detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":  # Fixed this line
    extract_models()

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start camera processing
    camera_processing()

    print("Shutting down.")
