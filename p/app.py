# Import necessary libraries
from flask import Flask, render_template, Response  # Flask web framework components
# render_template is used for sending the HTML file "index.html" to the web
# Response is used to stream video frames to the browser as a continuous stream (not just a single static image)
import cv2  # OpenCV for video capture and image processing
import numpy as np  # NumPy for numerical operations
from collections import deque  # For fixed-length queue to store predictions
import tensorflow as tf  # TensorFlow for deep learning (backend)
from tensorflow import keras  # Keras high-level API
from ultralytics import YOLO  # YOLO object detection from the Ultralytics package
import threading  # For running beep alert in parallel
import time  # For sleep in beeping loop
import winsound  # Windows-only library to play beeps

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model (trained to detect open/closed eyes), adjust path if needed
yolo_model = YOLO("best.pt")

# Load the final trained LSTM drowsiness detection model
final_model = keras.models.load_model("drowsiness_lstm_model.keras")

# Constants
SEQ_LENGTH = 30  # Number of eye states to consider for LSTM input (30 FPS is exactly 1 second)
# 30 is a good trade-off between speed and accuracy
rolling_predictions = deque(maxlen=SEQ_LENGTH)  # Rolling window of eye predictions
# It stores the most recent 30 eye state predictions. This is passed into the LSTM
# double-ended queue has faster appends/pops than regular queue. deque's maxlen=30, meaning it auto discards old eye states

# Beeping state flag (controls alert sound)
beeping = False

# Background thread function to play a beep sound repeatedly while drowsy
def sound_alert():
    while beeping:
        winsound.Beep(1000, 500)  # Play 1000 Hz tone for 500 ms
        time.sleep(0.1)  # Short delay between beeps

# Function to predict eye state from a video frame
def get_eye_state(frame):
    results = yolo_model(frame)  # Run YOLO model on the frame
    detected_classes = []  # Store detected classes (e.g., 0=open, 1=closed)

    for result in results:
        if result.boxes is not None:
            detected_classes.extend(result.boxes.cls.tolist())  # Add detected class IDs

    # If no eyes detected (no class 0 or 1), return None
    if not any(cls in [0, 1] for cls in detected_classes):
        return None

    # If closed eye class (1) detected, return 1; otherwise return 0 (open)
    return 1 if 1 in detected_classes else 0

# Generator function to transfer video frames to Flask web stream
def generate_frames():
    global beeping  # Access the global beeping flag
    cap = cv2.VideoCapture(0)  # Open webcam (device 0 is the default camera - laptop's)

    while True:
        success, frame = cap.read()  # Read frame from webcam
        if not success:
            break  # Exit loop if reading fails

        resized_frame = cv2.resize(frame, (640, 480))  # Resize frame - less computation
        eye_state = get_eye_state(resized_frame)  # Get eye state (0=open, 1=closed, or None)

        if eye_state is not None:
            rolling_predictions.append(eye_state)  # Add eye state to rolling sequence

            if len(rolling_predictions) == SEQ_LENGTH:
                # Prepare input for LSTM model (shape: 1 x 30 x 1)
                input_seq = np.array(rolling_predictions).reshape(1, SEQ_LENGTH, 1)
                # # Reshapes into 3D tensor for LSTM input: (batch_size, time_steps, eye state per frame)
                lstm_pred = final_model.predict(input_seq, verbose=0)[0, -1, 0]  # Get latest prediction
                # verbose=0 shows the line of code in vs
                # [0, -1, 0]: 0 = first and only batch, -1 = last time step, 0 = eye state of the last time step

                if lstm_pred > 0.5:
                    # If predicted drowsy
                    if not beeping:
                        beeping = True
                        threading.Thread(target=sound_alert, daemon=True).start()  # Start beeping thread
# daemon=True makes sure the thread will run in the background and automatically close when the main program exits 

                    # Draw alert text on screen
                    cv2.putText(resized_frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw red border to indicate alert
                    cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1]-1, resized_frame.shape[0]-1),
                                  (0, 0, 255), 10)
                else:
                    # Not drowsy — stop alert
                    beeping = False

                    # Draw green border
                    cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1]-1, resized_frame.shape[0]-1),
                                  (0, 255, 0), 10)
            else:
                # Not enough data yet for prediction
                beeping = False
                cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1]-1, resized_frame.shape[0]-1),
                              (0, 255, 0), 10)  # Green border
        else:
            # No eye detection
            beeping = False
            cv2.putText(resized_frame, "No eyes detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1]-1, resized_frame.shape[0]-1),
                          (128, 128, 128), 10)  # Gray border for unknown state

        # Encode frame as JPEG to send to browser, instead of a NumPy array
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_bytes = buffer.tobytes()

        # Send frame to Flask route
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') # Browser gets a continuous stream of images, which looks like video

# Flask route for main page
@app.route('/')
def index():
    return render_template('index.html')  # HTML page is index.html

# Flask route to serve video stream to HTML
@app.route('/video_feed') # to show the live camera feed in the HTML
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # lets the browser update images frame by frame, simulating a video feed

# Start Flask app on local server
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)  # Accessible at http://127.0.0.1:5000/
    # port=5000 is the location in the server
    # debug=True automatically restarts the server when code changes