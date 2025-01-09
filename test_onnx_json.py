from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import json

# Load configuration from JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Extract model and parameters from the configuration
model_path = config["model"]
parameters = config["parameters"]

# Initialize YOLO model
model = YOLO(model_path)

# Flask app setup
app = Flask(__name__)

# Video processing function
def generate_frames():
    # Load the video source from the JSON file
    cap = cv2.VideoCapture(parameters["source"])
    
    if not cap.isOpened():
        print("Error: Unable to access video feed.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO prediction with parameters from JSON
        results = model.predict(
            source=frame,
            conf=parameters["conf"],
            iou=parameters["iou"],
            save_conf=parameters["save_conf"],
            imgsz=(480, 640),  # Optional fixed resolution
            classes=parameters["classes"]  # Optional class filtering
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()  # Visualization of YOLO results
        
        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        # Yield the encoded frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')  # HTML for the video stream

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
