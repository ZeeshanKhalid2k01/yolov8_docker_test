# from ultralytics import YOLO

# # Configure the tracking parameters and run the tracker
# model = YOLO("yolov8n.pt")
# results = model.predict(source=0, conf=0.3, iou=0.5, show=True, save_conf=False)


import json
from ultralytics import YOLO

# Load parameters from the JSON file
with open("config.json", "r") as file:
    config = json.load(file)

# Load the model specified in the JSON
model = YOLO(config["model"])

# Extract parameters dynamically from JSON
params = config["parameters"]

# Run the prediction using parameters from JSON
results = model.predict(
    source=params["source"],
    conf=params["conf"],
    iou=params["iou"],
    show=params["show"],
    save=params["save"],
    save_frames=params["save_frames"],
    save_txt=params["save_txt"],
    save_conf=params["save_conf"],
    save_crop=params["save_crop"],
    show_labels=params["show_labels"],
    show_conf=params["show_conf"],
    show_boxes=params["show_boxes"],
    line_width=params["line_width"]
)
