import argparse
from ultralytics import YOLO

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="YOLO model for object detection with various customization options.")
    
    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file (e.g., yolov8n.pt).")
    parser.add_argument("--source", type=str, default='ultralytics/assets', help="Data source for inference.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--imgsz", type=lambda s: tuple(map(int, s.split(','))) if ',' in s else int(s), default=640, help="Image size for inference (int or height,width).")
    parser.add_argument("--half", action="store_true", help="Enable half-precision (FP16) inference.")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum number of detections per image.")
    parser.add_argument("--vid_stride", type=int, default=1, help="Frame stride for video inputs.")
    parser.add_argument("--stream_buffer", action="store_true", help="Queue incoming frames for video streams.")
    parser.add_argument("--visualize", action="store_true", help="Visualize model features during inference.")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--agnostic_nms", action="store_true", help="Enable class-agnostic Non-Maximum Suppression.")
    parser.add_argument("--classes", type=lambda s: list(map(int, s.split(','))), default=None, help="Filter predictions to specific class IDs.")
    parser.add_argument("--retina_masks", action="store_true", help="Return high-resolution segmentation masks.")
    parser.add_argument("--embed", type=lambda s: list(map(int, s.split(','))), default=None, help="Extract feature vectors from specific layers.")
    parser.add_argument("--project", type=str, default=None, help="Name of the project directory for outputs.")
    parser.add_argument("--name", type=str, default=None, help="Name of the prediction run.")
    
    # Visualization arguments
    parser.add_argument("--show", action="store_true", help="Display annotated images or videos.")
    parser.add_argument("--save", action="store_true", help="Save annotated images or videos to file.")
    parser.add_argument("--save_frames", action="store_true", help="Save individual video frames as images.")
    parser.add_argument("--save_txt", action="store_true", help="Save detection results as a text file.")
    parser.add_argument("--save_conf", action="store_true", help="Include confidence scores in saved text files.")
    parser.add_argument("--save_crop", action="store_true", help="Save cropped images of detections.")
    parser.add_argument("--show_labels", action="store_true", default=True, help="Display labels for detections.")
    parser.add_argument("--show_conf", action="store_true", default=True, help="Display confidence scores for detections.")
    parser.add_argument("--show_boxes", action="store_true", default=True, help="Draw bounding boxes on detections.")
    parser.add_argument("--line_width", type=int, default=None, help="Line width for bounding boxes.")

    # Parse arguments
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Run prediction
    print(f"Running prediction on source: {args.source}...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        batch=args.batch,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        stream_buffer=args.stream_buffer,
        visualize=args.visualize,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        retina_masks=args.retina_masks,
        embed=args.embed,
        project=args.project,
        name=args.name,
        show=args.show,
        save=args.save,
        save_frames=args.save_frames,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        show_boxes=args.show_boxes,
        line_width=args.line_width
    )
    
    print("Inference complete.")

if __name__ == "__main__":
    main()
