# object_detection.py
import torch

# Load custom-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='local')

def detect_objects(image_path):
    results = model(image_path)
    detections = []
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        detections.append({
            "object": row['name'],
            "confidence": float(row['confidence']),
            "bbox": [float(row['xmin']), float(row['ymin']),
                     float(row['xmax']), float(row['ymax'])]
        })
    return detections

# Test the module
frame_img = "dataset/objects/cctv_bag_scene.jpg"
print(detect_objects(frame_img))
