from ultralytics import YOLOWorld
from ultralytics . engine . results import Boxes
def save_detection_results( results : Boxes):
    for i, result in enumerate( results):
        result.save( f"run/ detection - {i}.jpg")
model = YOLOWorld(" yolov8s - world .pt")
model.set_classes (["bus"]) 
results : Boxes = model.predict("samples/bus.jpg")
save_detection_results( results )