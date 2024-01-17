from ultralytics import YOLO
model = YOLO('../models/yolov8n-face.pt')
model.export(format='onnx', imgsz=640, opset=12, dynamic=True) # dynamic
# model.export(format='onnx', imgsz=640, opset=12, dynamic=False) # static