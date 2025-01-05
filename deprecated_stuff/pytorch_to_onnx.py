from ultralytics import YOLOv10 as YOLO

# NOTE: Use google colab
model = YOLO("best.pt")

model.export(format="onnx")