import cv2 as cv
import datetime
import os
import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = "weights\\runs-yolov10-s-300-earlystop\\best.onnx"
session = ort.InferenceSession(onnx_model_path)

# Set up paths
val_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\val\\"
train_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\train\\"
val_imgs = map(lambda x: val_path + x, os.listdir(val_path))
train_imgs = map(lambda x: train_path + x, os.listdir(train_path))

evaluation = f"evaluation-{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}\\"
os.makedirs(evaluation, exist_ok=True)

# Function for preprocessing image for ONNX inference
def preprocess_image(image):
    # Resize image to YOLO input size (typically 640x640)
    image_resized = cv.resize(image, (640, 640))
    # Normalize the image (if necessary)
    image_normalized = image_resized.astype(np.float32) / 255.0
    # Change the image to (batch_size, channels, height, width) format
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image_transposed, axis=0)  # Add batch dimension

# Inference and saving annotated images
for img_path in list(val_imgs):
    img = cv.imread(img_path)
    
    # Preprocess image
    input_tensor = preprocess_image(img)
    
    # Run inference
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    
    # Extract results (bounding boxes, confidences, and class labels)
    # This assumes the model outputs in YOLO format (you may need to adjust depending on model)
    boxes = outputs[0]  # For YOLOv10, the first output might be boxes
    confidences = outputs[1]  # The second output could be confidences (probabilities)
    class_ids = outputs[2]  # The third output could be class indices
    
    # Draw annotations on the image
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if confidence > 0.2:  # Threshold confidence
            x1, y1, x2, y2 = box
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.putText(img, f"Class: {class_id}, Conf: {confidence:.2f}", 
                       (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Generate a filename for saving
    filename = os.path.join(evaluation, os.path.basename(img_path[img_path.rfind("\\") + 1:-4] + ".png"))
    
    # Save the annotated image
    cv.imwrite(filename, img)
