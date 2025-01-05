from ultralytics import YOLOv10 as YOLO
import some_utils
import cv2 as cv
import datetime
import os

model = YOLO("weights\\runs-yolov10-s-300-earlystop\\best.pt")

val_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\val\\"
train_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\train\\"
val_imgs = map(lambda x: val_path + x, os.listdir(val_path))
train_imgs = map(lambda x: train_path + x, os.listdir(train_path))

evaluation = f"evaluation-{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}\\"
os.makedirs(evaluation, exist_ok=False)

model = YOLO("weights\\runs-yolov10-s-300-earlystop\\best.pt")

for img_path in list(val_imgs):
    img = cv.imread(img_path)
    
    # Predict using the YOLO model
    results = model.predict(img, conf=0.2)
    
    # Annotate the image with predictions
    annotated_img = results[0].plot()  # Get the annotated image
    
    # Generate a filename for saving
    filename = os.path.join(evaluation, os.path.basename(img_path[img_path.rfind("\\") + 1:-4] + ".png"))
    
    # Save the annotated image
    cv.imwrite(filename, annotated_img)