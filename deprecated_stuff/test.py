from ultralytics import YOLOv10 as YOLO
import cv2 as cv
import some_utils

model = YOLO("weights\\last.pt")

path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\val\\s-05-31-aug-h-v.png"

img = cv.imread(path)

# results = model(img, conf=0.2, iou=0.5, nms=True)[0] # nms is not working
results = model(img, conf=0.2)[0]

label_path = path.replace("images", "labels")[:-3] + "txt"

with open(label_path, 'r') as label_file:
    num_objs = len(label_file.readlines())

filtered_boxes = []
iou_threshold = 0.3

for box in results.boxes.data.tolist():
    keep = True
    for fb in filtered_boxes:
        iou = some_utils.compute_iou(box[:4], fb[:4])
        if iou > iou_threshold:  # Overlapping, discard
            keep = False
            break
    if keep:
        filtered_boxes.append(box)

n_detections = 0
for box in filtered_boxes:
    x1, y1, x2, y2, conf, cls = box[:6]
    label = f"{model.names[cls]} {conf:.2f}"

    # if conf > 0.2:
    print((int(x1), int(y1)), (int(x2), int(y2)))
    cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    n_detections += 1

cv.putText(img,
           f"detections: {n_detections}, real quant: {num_objs}",
           (10, 30),
           cv.FONT_HERSHEY_SIMPLEX,
           0.8,  # fontScale
           (0, 255, 0),  # color
           2,  # thickness
           )

cv.imshow("img", img)
cv.waitKey()
cv.destroyAllWindows()
