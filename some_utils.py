import numpy as np


def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def yolov11_decode(
    output, image_shape, input_dim=640, conf_threshold=0.2, onnx=True
):
    boxes = []
    h, w = image_shape

    # (1, 5, 8400) to (1, 8400, 5)
    output = np.transpose(output[0], (0, 2, 1))
    print(np.shape(output))
    # print(output)

    for box in output[0]:  # Loop over all 8400 grid cells/anchors
        x, y, width, height, conf = box[:5]
        if conf > conf_threshold:
            # Calculate top-left and bottom-right coordinates
            if onnx:
                x1 = int((x - width / 2) * w / input_dim)
                y1 = int((y - height / 2) * h / input_dim)
                x2 = int((x + width / 2) * w / input_dim)
                y2 = int((y + height / 2) * h / input_dim)
            else:
                x1 = int((x - width / 2))
                y1 = int((y - height / 2))
                x2 = int((x + width / 2))
                y2 = int((y + height / 2))

            boxes.append([x1, y1, x2, y2, conf, 0])

    return boxes


def yolov10_decode(
    output, image_shape, input_dim=640, conf_threshold=0.2, onnx=True
):
    boxes = []
    h, w = image_shape
    results = output[0][0] if onnx else output[0].boxes.data.tolist()
    for box in results:
        x1, y1, x2, y2, conf, cls = box
        if conf > conf_threshold:
            # Scale the coordinates back to the original image size
            if onnx:
                x1 = int(x1 * w / input_dim)
                y1 = int(y1 * h / input_dim)
                x2 = int(x2 * w / input_dim)
                y2 = int(y2 * h / input_dim)
            boxes.append([x1, y1, x2, y2, conf, cls])
    return boxes


def non_max_suppression(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    # Extract coordinates and confidence scores
    boxes = np.array(boxes)
    x, y, w, h, conf = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = w * h
    indices = np.argsort(conf)[::-1]  # Sort by confidence score (descending)

    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        # Compute IoU between the top box and all remaining boxes
        xx1 = np.maximum(x[current], x[indices[1:]])
        yy1 = np.maximum(y[current], y[indices[1:]])
        xx2 = np.minimum(x[current] + w[current], x[indices[1:]] + w[indices[1:]])
        yy2 = np.minimum(y[current] + h[current], y[indices[1:]] + h[indices[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[indices[1:]] - intersection
        iou = intersection / union

        # Keep only boxes with IoU below the threshold
        indices = indices[np.where(iou < iou_threshold)[0] + 1]

    return boxes[keep].tolist()
