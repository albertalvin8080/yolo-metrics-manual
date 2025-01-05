from some_utils import MODEL, YOLOV10, YOLOV11

if MODEL == YOLOV10:
    from ultralytics import YOLOv10 as YOLO
elif MODEL == YOLOV11:
    from ultralytics import YOLO
import some_utils
import cv2 as cv
import datetime
import os

from collections import defaultdict

def calculate_tp_fp_fn(predictions, ground_truths, iou_threshold=0.5):
    matched_gt = set()  # Track matched ground truth indices
    tp = 0
    fp = 0
    iou_matrix = defaultdict(list)  # Store IoUs for debugging

    for pred in predictions:
        ious = [some_utils.compute_iou(pred[:4], gt) for gt in ground_truths]
        max_iou = max(ious) if ious else 0
        max_index = ious.index(max_iou) if ious else -1

        iou_matrix[tuple(pred)] = ious  # Debugging information

        if max_iou >= iou_threshold and max_index not in matched_gt:
            # True Positive: Prediction matches a ground truth box
            tp += 1
            matched_gt.add(max_index)
        else:
            # False Positive: Prediction does not match any ground truth box
            fp += 1

    # False Negatives: Ground truth boxes not matched to any prediction
    fn = len(ground_truths) - len(matched_gt)

    print(f"IoU Matrix (Predictions vs GT): {iou_matrix}")  # Debugging
    return tp, fn, fp

def calculate(img, predicted_boxes, ground_truth_boxes, iou_threshold):
    tp = fp = 0
    fn = len(ground_truth_boxes)
    
    for pred_box in predicted_boxes:
        cv.rectangle(img,
                (int(pred_box[0]), int(pred_box[1])),
                (int(pred_box[2]), int(pred_box[3])),
                (0, 255, 0),
                4)
        match_found = False
        for gt_box in ground_truth_boxes: # O(log n) ?
            iou = some_utils.compute_iou(pred_box, gt_box)
            if iou < iou_threshold:
                continue
            tp += 1
            fn -= 1
            match_found = True
            ground_truth_boxes.remove(gt_box)  # Avoid multiple matches to the same ground truth
            break
        if not match_found:
            fp += 1
            
    return tp, fn, fp


def evaluate(model_weights, confidence=0.2, should_save=False, verbose=False):
    val_path = "../YOLOv10-structure/images/val/"
    train_path = "../YOLOv10-structure/images/train/"
    val_imgs = map(lambda x: val_path + x, os.listdir(val_path))
    train_imgs = map(lambda x: train_path + x, os.listdir(train_path))
    
    evaluation = f"evaluation-{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(evaluation, exist_ok=False)

    model = YOLO(model_weights)

    iou_threshold = 0.5
    # Use this = 1.0 for disabling filter.
    # iou_discard_threshold = 1.0 
    # iou_discard_threshold = 0.2 
    iou_discard_threshold = 0.7
        
    total_ground_truth = 0
    total_detections = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # for img_path in filter(lambda x: "leandroplaca96-s-012-13" in x, list(val_imgs)):
    # for img_path in filter(lambda x: "leandroplaca52-LeandroPlaca52-25.png" in x, list(train_imgs)):
    for img_path in list(val_imgs):
    # for img_path in list(val_imgs) + list(train_imgs):
        
        label_path = img_path.replace("images", "labels")[:-3] + "txt"
        img = cv.imread(img_path)
        
        ground_truth_boxes = []
        with open(label_path, "r") as label_file:
            for line in label_file:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                img_h, img_w, _ = img.shape
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)
                ground_truth_boxes.append([x1, y1, x2, y2])
                cv.rectangle(img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        4)

        results = model(
            img, 
            conf=confidence, 
            verbose=verbose,
            iou=iou_discard_threshold,
        )[0]
        # Sort by confidence
        results = sorted(results.boxes.data.tolist(), key=lambda x: x[4], reverse=True)  
        
        raw_predicted_boxes = []
        # for box in results.boxes.data.tolist():
        for box in results:
            x1, y1, x2, y2, conf, cls = box[:6]
            raw_predicted_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        filtered_boxes = []
        for box in raw_predicted_boxes:
            keep = True
            for fb in filtered_boxes:
                iou = some_utils.compute_iou(box[:4], fb[:4])
                if iou > iou_discard_threshold:  # Overlapping, discard
                    keep = False
                    break
            if keep:
                filtered_boxes.append(box)
                
        predicted_boxes = filtered_boxes
        
        # print("-"*30)
        # print(raw_predicted_boxes)
        # print(predicted_boxes)
        # print("-"*30)

        current_detections = len(predicted_boxes)
        total_detections += current_detections
        ground_truth_boxes_len = len(ground_truth_boxes)
        total_ground_truth += ground_truth_boxes_len
        
        # tp, fn, fp = calculate(img, predicted_boxes, ground_truth_boxes, iou_threshold)
        tp, fn, fp = calculate_tp_fp_fn(predicted_boxes, ground_truth_boxes, iou_threshold)

        total_tp += tp
        total_fn += fn
        total_fp += fp
        # print(tp, fn, fp)
        
        if not should_save:
            continue
        individual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        individual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Shadow effect: Draw text with a thicker, dark border
        cv.putText(img,
                f"Detections: {current_detections}, GroundTruth: {ground_truth_boxes_len}, Precision: {individual_precision:.5f}, Recall: {individual_recall:.5f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,  # fontScale
                (0, 0, 0),  # border color (black)
                4,  # border thickness
                cv.LINE_AA,
                )
        cv.putText(img,
                f"Detections: {current_detections}, GroundTruth: {ground_truth_boxes_len}, Precision: {individual_precision:.5f}, Recall: {individual_recall:.5f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,  # fontScale
                (0, 255, 0),  # color
                2,  # thickness
                )
        new_img_path = evaluation + img_path[img_path.rfind("/") + 1:-4] + ".png"
        cv.imwrite(new_img_path, img)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    # accuracy = total_fp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    lines = [
        f"weights: {model_weights}\n",
        f"Confidence: {confidence}\n",
        f"IoU Threshold: {iou_threshold}\n",
        f"IoU Discard Threshold: {iou_discard_threshold}\n",
        f"TP: {total_tp}\n",
        f"FP: {total_fp}\n",
        f"FN: {total_fn}\n",
        f"Ground Truth: {total_ground_truth}\n",
        f"N Detections: {total_detections}\n",
        f"Precision: {precision:.5f}\n",
        f"Recall: {recall:.5f}\n",
        # f"Accuracy: {accuracy : .5f}\n",
    ]
    with open(os.path.join(evaluation, "evaluation.txt"), "w") as f:
        f.writelines(lines)
        
    print(lines)

if __name__ == "__main__":
    if MODEL == YOLOV10:
        # evaluate("weights/train-yolov10-s-base-300-earlystop/best.pt")
        # evaluate("weights/train-yolov10-s-base-300-earlystop/last.pt")
        evaluate("weights/runs-yolov10-s-jameslahn-500-full/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-500-full/last.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-sgd/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-sgd/last.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-adamw/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-adamw/last.pt")
    elif MODEL == YOLOV11:
        evaluate("weights/runs-yolov11-s-base-1000-full-sgd/best.pt")
        # evaluate("weights/runs-yolov11-s-base-1000-full-sgd/last.pt")
    