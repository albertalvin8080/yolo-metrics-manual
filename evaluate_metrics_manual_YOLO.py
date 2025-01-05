from ultralytics import YOLO
import some_utils
import cv2 as cv
import datetime
import os


def evaluate(
    model_weights,
    confidence=0.2,
    should_save=False,
    verbose=False,
    model="10",
    onnx=False,
):
    # NOTE: If you're using yolo directly from ultralytics, you must use yolov10_decode.
    decoder = (
        some_utils.yolov11_decode
        if model == "11" and onnx
        else some_utils.yolov10_decode
    )

    val_path = "../YOLOv10-structure/images/val/"
    train_path = "../YOLOv10-structure/images/train/"
    val_imgs = map(lambda x: val_path + x, os.listdir(val_path))
    train_imgs = map(lambda x: train_path + x, os.listdir(train_path))

    evaluation = (
        f"evaluation-{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}/"
    )
    os.makedirs(evaluation, exist_ok=False)

    model = YOLO(model_weights)

    iou_gt_threshold = 0.15
    # Use this = 1.0 for disabling filter.
    # iou_discard_threshold = 1.0
    # iou_discard_threshold = 0.2
    iou_nms_threshold = 0.7

    total_ground_truth = 0
    total_detections = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # for img_path in filter(lambda x: "leandroplaca96-s-012-13" in x, list(val_imgs)):
    # for img_path in filter(lambda x: "leandroplaca96-s-03-22" in x, list(val_imgs)):
    # for img_path in filter(lambda x: "leandroplaca52-LeandroPlaca52-27.png" in x, list(val_imgs)):
    # for img_path in filter(lambda x: "leandroplaca96-s-09-12.png" in x, list(val_imgs)):
    # for img_path in filter(lambda x: "leandroplaca52-LeandroPlaca52-25.png" in x, list(train_imgs)):
    # for img_path in list(val_imgs) + list(train_imgs):
    for img_path in list(val_imgs):

        label_path = img_path.replace("images", "labels")[:-3] + "txt"
        img = cv.imread(img_path)

        ground_truth_boxes = []
        with open(label_path, "r") as label_file:
            for line in label_file:
                cls, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                img_h, img_w, _ = img.shape
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)
                ground_truth_boxes.append([x1, y1, x2, y2])
                cv.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4
                )

        total_ground_truth += len(ground_truth_boxes)

        results = model.predict(
            img,
            conf=confidence,
            verbose=verbose,
            iou=iou_nms_threshold,
            visualize=False,
        )

        detected_boxes = decoder(results, image_shape=(480, 640), onnx=False)
        before_nms = len(detected_boxes)
        # NMS apparently not being necessary.
        detected_boxes = some_utils.non_max_suppression(detected_boxes, iou_threshold=iou_nms_threshold)
        after_nms = len(detected_boxes)
        if after_nms != before_nms:
            print(len(detected_boxes))

        matched_gt = set()
        matched_detections = set()

        tp = 0
        # Compare detected boxes with ground truth boxes
        for i, gt_box in enumerate(ground_truth_boxes):
            for j, det_box in enumerate(detected_boxes):
                iou = some_utils.compute_iou(gt_box, det_box)
                if iou >= iou_gt_threshold and j not in matched_detections:
                    matched_gt.add(i)
                    matched_detections.add(j)
                    tp += 1
                    cv.rectangle(
                        img,
                        (int(det_box[0]), int(det_box[1])),
                        (int(det_box[2]), int(det_box[3])),
                        (0, 255, 0),
                        1,
                    )
                    break

        for i, det_box in enumerate(detected_boxes):
            if i in matched_detections:
                continue
            cv.rectangle(
                img,
                (int(det_box[0]), int(det_box[1])),
                (int(det_box[2]), int(det_box[3])),
                (0, 255, 0),
                1,
            )

        fn = len(ground_truth_boxes) - len(matched_gt)
        fp = len(detected_boxes) - len(matched_detections)
        total_fn += fn
        total_fp += fp
        total_tp += tp

        concerning = False
        if fp > 0 or fn > 0:
            print(f"{img_path}\n-tp: {tp}\n-fp: {fp}\n-fn: {fn}")
            concerning = True

        # Shadow effect: Draw text with a thicker, dark border
        # cv.putText(
        #     img,
        #     f"Detections: {current_detections}, GroundTruth: {ground_truth_boxes_len}, Precision: {individual_precision:.5f}, Recall: {individual_recall:.5f}",
        #     (10, 30),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     0.6,  # fontScale
        #     (0, 0, 0),  # border color (black)
        #     4,  # border thickness
        #     cv.LINE_AA,
        # )
        # cv.putText(
        #     img,
        #     f"Detections: {current_detections}, GroundTruth: {ground_truth_boxes_len}, Precision: {individual_precision:.5f}, Recall: {individual_recall:.5f}",
        #     (10, 30),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     0.6,  # fontScale
        #     (0, 255, 0),  # color
        #     2,  # thickness
        # )
        new_img_path = evaluation + img_path[img_path.rfind("/") + 1 : -4] + ".png"
        if should_save or concerning:
            cv.imwrite(new_img_path, img)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    # accuracy = total_fp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    lines = [
        f"weights: {model_weights}\n",
        f"Confidence: {confidence}\n",
        f"IoU gt Threshold: {iou_gt_threshold}\n",
        f"IoU NMS Threshold: {iou_nms_threshold}\n",
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
    # evaluate("weights_v2/runs-yolov11-s-600-full-adamw/best.pt")
    evaluate("weights_v2/runs-yolov11-s-1000-full-sgd/best.pt")
    # evaluate("weights_v2/runs-yolov11-s-2000-full-sgd/best.pt")
    # evaluate("weights_v2/runs-yolov10-s-600-full-adamw/best.pt")
    # evaluate("weights_v2/runs-yolov10-s-1000-full-sgd/best.pt")
