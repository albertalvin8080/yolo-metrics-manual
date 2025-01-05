from some_utils import MODEL, YOLOV10, YOLOV11

if MODEL == YOLOV10:
    from ultralytics import YOLOv10 as YOLO
elif MODEL == YOLOV11:
    from ultralytics import YOLO
import some_utils
import cv2 as cv
import datetime
import json
import os

def evaluate(model_weights, confidence=0.2, should_save=True, verbose=False):
    data_yaml = """
    path: C:/Users/Albert/Documents/A_Programacao/_GITIGNORE/cataovo-annotations/YOLOv10-structure
    train: images/train  
    val: images/val      
    
    names:
      0: egg
    """

    with open("data.yaml", "w") as f:
        f.write(data_yaml)

    # evaluation = f"evaluation-{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}/"
    # os.makedirs(evaluation, exist_ok=False)
    split = model_weights.split("/")
    evaluation = "evaluation/" + split[1] + "-" + split[2]
    os.makedirs(evaluation, exist_ok=True)

    old_dirs = set(os.listdir(evaluation))
    model = YOLO(model_weights)

    metrics = model.val(
        split="val",
        batch=32,
        iou=0.7,
        conf=confidence,
        data="data.yaml",
        project=evaluation,
        verbose=verbose,
        save_txt=False,
        save_json=True,
        save_conf=should_save,
        plots=should_save,
    )
    
    # print(old_dirs)
    # print(set(os.listdir(evaluation)))
    # print((set(os.listdir(evaluation)) - old_dirs))
    path = (set(os.listdir(evaluation)) - old_dirs).pop()  # last created val*/ dir
    metrics_json_path = evaluation + "/" + path + "/metrics.json"
    with open(metrics_json_path, "w") as json_file:
        # json.dump(str(metrics), json_file, indent=4)
        json.dump(metrics.results_dict, json_file, indent=4)
        # json.dump(metrics.mean_results, json_file, indent=4)
        # json.dump(metrics.fitness, json_file, indent=4)
        # json.dump(metrics.ap_class_index, json_file, indent=4)
        pass

    return metrics


if __name__ == "__main__":
    if MODEL == YOLOV10:
        # evaluate("weights/train-yolov10-s-base-300-earlystop/best.pt")
        # evaluate("weights/train-yolov10-s-base-300-earlystop/last.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-500-full/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-500-full/last.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-sgd/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-sgd/last.pt")
        evaluate("weights/runs-yolov10-s-jameslahn-1000-full-adamw/best.pt")
        # evaluate("weights/runs-yolov10-s-jameslahn-1000-full-adamw/last.pt")
    elif MODEL == YOLOV11:
        evaluate("weights/runs-yolov11-s-base-1000-full-sgd/best.pt")
        # evaluate("weights/runs-yolov11-s-base-1000-full-sgd/last.pt")
