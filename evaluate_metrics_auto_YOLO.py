from ultralytics import YOLO
import some_utils
import cv2 as cv
import datetime
import json
import os

def evaluate(model_weights, confidence=0.15, should_save=True, verbose=False):
    data_yaml = """
    #path: C:/Users/Albert/Documents/A_Programacao/_GITIGNORE/cataovo-annotations/old-YOLOv10-structure
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
    evaluation = "evaluation_v2/" + split[1] + "-" + split[2]
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
    evaluate("weights_v2/runs-yolov10-s-600-full-adamw/best.pt")
    evaluate("weights_v2/runs-yolov10-s-1000-full-sgd/best.pt")

    evaluate("weights_v2/runs-yolov11-s-600-full-adamw/best.pt")
    evaluate("weights_v2/runs-yolov11-s-1000-full-sgd/best.pt")
    evaluate("weights_v2/runs-yolov11-s-2000-full-sgd/best.pt")
