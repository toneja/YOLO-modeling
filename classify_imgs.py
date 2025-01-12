#!/usr/bin/env python3

""" docstring goes here"""

import os
import shutil
import sys
import numpy as np
from tabulate import tabulate
from ultralytics import YOLO


def classify_images(model_path, image_dir):
    """docstring goes here"""
    # Clear out any old prediction results
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    # Load the previously trained model
    model = YOLO(model_path)
    # Loop through the testing directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if not image_name.lower().endswith((".jpg", ".jpeg")):
            continue
        # Evaluate each image
        model.predict(
            image_path, save=True, imgsz=1024, conf=0.5, save_txt=True, save_conf=True
        )
        print(f"Processed {image_name}")


def count_predictions(project):
    """docstring goes here"""
    classes = np.genfromtxt(f"{project}/classes.txt", dtype=str)
    pred_counts = [0] * len(classes)
    for file in os.listdir("runs/detect/predict/labels"):
        if file.endswith(".txt"):
            preds = np.genfromtxt(
                f"runs/detect/predict/labels/{file}", dtype=str, delimiter=" "
            )
            for pred in preds:
                pred_counts[int(pred[0])] += 1
    total_preds = sum(pred_counts)
    sep = "=" * (9 + len(project))
    print(f"\n{sep}\nProject: {project.upper()}\n{sep}")
    results = []
    headers = ["Class", "Count", "Percentage"]
    i = 0
    for pred_class in classes:
        results.append(
            [
                pred_class,
                pred_counts[i],
                f"{round(pred_counts[i] / total_preds * 100, 1)}%",
            ]
        )
        i += 1
    print(tabulate(results, headers=headers))


def main():
    """docstring goes here"""
    project = ""
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        project = sys.argv[1]
    else:
        for folder in os.listdir("."):
            if os.path.isdir(folder) and os.path.exists(f"{folder}/data.yaml"):
                project = folder
    if not project:
        sys.exit(f"Usage: {sys.argv[0]} [DATASET]")
    model_path = f"{project}/runs/train/yolo_labelstudio_train/weights/best.pt"
    image_dir = f"{project}/images/classify"
    if os.path.exists(model_path) and os.path.exists(image_dir):
        classify_images(model_path, image_dir)
        count_predictions(project)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()
