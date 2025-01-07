#!/usr/bin/env python3

""" docstring goes here"""

import os
import shutil
import sys
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()
