#!/usr/bin/env python3

""" docstring goes here"""

import os
import shutil
import sys
import torch
from ultralytics import YOLO


def train_yolo(dataset):
    """docstring goes here"""
    # Initialize GPU acceleration if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = 0
    else:
        device = "cpu"
    # Clear out any old results and cache files
    if os.path.exists(f"{dataset}/runs"):
        shutil.rmtree(f"{dataset}/runs")
    for file in os.listdir(f"{dataset}/labels"):
        if file.endswith(".cache"):
            os.remove(f"{dataset}/labels/{file}")
    # load small yolo dataset
    model = YOLO("yolov8s.pt")
    # train model
    model.train(
        data=f"{dataset}/data.yaml",
        epochs=50,
        batch=16,
        imgsz=1024,
        workers=4,
        project=f"{dataset}/runs/train",
        name="yolo_labelstudio_train",
        device=device,
    )
    # eval model
    model.val()
    # export model
    model.export(format="engine")


def main():
    """docstring goes here"""
    ds_count = 0
    if len(sys.argv) == 2 and os.path.exists(f"{sys.argv[1]}/data.yaml"):
        train_yolo(sys.argv[1])
        ds_count += 1
    else:
        for folder in os.listdir("."):
            if os.path.isdir(folder) and os.path.exists(f"{folder}/data.yaml"):
                # fix-up path on first run
                with open(f"{folder}/data.yaml", "r") as file:
                    file_contents = file.read()
                if "DATASET_PATH" in file_contents:
                    new_contents = file_contents.replace("%%DATASET_PATH%%", os.path.abspath(folder))
                    with open(f"{folder}/data.yaml", "w") as file:
                        file.write(new_contents)
                train_yolo(os.path.basename(folder))
                ds_count += 1
    if ds_count == 0:
        sys.exit(f"Usage: {sys.argv[0]} [DATASET]")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()
