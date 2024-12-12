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
    results = model.train(
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
    model.export(format="onnx")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_yolo(sys.argv[1])
    else:
        ds_count = 0
        for folder in os.listdir("."):
            if os.path.isdir(folder) and os.path.exists(f"{folder}/data.yaml"):
                train_yolo(os.path.basename(folder))
                ds_count += 1
    if ds_count == 0:
        sys.exit(f"Usage: {sys.argv[0]} [DATASET]")
