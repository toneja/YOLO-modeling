#!/usr/bin/env python3

""" docstring goes here"""

import sys
from ultralytics import YOLO


def train_yolo(dataset):
    """ docstring goes here """
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
        device="cpu",   # Fix this to use GPU
    )
    # eval model
    model.val()
    # export model
    # model.export(format="onnx")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_yolo(sys.argv[1])
    else:
        sys.exit(f"Usage: {sys.argv[0]} [DATASET]")
