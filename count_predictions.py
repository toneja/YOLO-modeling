#!/usr/bin/env python3

import numpy as np
import os


def count_predictions():
    pred_counts = [0, 0, 0, 0]
    for file in os.listdir("runs/detect/predict/labels"):
        if file.endswith(".txt"):
            preds = np.genfromtxt(
                f"runs/detect/predict/labels/{file}", dtype=str, delimiter=" "
            )
            for pred in preds:
                pred_counts[int(pred[0])] += 1
    total_preds = sum(pred_counts)
    print("Objects counted:")
    print("================")
    print(f"Appressoria:  {pred_counts[0]}, {round(pred_counts[0] / total_preds * 100, 1)}%")
    print(f"Debris:       {pred_counts[1]}, {round(pred_counts[1] / total_preds * 100, 1)}%")
    print(f"Germinated:   {pred_counts[2]}, {round(pred_counts[2] / total_preds * 100, 1)}%")
    print(f"Ungerminated: {pred_counts[3]}, {round(pred_counts[3] / total_preds * 100, 1)}%")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    count_predictions()
