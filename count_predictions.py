#!/usr/bin/env python3

import numpy as np
import os


def return_results(predictions, total):
    return (
        f"Appressoria :  {predictions[0]}, {round(predictions[0] / total * 100, 1)}%\n"
        f"Debris      :  {predictions[1]}, {round(predictions[1] / total * 100, 1)}%\n"
        f"Germinated  :  {predictions[2]}, {round(predictions[2] / total * 100, 1)}%\n"
        f"Ungerminated:  {predictions[3]}, {round(predictions[3] / total * 100, 1)}%"
    )


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
    print(return_results(pred_counts, total_preds))


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    count_predictions()
