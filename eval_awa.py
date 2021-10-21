import os
import argparse

import numpy as np
from collections import defaultdict

"""
Evalution script:

example execution:
    python eval_awa.py --gt test_images.txt --pred example_submission.txt

see example_submission.txt for correct submission format

"""
def labelToIdx(pred_label):
    idx = -1
    if pred_label == "chimpanzee":
        idx = 0
    elif pred_label == "giant+panda":
        idx = 1
    elif pred_label == "leopard":
        idx = 2
    elif pred_label == "persian+cat":
        idx = 3
    elif pred_label == "pig":
        idx = 4
    elif pred_label == "hippopotamus":
        idx = 5
    elif pred_label == "humpback+whale":
        idx = 6
    elif pred_label == "raccoon":
        idx = 7
    elif pred_label == "rat":
        idx = 8
    elif pred_label == "seal":
        idx = 9
    return idx

def read_animal_file(fname):
    image_label_dict = {}
    with open(fname) as f:
        for line in f:
            image, label = line.split()
            image_label_dict[image] = label

    return image_label_dict

parser = argparse.ArgumentParser()
parser.add_argument('--gt', help="ground truth labels")
parser.add_argument('--pred', help="file of predictions")
args = parser.parse_args()


gt_dict = read_animal_file(args.gt)
pred_dict = read_animal_file(args.pred)

per_class_accuracy = {"all": []}
incorrect_accuracies = {"chimpanzee": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "giant+panda": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "leopard": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "persian+cat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "pig": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "hippopotamus": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "humpback+whale": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "raccoon": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "rat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "seal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

for image in gt_dict:
    if image not in pred_dict:
        print("Error: {} not in prediction file".format(image))
        raise Exception()

    gt_label = gt_dict[image]
    pred_label = pred_dict[image]

    idx = labelToIdx(pred_label)
    if gt_label == pred_label:
        per_class_accuracy["all"].append(1)
    else:
        per_class_accuracy["all"].append(0)
    incorrect_accuracies[gt_label][idx] += 1



print("Final Accuracy: {:.5f}".format(np.mean(per_class_accuracy["all"])))
percent_acc = {}
for key, list in incorrect_accuracies.items():
    percent_acc[key] = list[labelToIdx(key)]/sum(list)
print(incorrect_accuracies)
print(percent_acc)
