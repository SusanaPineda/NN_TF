import json
import numpy as np
import cv2
import os
from data_loader import DataManager


def get_mask(path, classes):
    # get data from json
    with open(path) as f:
        data = json.load(f)

    h = data['imgHeight']
    w = data['imgWidth']

    masks = np.zeros((h, w, len(classes)), dtype=np.uint8)

    for object in data['objects']:
        if object['label'] in classes:
            idx = classes.index(object['label'])
            points = np.array(object['polygon'], np.int32)
            mask = cv2.fillConvexPoly(np.array(masks[:, :, idx]), points, 255)
            masks[:, :, idx] = cv2.bitwise_or(masks[:, :, idx], mask)

    return masks


def save_npy(path, data):
    np.save(path, data)


if __name__ == '__main__':
    DATASET_PATH = ""
    DATASET_NAME = ""

    DIVISIONS = ["train", "val", "test"]
    classes = ['road', 'car']

    name_cont = 0

    for DIVISION in DIVISIONS:
        path = os.path.join(DATASET_PATH, DIVISION)
        data = os.listdir(path)
        for d in data:
            data1 = os.listdir(os.path.join(path, d))
            for d1 in data1:
                if d1.split('.')[1] == "json":
                    masks = get_mask(os.path.join(path, os.path.join(d, d1)), classes)

                    img_path = f'{d1.split("_")[0]}_{d1.split("_")[1]}_{d1.split("_")[2]}_leftImg8bit.png'
                    img = cv2.imread(os.path.join(path, os.path.join(d, img_path)))

                    path_mask = ""
                    try:
                        os.mkdir(path_mask)
                    except OSError:
                        print("Creation of the directory masks failed")
                    else:
                        print("Successfully created the masks directory")

                    path_img = ""
                    try:
                        os.mkdir(path_img)
                    except OSError:
                        print("Creation of the directory failed")
                    else:
                        print("Successfully created the directory")

                    save_npy(path_mask, masks)
                    save_npy(path_img, img)
