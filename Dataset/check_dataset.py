import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    PATH_DATASET = "CityScapes_road_car_segmentation/train"

    data = os.listdir(os.path.join(PATH_DATASET, "xs"))

    for img in data:
        image = np.load(os.path.join(PATH_DATASET, os.path.join("xs"), img), allow_pickle=True)
        masks = np.load(os.path.join(PATH_DATASET, os.path.join("ys"), img), allow_pickle=True)

        f, pos = plt.subplots(1, 3)
        pos[0].imshow(image)
        pos[1].imshow(masks[:, :, 0])
        pos[2].imshow(masks[:, :, 1])
        plt.show()
        print()
