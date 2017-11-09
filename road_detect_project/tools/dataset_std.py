import os
import numpy as np
from PIL import Image

import road_detect_project.augmenter.utils as utils

"""
Tool to find std of a dataset. This can then be put into the config area.
Need to do this before running cnn because it's expensive to go through all the images.
"""


def calculate_std_from_dataset(path, dataset):
    variance = []
    folder_path = os.path.join(path, dataset, 'data')

    tiles = utils.get_image_files(folder_path)

    samples = []

    i = 0
    for tile in tiles:
        i += 1
        print(tile)
        img = Image.open(os.path.join(folder_path, tile), 'r')

        s, v = get_image_estimate(img)

        samples.append(s)
        variance.append(v)

        if (i % 10 == 0):
            print("Progress", i)

    combined = np.concatenate(samples)
    print("Real std :", np.std(combined))

    dataset_std = np.sqrt(np.sum(variance) / len(variance))

    return dataset_std


def get_image_estimate(image):
    image_arr = np.asarray(image)
    image_arr = image_arr.reshape(
        image_arr.shape[0] * image_arr.shape[1], image_arr.shape[2])
    channels = image_arr.shape[1]

    if channels == 4:
        temp = image_arr[:, 3] > 0
        new_arr = image_arr[temp]
        new_arr = new_arr[:, 0:3]
    else:
        new_arr = image_arr

    new_arr = new_arr / 255.0
    arr = new_arr.reshape(new_arr.shape[0] * new_arr.shape[1])
    np.random.shuffle(arr)

    return np.array(arr[0:1000]), np.var(new_arr)


if __name__ == '__main__':
    path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
    dataset = "train"

    std = calculate_std_from_dataset(path, dataset)
    print(std)

"""
Test:
Real std : 0.184645474245
0.178000648409

Train:
Real std : 0.189560314044
0.182885138909

Val:
Real std : 0.193021380266
0.184505713473

"""
