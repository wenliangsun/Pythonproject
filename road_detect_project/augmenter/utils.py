import os, random
import numpy as np
from PIL import Image


def normalize(data, std):
    """Data patch is constrast normalized by this method. 
    The second argument std should be the standard deviation of
    the patch dataset."""
    m = np.mean(data)
    data = (data - m) / std
    return data


def get_image_files(path):
    """get the files that in ... """
    print("Retrieving {}".format(path))
    include_extenstions = ['jpg', 'png', 'tiff', 'tif']
    files = [fn for fn in os.listdir(path)
             if any([fn.endswith(ext) for ext in include_extenstions])]
    files.sort()
    return files


def get_dataset(path):
    content = os.listdir(path)
    if not all(x in ['test_PyQt5', 'train', 'val'] for x in content):
        print('Folder does not contain image or label folder. Path probably not correct')
        raise Exception("Please check the path!")
    content.sort()
    return content


def from_rgb_to_array(image):
    data = np.asarray(image, dtype="float32") / 255.0
    return data


def create_image_label(image, dim_data, dim_label):
    y = dim_label
    padding = int((dim_data - y) / 2)
    label = np.asarray(image, dtype="float32")
    label = label[padding:padding + y, padding:padding + y]

    label = label / 255.0
    return label


def create_threshold_image(image, threshold):
    """
    threshold value define the binary split. 
    Resulting binary image only contains 0 and 1, while image contains
    values between 0 and 1.
    """
    binary_arr = np.ones(image.shape)
    low_value_indices = image <= threshold  # Where values are low
    binary_arr[low_value_indices] = 0  # All low values set to 0
    return binary_arr


if __name__ == '__main__':
    path_files = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/val/data"
    path_data = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/val/labels"
    path_content = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts"

    file = get_image_files(path_files)
    data = get_image_files(path_data)
    conetnt = get_dataset(path_content)
    print(conetnt)
    # a = True
    # index = 0
    # while a:
    #     for i in range(14):
    #         if file[i].split('.')[0] != data[i].split('.')[0]:
    #             index = i
    #             a = False
    #             break
    #     a = False
    #         # print(file[i].split('.')[0], data[i].split('.')[0])
    # print(index)
    # print(file[index])
    # print(data[index])
