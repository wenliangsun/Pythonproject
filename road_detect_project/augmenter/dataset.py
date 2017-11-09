import os
from PIL import Image

import road_detect_project.augmenter.utils as utils


class Dataset(object):
    """
    Helper object, that uses os methods to check validity of test_PyQt5, valid or train dataset.
    Collect all image files and base path. Reduce property used to limit the sampling rate.
    """

    def __init__(self, name, base, folder, reduce=None):
        self.name = name
        self.base = os.path.join(base, folder)
        self.img_paths = self._get_image_files(self.base)
        self.reduce = reduce
        # self.num_img = len(self.img_paths)
        self.num_img = 10

    def open_image(self, i):
        image_path, label_path = self.img_paths[i]
        image = Image.open(os.path.join(self.base, 'data', image_path), 'r').convert('RGBA')
        label = Image.open(os.path.join(self.base, 'labels', label_path), 'r').convert('L')
        return image, label

    def _get_image_files(self, path):
        '''
        Each path should contain a data and labels folder containing images.
        Creates a list of tuples containing path name for data and label.
        '''
        images = utils.get_image_files(os.path.join(path, 'data'))
        labels = utils.get_image_files(os.path.join(path, 'labels'))

        self._is_valid_dataset(images, labels)
        return list(zip(images, labels))

    def _is_valid_dataset(self, images, labels):
        if len(images) == 0 or len(labels) == 0:
            raise Exception("Data or labels folder does not contain any images")

        if len(images) != len(labels):
            raise Exception("Not the same number of images and labels")

        for i in range(len(images)):
            # if os.path.splitext(images[i])[0] != os.path.splitext(labels[i])[0]:
            #     raise Exception("images", images[i], "does not match label", labels[i])
            if images[i].split(".")[0] != labels[i].split(".")[0]:
                raise Exception("images", images[i], "does not match label", labels[i])


# 测试
if __name__ == '__main__':
    path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
    train_data = Dataset("train", path, "train")

    img, label = train_data.open_image(2)
    print(img)
    print(label)
