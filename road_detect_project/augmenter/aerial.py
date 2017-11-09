import random
import numpy as np

import road_detect_project.augmenter.utils as utils
from road_detect_project.augmenter.dataset import Dataset


class Creator(object):
    """
    Dynamically load and convert data to appropriate format for Tensorflow.
    """

    def __init__(self, dataset_path, dim=(64, 16), rotation=False,
                 preprocessing=True, only_mixed=False, std=1,
                 mix_ratio=0.5, reduce_testing=1, reduce_training=1,
                 reduce_validation=1):
        self.dim_data = dim[0]
        self.dim_label = dim[1]
        self.only_mixed_labels = only_mixed
        self.rotation = rotation
        self.preprocessing = preprocessing
        self.mix_ratio = mix_ratio
        self.std = std
        self.reduce_testing = reduce_testing
        self.reduce_training = reduce_training
        self.reduce_validation = reduce_validation
        self.dataset_path = dataset_path  # Load paths to all images found in dataset

        self.img_have_alpha = True  # TODO: config

        self.print_verbose()

    def load_dataset(self):
        test_path, train_path, valid_path = utils.get_dataset(self.dataset_path)
        self.test = Dataset("Test set", self.dataset_path, test_path, self.reduce_testing)
        self.train = Dataset("Train set", self.dataset_path, train_path, self.reduce_training)
        self.valid = Dataset("Valid", self.dataset_path, valid_path, self.reduce_validation)

    def dynamically_create(self, sample_per_images, enable_label_noise=False,
                           label_noise=0.0, only_mixed=False):
        """
        Samples patch datasets at runtime. Creates a validation, test_PyQt5, and training set.
        :return train,valid,test_PyQt5
        """
        self.load_dataset()
        print('')
        print("number of test_PyQt5 images : {}, number of train images {},number of valid images {}".format(
            self.test.num_img, self.train.num_img, self.valid.num_img))
        test = self.sample_data(self.test, sample_per_images)
        # TODO: Rotation should be renamed to data augmentation, or a new parameter. Only if rotation currently.
        # TODO: only_mixed_labels not in init!
        train = self.sample_data(self.train, sample_per_images,
                                 mixed_labels=self.only_mixed_labels,
                                 rotation=self.rotation,
                                 label_noise=label_noise,
                                 label_noise_enable=enable_label_noise)
        valid = self.sample_data(self.valid, sample_per_images)
        return train, valid, test

    def sample_data(self, dataset, sample_per_images, mixed_labels=False, rotation=False,
                    curriculum=None, curriculum_threshold=1.0, label_noise_enable=False,
                    label_noise=0.0, best_trade_off=0.5):
        """
        Use paths to open data image and corresponding label image. 
        Can apply random rotation, and then samples samples_per_images 
        amount of images which is returned in data and label array.
        In addition, the sampling considers the balance between road 
        and non-road pixels, if mixed_labels are set toTrue, 
        label noise is added to label images if enabled,
         and curriculum enables sampling for a staged dataset.
         """
        nr_class = 0
        nr_total = 0

        dropped_images = 0
        curriculum_dropped = 0
        curriculum_road_dropped = 0
        nr_open_images = 0

        dim_data = self.dim_data
        dim_label = self.dim_label

        max_image_samples = int(sample_per_images * dataset.reduce)

        max_arr_size = dataset.num_img * max_image_samples

        # data = np.empty((max_arr_size, dim_data * dim_data * 3), dtype="float32")
        # label = np.empty((max_arr_size, dim_label * dim_label), dtype="float32")

        data = np.empty((max_arr_size, dim_data, dim_data, 3), dtype="float32")
        label = np.empty((max_arr_size, dim_label, dim_label), dtype="float32")

        print('')
        if label_noise_enable:
            print('==============================================')
            print('Noise added to labels: {}'.format(label_noise))
            print('==============================================')
        print("Sampling examples for {}".format(dataset.base))

        # If mixed labels , there will be a lot of trial and
        # if mixed_labels:
        #    max_image_samples *= 2

        # Images are opened, rotated and max_image_Samples examples are extracted per image.
        image_queue = list(range(dataset.num_img))
        example_counter = max_arr_size
        idx = 0
        while example_counter > 0:
            if (nr_open_images % dataset.num_img == 0):
                # Shuffle image queue list,so there is no pattern in order.
                random.shuffle(image_queue)

            # rotating queue
            image_idx = image_queue.pop(0)
            image_queue.append(image_idx)
            nr_open_images += 1

            img, lab = dataset.open_image(image_idx)
            width, height = img.size
            width = width - dim_data
            height = height - dim_data

            # print("sampling", width, height)

            # TODO: add noise
            # if label_noise_enable:
            #     lab, prob = utils.add_artificial_road_noise(lab, label_noise)

            rot = 0
            if rotation:
                rot = random.uniform(0.0, 360.0)
            image_img = np.asarray(img.rotate(rot))
            label_lab = np.asarray(lab.rotate(rot))
            # Some selections will definitely fail, but because of the rotating queue,
            # eventually we have enough examples.
            # This will also mean images that have a lot of no-content will have less samples.
            for i in range(max_image_samples):
                x = random.randint(0, width)
                y = random.randint(0, height)
                data_temp = image_img[x:x + dim_data, y:y + dim_data]
                label_temp = label_lab[x:x + dim_data, y:y + dim_data]

                if self.img_have_alpha:
                    alpha_min = np.amin(data_temp[0:dim_data, 0:dim_data, 3])
                    if alpha_min <= 0:
                        # If a single pixel is transparent, the patch is outside the border.
                        dropped_images += 1
                        continue

                    # convert to RGB
                    data_temp = data_temp[0:dim_data, 0:dim_data, 0:3]

                # TODO: new config parameter
                if rotation:
                    # Increase diversity of samples by flipping horizontal and vertical.
                    # Smart for aerial imagery, because you can flip in two directions.
                    # For natural imagery (sky etc) horizontal flips is bad.
                    # Characters all flips are probably bad.
                    choice = random.randint(0, 2)
                    if choice == 0:
                        data_temp = np.flipud(data_temp)
                        label_temp = np.flipud(label_temp)
                    elif choice == 1:
                        data_temp = np.fliplr(data_temp)
                        label_temp = np.fliplr(label_temp)
                data_sample = utils.from_rgb_to_array(data_temp)
                label_sample = utils.create_image_label(label_temp, dim_data, dim_label)
                # data_sample = data_temp
                # label_sample = label_temp
                if self.preprocessing:
                    data_sample = utils.normalize(data_sample, self.std)

                    # Count percentage of labels contain roads.
                contains_class = not label_sample.max() == 0
                if (mixed_labels and nr_class / float(nr_total + 1e-10) < self.mix_ratio and not contains_class):
                    # Will sample same amount from road and non-road class
                    continue

                if curriculum and curriculum_threshold < 1.0:
                    """这部分处理课程学习的样本"""
                    # This slows down sampling considerably, so only running once,
                    # and storing dataset is a given.
                    # If threshold is 1, only random sampling, with normal dataset distribution.
                    output = curriculum(np.array([data_sample]))
                    output = utils.create_threshold_image(output, best_trade_off)
                    diff = np.sum(np.abs(output[0] - label_sample)) / (dim_label * dim_label)

                    # Patches with roads, are automatically harder,
                    # and have a a bit more lenient threshold.
                    if diff > curriculum_threshold:
                        curriculum_road_dropped += int(contains_class)
                        curriculum_dropped += 1
                        continue

                nr_total += 1
                nr_class += int(contains_class)

                if not self.img_have_alpha:
                    # RGB only. Only filters out entirely white or black area
                    max_element = data_sample.max()
                    min_element = data_sample.min()

                    # will filter out a whole lot of images.
                    if max_element != min_element:
                        data[idx] = data_sample
                        label[idx] = label_sample
                        idx += 1
                        example_counter -= 1
                    else:
                        dropped_images += 1
                else:
                    # RGBA. Filters out areas that is determined to be non-content.
                    # (Bigger white areas set to transparent)
                    data[idx] = data_sample
                    label[idx] = label_sample
                    idx += 1
                    example_counter -= 1

                if example_counter <= 0:
                    break
            # Reduce samples per image after first pass through
            if not mixed_labels and nr_open_images % dataset.num_img == 0:
                max_image_samples = max(10, int(max_image_samples * 0.9))
                print("------ Reduce sampling rate to {}".format(max_image_samples))

            if nr_open_images % 50 == 0:
                print("------ Input image : {} / {}".format(nr_open_images, dataset.num_img))
                print("------ Patches remaining : {}".format(example_counter))

        print("------ Extracted {} images from {}".format(
            data.shape[0], dataset.name))
        print("------ Images containing class {} / {},which is {:.2f}%".format(
            nr_class, nr_total, nr_class * 100 / float(nr_total)))
        print("------ Dropped {} images".format(dropped_images))

        if curriculum:
            print("------ Dropped {} patches because of curriculum".format(curriculum_dropped))
            if curriculum_dropped == 0:
                print("------ No road patches dropped.")
            else:
                print("------ {} Road patches dropped.".format(curriculum_road_dropped))
                print("------ Dropped {:.2f} patches because of curriculum".format(
                    curriculum_road_dropped / float(curriculum_dropped)))

        return data, label

    def print_verbose(self):
        print("Initializing dataset creator")
        print("------ Data size {} X {}".format(self.dim_data, self.dim_data))
        print("------ Label size {} X {}".format(self.dim_label, self.dim_label))
        print("------ Rotation : {}, Preprocessing : {}, and with std : {}".format(
            self.rotation, self.preprocessing, self.std))

        if self.only_mixed_labels:
            print("------ CAUTION: will only include labels containing class of interest")
        print('')


if __name__ == '__main__':
    path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts"
    train = Creator(path, preprocessing=False)
    data, test, valid = train.dynamically_create(100)
    print(data[0].shape)
    print(data[1].shape)
    print(test[0].shape)
    print(valid[0].shape)
    print(data[1][1])
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(data[0][3])
    plt.figure()
    plt.imshow(data[1][3], cmap='gray')
    plt.show()
