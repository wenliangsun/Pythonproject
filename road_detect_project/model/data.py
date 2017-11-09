import os, math, random
import numpy as np
from abc import ABCMeta, abstractclassmethod

from road_detect_project.augmenter.aerial import Creator
from road_detect_project.model.params import dataset_params


# class DataLoader():
#     """load a class """
#
#     @staticmethod
#     def create():
#         pass


class AbstractDataset(metaclass=ABCMeta):
    """
    All dataloader should inherit from this class. 
    Implements chunking of the dataset. For instance,
    subsets of examples are loaded onto GPU iteratively during an epoch.
    This also includes a chunk switching method.
    """

    def __init__(self):
        self.data_set = {"test_PyQt5": None,
                         "train": None,
                         "valid": None}
        self.all_training = []
        self.active = []
        self.all_shared_hook = []  # WHen casted, cannot set_value on them
        self.nr_examples = {}

    @abstractclassmethod
    def load(self, dataset_path):
        """Loading and transforming logic for dataset"""
        return

    def destroy(self):
        pass

    def get_chunk_number(self):
        return len(self.all_training)

    def get_elements(self, idx):
        return len(self.all_training[idx][0])

    def get_total_number_of_batches(self, batch_size):
        s = sum(len(c[0]) for c in self.all_training)
        return math.ceil(s / batch_size)

    def _chunkify(self, dataset, nr_of_chunks, batch_size):
        # Round items per chunk down until there is an exact number of minibatches.
        # Multiple of batch_size
        items_per_chunk = len(dataset[0]) / nr_of_chunks
        if items_per_chunk < batch_size:
            print("Chunk limit too small,or batch size too large.\n"
                  "Each chunk must include at least one batch.")
            raise Exception("Fix chunk_size and batch size.")
        temp = int(items_per_chunk / batch_size)
        items_per_chunk = batch_size * temp
        data, labels = dataset
        # TODO:do floatX operation twice.
        chunks = [[AbstractDataset._float32(data[x:x + items_per_chunk]),
                   AbstractDataset._float32(labels[x:x + items_per_chunk])]
                  for x in range(0, len(dataset[0]), items_per_chunk)]
        # If the last chunk is less than batch size, it is cut.
        # No reason for an unnecessary swap.
        last_chunk_size = len(chunks[-1][0])
        if last_chunk_size < batch_size * 15:
            chunks.pop(-1)
            print("------ Remove last chunk."
                  " {} elements not enough for at least one minibatch of {}".format(
                last_chunk_size, batch_size))

        return chunks

    def set_nr_examples(self, train, valid, test):
        self.nr_examples["train"] = train[0].shape[0]
        self.nr_examples["valid"] = valid[0].shape[0]
        self.nr_examples["test_PyQt5"] = test[0].shape[0]

    def get_report(self):
        return self.nr_examples

    def switch_active_training_set(self, idx):
        """
        Each epoch a large number of examples will be seen by model. 
        Often all examples will not fit on the GPU atthe same time.
        This method, switches the data that are currently reciding in the gpu.
        Will be called nr_of_chunks times per epoch.
        """
        new_chunk_x, new_chunk_y = AbstractDataset._list_to_arr(self.all_training[idx])
        self.active[0] = new_chunk_x
        self.active[1] = new_chunk_y

    def shared_dataset(self, data_xy, cast_to_int=True):
        data_x, data_y = data_xy
        # print(data_x.shape)
        # print(data_y.shape)
        # shared_x = tf.Variable(data_x)
        # shared_y = tf.Variable(data_y)
        shared_x = data_x
        shared_y = data_y
        self.all_shared_hook.append(shared_y)
        if cast_to_int:
            # print("---- Casted to int")
            # Since labels are index integers they have to be treated as such during computations.
            # Shared_y is therefore cast to int.
            return shared_x, shared_y
        else:
            return shared_x, shared_y

    @staticmethod
    def _float32(d):
        return np.asarray(d, dtype="float32")

    @staticmethod
    def _list_to_arr(d):
        d_x, d_y = d
        d_x = np.asarray(d_x, dtype="float32")
        d_y = np.asarray(d_y, dtype="float32")
        return d_x, d_y

    @staticmethod
    def _get_file_path(dataset):
        data_dir, data_file = os.path.split(dataset)
        # TODO: Add some robustness, like checking if file is folder and correct that
        assert os.path.isfile(dataset)
        return dataset

    @staticmethod
    def dataset_check(name, dataset, batch_size):
        # If there are are to few examples for at least one batch,
        # the dataset is invalid.
        if len(dataset[0]) < batch_size:
            print("Insufficent examples in {}. {} examples not enough "
                  "for at least one minibatch".format(
                name, len(dataset[0])))
            raise Exception("Decrease batch_size or increase samples_per_image")

    @staticmethod
    def dataset_sizes(train, valid, test, chunks):
        mb = 1000000.0
        train_size = sum(data.nbytes for data in train) / mb
        valid_size = sum(data.nbytes for data in valid) / mb
        test_size = sum(data.nbytes for data in test) / mb
        nr_of_chunks = math.ceil(train_size / chunks)

        print('---- Minimum number of training chunks: {}'.format(nr_of_chunks))
        print('---- Dataset at least:')
        print('---- Training: \t {}mb'.format(train_size))
        print('---- Validation: {}mb'.format(valid_size))
        print('---- Testing: \t {}mb'.format(test_size))
        return nr_of_chunks

    @staticmethod
    def dataset_shared_stats(image_shape, label_shape, chunks):
        print('')
        print('Preparing shared variables for datasets')
        print('---- Image data shape: {}, label data shape: {}'.format(image_shape, label_shape))
        print('---- Max chunk size of {}mb'.format(chunks))

    @staticmethod
    def dataset_chunk_stats(nr_training_chunks, elements_pr_chunk, elements_last_chunk):
        print('---- Actual number of training chunks: {}'.format(nr_training_chunks))
        print('---- Elements per chunk: {}'.format(elements_pr_chunk))
        print('---- Last chunk size: {}'.format(elements_last_chunk))


class AerialDataset(AbstractDataset):
    def load(self, dataset_path, params=None, batch_size=16):
        print("Creating aerial image dataset")

        self.std = params.dataset_std
        chunks = params.chunk_size

        # TODO: ensure that the dataset is as expected.
        creator = Creator(dataset_path,
                          dim=(params.input_dim, params.output_dim),
                          rotation=params.use_rotation,
                          preprocessing=params.use_preprocessing,
                          std=self.std,
                          only_mixed=params.only_mixed_labels,
                          reduce_testing=params.reduce_testing,
                          reduce_training=params.reduce_training,
                          reduce_validation=params.reduce_validation)
        train, valid, test = creator.dynamically_create(
            params.samples_per_image,
            enable_label_noise=params.use_label_noise,
            label_noise=params.label_noise,
            only_mixed=params.only_mixed_labels)

        # Testing dataset size requirements
        AerialDataset.dataset_check("train", train, batch_size)
        AerialDataset.dataset_check("valid", valid, batch_size)
        AerialDataset.dataset_check("test_PyQt5", test, batch_size)
        # print("*********************************************")
        # print(train[1].shape)
        # print(test_PyQt5[0].shape)

        AbstractDataset.dataset_shared_stats(train[0].shape, train[1].shape, chunks)

        self.set_nr_examples(train, valid, test)

        nr_of_chunks = AbstractDataset.dataset_sizes(train, valid, test, chunks)

        training_chunks = self._chunkify(train, nr_of_chunks, batch_size)

        AerialDataset.dataset_chunk_stats(len(training_chunks),
                                          len(training_chunks[0][0]),
                                          len(training_chunks[-1][0]))

        AbstractDataset.dataset_chunk_stats(
            len(training_chunks),
            len(training_chunks[0][0]),
            len(training_chunks[-1][0]))

        self.active = list(self.shared_dataset(training_chunks[0], cast_to_int=False))

        self.data_set['train'] = self.active
        self.data_set['valid'] = self.shared_dataset(valid, cast_to_int=True)
        self.data_set['test_PyQt5'] = self.shared_dataset(test, cast_to_int=True)

        # Not stored on the GPU, unlike the shared variables defined above.
        self.all_training = training_chunks

        return True

    def gen_data(self, data_name, epoch=1000, batch_size=16):
        chunks = self.get_chunk_number()
        for i in range(epoch):
            if data_name == "train":
                for chunk in range(chunks):
                    self.switch_active_training_set(chunk)
                    nr_elements = self.get_elements(chunk)
                    train_data = self.data_set[data_name]
                    batches = [[train_data[0][x:x + batch_size], train_data[1][x:x + batch_size]]
                               for x in range(0, nr_elements, batch_size)]
                    for batch in batches:
                        yield batch
            else:
                data = self.data_set[data_name]
                nr_elements = data[0].shape[0]
                batches = [[data[0][x:x + batch_size], data[1][x:x + batch_size]]
                           for x in range(0, nr_elements, batch_size)]
                for batch in batches:
                    yield batch


class AerialCurriculumDataset(AbstractDataset):
    """
    Data loader for pre-generated dataset. 
    IE, curriculum learning and datasets too big to fit in main memory.
    The class includes  a method for stage switching and mixing. 
    this method switches the training set and control the behavior of the switch.
    """

    def load_set(self, path, set, stage=None):
        base_path = ''
        if stage is not None:
            base_path = os.path.join(path, set, stage)
        else:
            base_path = os.path.join(path, set)
        data = np.load(os.path.join(base_path, "data", "examples.npy"))
        labels = np.load(os.path.join(base_path, "labels", "examples.npy"))
        return data, labels

    def mix_in_next_stage(self):
        self.stage += 1
        if self.nr_of_stages <= self.stage:
            print("No more stage available")
            return

        current_stage = "stage{}".format(self.stage)

        labels = np.load(os.path.join(self.stage_path, current_stage, "labels", "examples.npy"))
        data = np.load(os.path.join(self.stage_path, current_stage, "data", "examples.npy"))
        print("------ Mixing in {} with {} examples".format(current_stage, data.shape[0]))

        if not dataset_params.with_replacement:
            elements = data.shape[0]
            shuffle_count = 0
            shuffle_index = list(range(elements))
            random.shuffle(shuffle_index)

            for c in range(len(self.all_training)):
                nr_chunk_examples = self.all_training[c][0].shape[0]
                for x in range(nr_chunk_examples):
                    if shuffle_count < elements:
                        i = shuffle_index.pop()

                        self.all_training[c][0][x] = data[i]
                        self.all_training[c][1][x] = labels[i]
                    else:
                        break
                    shuffle_count += 1
        else:
            nr_chunks = len(self.all_training)
            for i in range(data.shape[0]):
                c = random.randint(0, nr_chunks - 1)
                nr_chunk_examples = self.all_training[c][0].shape[0]
                x = random.randint(0, nr_chunk_examples - 1)
                self.all_training[c][0][x] = data[i]
                self.all_training[c][0][x] = labels[i]

    def load(self, dataset_path, params=None, batch_size=16):
        print("------- Loading aerial curriculum dataset")

        chunks = params.chunk_size
        self.std = params.dataset_std

        # For later stage loading
        self.stage = 0
        self.stage_path = os.path.join(dataset_path, "train")
        self.nr_of_stages = len(os.listdir(self.stage_path))

        train = self.load_set(dataset_path, "train", stage="stage{}".format(self.stage))
        valid = self.load_set(dataset_path, "valid")
        test = self.load_set(dataset_path, "test_PyQt5")

        # Testing dataset size requirements
        AerialCurriculumDataset.dataset_check("train", train, batch_size)
        AerialCurriculumDataset.dataset_check("valid", valid, batch_size)
        AerialCurriculumDataset.dataset_check("test_PyQt5", test, batch_size)

        AerialCurriculumDataset.dataset_shared_stats(train[0].shape, train[1].shape, chunks)

        self.set_nr_examples(train, valid, test)

        nr_of_chunks = AerialCurriculumDataset.dataset_sizes(train, valid, test, chunks)

        training_chunks = self._chunkify(train, nr_of_chunks, batch_size)

        AerialCurriculumDataset.dataset_chunk_stats(len(training_chunks),
                                                    len(training_chunks[0][0]),
                                                    len(training_chunks[-1][0]))

        self.active = list(self.shared_dataset(training_chunks[0], cast_to_int=False))
        self.data_set["train"] = self.active
        self.data_set["valid"] = self.shared_dataset(valid, cast_to_int=True)
        self.data_set["test_PyQt5"] = self.shared_dataset(test, cast_to_int=True)

        # Not stored on the GPU, unlike the shared variables defined above.
        self.all_training = training_chunks

        return True


if __name__ == '__main__':
    # dataset = AerialDataset()
    # path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
    # params = dataset_params
    # dataset.load(path, params=params)
    # dataset.switch_active_training_set(0)
    # train_data = dataset.data_set['train']
    # print("train")
    # print(train_data[0].shape)
    # print(train_data[1].shape)
    # valid_data = dataset.data_set['valid']
    # print("valid")
    # next(dataset.gen_data("valid", epoch=1))
    # print(valid_data[0].shape)
    # print(valid_data[1].shape)
    # test_data = dataset.data_set['test_PyQt5']
    # print("test_PyQt5")
    # print(test_data[0].shape)
    # print(test_data[1].shape)
    # print(dataset.get_report())
    # dataset.switch_active_training_set(2)
    # dataset.get_elements(2)
    # new_data = dataset.data_set['train']
    # print(new_data[0].shape)


    # Test curriculum dataset
    dataset_2 = AerialCurriculumDataset()
    path = "../tools/my_data/"
    dataset_2.load(path, params=dataset_params)
    dataset_2.switch_active_training_set(0)
    print("train\n")
    train_data = dataset_2.data_set["train"]
    print(train_data[0].shape)
    print(train_data[1].shape)
    print("valid\n")
    valid_data = dataset_2.data_set["valid"]
    print(train_data[0].shape)
    print(train_data[1].shape)
    print("test_PyQt5\n")
    test_data = dataset_2.data_set["test_PyQt5"]
    print(train_data[0].shape)
    print(train_data[1].shape)
