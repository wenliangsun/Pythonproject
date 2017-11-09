import os, sys
import numpy as np

from road_detect_project.augmenter.aerial import Creator
from road_detect_project.model.params import dataset_params, optimization_params
from road_detect_project.model.road_model_02 import ConvModel, Evaluator


class CurriculumDataset(object):
    def __init__(self, teacher, dataset_path, store_path,
                 dataset_config, best_trade_off=0.5):
        self.teacher = teacher
        self.dataset_path = dataset_path
        self.store_path = store_path
        self.dataset_config = dataset_config
        self.rotate = dataset_config.use_rotation
        self.trade_off = best_trade_off

        if os.path.exists(self.store_path):
            raise Exception("Store path already exists")
        else:
            os.makedirs(self.store_path)
            os.makedirs(os.path.join(self.store_path, "train"))
            os.makedirs(os.path.join(self.store_path, "valid"))

        # TODO: create a measurement about sample
        self.teacher.build()
        self.evaluate = self.teacher.simple_predict

        self.creator = Creator(self.dataset_path,
                               dim=(self.dataset_config.input_dim, self.dataset_config.output_dim),
                               rotation=self.rotate,
                               preprocessing=self.dataset_config.use_preprocessing,
                               only_mixed=self.dataset_config.only_mixed_labels,
                               std=self.dataset_config.dataset_std,
                               mix_ratio=self.dataset_config.mix_ratio,
                               reduce_testing=self.dataset_config.reduce_testing,
                               reduce_training=self.dataset_config.reduce_training,
                               reduce_validation=self.dataset_config.reduce_validation)
        self.creator.load_dataset()

    def create_dataset(self, is_baseline, threshold=None,
                       base_sample=100, secondary_sample=100):
        print("------ Starting sampling.  WARNING: this might take a while.")

        # Sampling at different thresholds.
        if threshold is None:
            threshold = np.arange(0.05, 1, 0.05)
        if is_baseline:
            threshold = np.ones(shape=threshold.shape)

        print("------ Main dataset")
        self._generate_stage("stage0", threshold[0], base_sample)
        for i in range(1, threshold.shape[0]):
            print("------ Stage{} dataset".format(i))
            self._generate_stage("stage{}".format(i), threshold[i], secondary_sample)

        self._generate_set("valid", self.creator.valid, base_sample)
        self._generate_set("test_PyQt5", self.creator.test, base_sample)

    def _generate_set(self, set_name, dataset, samples):
        """
        Validation and test_PyQt5 data is also pre-generated,
        this mean the result is self contained
        """
        data, labels = self.creator.sample_data(
            dataset=dataset, sample_per_images=samples)
        stage_path = os.path.join(self.store_path, set_name)
        os.makedirs(os.path.join(stage_path, "labels"))
        os.makedirs(os.path.join(stage_path, "data"))
        np.save(os.path.join(stage_path, "labels", "examples"), labels)
        np.save(os.path.join(stage_path, "data", "examples"), data)

    def _generate_stage(self, name, threshold, samples):
        """
        Training set is a special case, which involve training folder with several stages. 
        These stages can be introduced in the active training data over time. 
        Slowly transforming the simple distribution to the real dataset distribution of data.
        """
        print("SAMPLES", samples)
        stage_path = os.path.join(self.store_path, "train", name)
        os.makedirs(stage_path)

        data, labels = self.creator.sample_data(
            self.creator.train,
            sample_per_images=samples,
            mixed_labels=self.dataset_config.only_mixed_labels,
            rotation=self.rotate,
            curriculum=self.evaluate,
            curriculum_threshold=threshold,
            best_trade_off=self.trade_off)

        os.makedirs(os.path.join(stage_path, "labels"))
        os.makedirs(os.path.join(stage_path, "data"))
        np.save(os.path.join(stage_path, "labels", "examples"), labels)
        np.save(os.path.join(stage_path, "data", "examples"), data)


if __name__ == '__main__':
    """This tool pre-generate a patch dataset. 
    The tool is especially necessary for curriculum learning. 
    The reason for not doing this every time the network is trained, 
    is that a previously trained model needs to be loaded in order to do difficulty estimation.
    """

    print("Creating curriculum learning dataset")
    path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
    stages = np.array([0.1, 1.0])
    trade_off = 0.5
    init_stage_sample = dataset_params.samples_per_image
    curr_stage_sample = init_stage_sample
    init_model = r"./road_model/model_2/model.ckpt"

    model = ConvModel()
    teacher = Evaluator(model=model, op_params=optimization_params, init_model=init_model)
    generator = CurriculumDataset(teacher, path, r"./my_data/",
                                  dataset_params, best_trade_off=0.5)
    generator.create_dataset(is_baseline=True, threshold=stages,
                             base_sample=init_stage_sample,
                             secondary_sample=curr_stage_sample)
