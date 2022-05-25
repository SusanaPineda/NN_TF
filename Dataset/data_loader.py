import numpy as np
import tensorflow as tf
from glob import glob

"""
Gestión del dataset para entrenamiento
"""


class DatasetLoader:
    TRAIN = "/train"
    TEST = "/test"
    XS = "/xs"
    YS = "/ys"
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, data_path: str, batch: int, resize: tuple = None):
        self.batch = batch
        self.train_path = data_path + self.TRAIN + self.XS + "/*.npy"
        self.test_path = data_path + self.TEST + self.XS + "/*.npy"
        self.resize = resize

    def read_npy(self, paths):
        data = []
        for path in paths.numpy():
            path_decoded = path.decode("utf-8")
            path_splited = path_decoded.split("\\")
            redirection = ""
            for splits in path_splited[:-4] + ["data"] + path_splited[-2:-1]:
                redirection += f"{splits}/"
            redirection += path_splited[-1]
            data.append(np.load(redirection) / 255)
        return np.array(data).astype(np.float32)

    @tf.function
    def _loader(self, xs_path: str) -> tuple:
        image_tf = tf.py_function(self.read_npy, [xs_path], tf.float32)

        ys_path = tf.strings.regex_replace(xs_path, "xs", "ys")
        mask_tf = tf.py_function(self.read_npy, [ys_path], tf.float32)

        return image_tf, mask_tf

    def get_sets(self, seed: int = 123) -> object:
        train = tf.data.Dataset.list_files(self.train_path, seed=seed)
        train = train.batch(self.batch, drop_remainder=True).map(self._loader, num_parallel_calls=self.AUTOTUNE)
        train = train.prefetch(buffer_size=self.AUTOTUNE)

        test = tf.data.Dataset.list_files(self.test_path, seed=seed)
        test = test.batch(self.batch, drop_remainder=True).map(self._loader, num_parallel_calls=self.AUTOTUNE)
        test = test.prefetch(buffer_size=self.AUTOTUNE)
        return train, test


class DataManager:
    # Constants
    README_JSON = "/readme.json"
    K_FOLD = "k_fold"
    XS_FORMAT = "xs_format"
    YS_FORMAT = "ys_format"
    SPLITS = 5
    TRAIN_TEST = ["train", "test"]
    XS_YS = ["xs", "ys"]
    template_kfold = {"train": {"xs": [],
                                "x_dest": "",
                                "ys": [],
                                "y_dest": ""},
                      "test": {"xs": [],
                               "x_dest": "",
                               "ys": [],
                               "y_dest": ""}
                      }

    @classmethod
    def loadDataset(cls, data_path: str, batch: int) -> DatasetLoader:
        print(f'load dataset{data_path}, number of batchs: {batch}')
        return DatasetLoader(data_path, batch)

    @classmethod
    def _get_files(cls, xs_path: str, ys_regex: list) -> tuple:
        xs = glob(xs_path)
        ys = []
        for x in xs:
            for y in ys_regex:
                x = x.replace(y[0], y[1])
            ys.append(x)
        return np.array(xs), np.array(ys)