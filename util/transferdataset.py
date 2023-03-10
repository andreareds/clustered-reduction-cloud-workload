"""
Interface of a Dataset class with shared functionalities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pickle
from util.dataset import DatasetInterface
from sklearn.utils import shuffle


class TransferDatasetInterface(DatasetInterface):
    def __init__(self, trainingfiles=[], testfiles=[], input_window=10, output_window=1, horizon=0,
                 training_features=[], target_name=[],
                 train_split_factor=0.8):
        """
        Constructor of the DatasetInterface class
        :param trainingfiles: list: paths of the datasets for training in .csv format. Default = []
        :param testfiles: list: paths of the datasets for test in .csv format. Default = []
        :param input_window: int: input sequence, number of timestamps of the time series used for training the model
        :param output_window: int: output sequence, length of the prediction. Default = 1 (one-step-ahead prediction)
        :param horizon: int: index of the first future timestamp to predict. Default = 0
        :param training_features: array of string: names of the features used for the training. Default = []
        :param target_name: array of strings: names of the column to predict. Default = []
        :param train_split_factor: float: Training/Test split factor Default = 0.8
        """
        super().__init__("", input_window, output_window, horizon, training_features, target_name, train_split_factor)

        self.name = trainingfiles[0] + '-' + testfiles[-1]
        "string: name pof the experiment"

        self.X_tmp = []
        """list: Full dataset features in windowed format"""
        self.y_tmp = []
        """list: Full dataset labels in windowed format"""
        self.X_array_tmp = []
        """list: Full dataset features in series format """
        self.y_array_tmp = []
        """list: Full dataset labels in series format """
        self.X_train_tmp = []
        """list: Training features in windowed format"""
        self.X_test_tmp = []
        """list: Test features in windowed format"""
        self.y_train_tmp = []
        """list: Training labels in windowed format"""
        self.y_test_tmp = []
        """list: Test labels in windowed format"""
        self.X_train_array_tmp = []
        """list: Training features in series format"""
        self.y_train_array_tmp = []
        """list: Training labels in series format"""
        self.X_test_array_tmp = []
        """list: Test features in series format"""
        self.y_test_array_tmp = []
        """list: Test labels in series format"""

        # Input files
        self.data_file = trainingfiles + testfiles
        """list: dataset file names"""
        self.training_file = trainingfiles
        """list: training dataset names"""
        self.test_file = testfiles
        """list: test dataset names"""

    def data_save(self, name):
        """
        Save the dataset using pickle package
        :param name: string: name of the output file
        :return: None
        """
        super(TransferDatasetInterface, self).data_save()

    def data_load(self, name):
        """
        Load the dataset using pickle package
        :param name: string: name of the inout file
        :return: None
        """
        return super(TransferDatasetInterface, self).data_load()

    def data_summary(self):
        """
        Print a summary of the dataset
        :return: None
        """
        return super(TransferDatasetInterface, self).data_summary()

    def dataset_creation(self):
        """
        Create all the datasets components with the training and test sets split.
        :return: None
        """
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")

        for f in self.training_file:
            # read the csv file into a pandas dataframe
            df = pd.read_csv(self.data_path + f)

            # windowed dataset creation
            columns = df[self.training_features].to_numpy()
            self.X_tmp, self.y_tmp = self.windowed_dataset(columns)
            split_value = int(self.X_tmp.shape[0] * self.train_split_factor)
            self.y_train_tmp = self.y_tmp[:split_value]
            # self.y_test_tmp = self.y_tmp[split_value:]
            self.X_train_tmp = self.X_tmp[:split_value]
            # self.X_test_tmp = self.X_tmp[split_value:]

            # unidimensional dataset creation
            self.X_array_tmp = df[self.target_name].to_numpy()
            if len(self.target_name) == 1:
                self.X_array_tmp = self.X_array_tmp.reshape(-1, 1)
            split_value = int(self.X_array_tmp.shape[0] * self.train_split_factor)
            self.X_train_array_tmp = self.X_array_tmp[:split_value]
            self.y_train_array_tmp = self.X_array_tmp[self.horizon + 1:self.horizon + split_value + 1]

            self.X_array_tmp = self.X_array_tmp[:split_value]
            if len(self.target_name) == 1:
                self.X_array_tmp = self.X_array_tmp.reshape(-1, 1)

            # if self.horizon:
            #     self.X_test_array_tmp = self.X_array_tmp[split_value: -self.horizon - 1]
            # else:
            #     self.X_test_array_tmp = self.X_array_tmp[split_value:-1]
            # self.y_test_array_tmp = self.X_array_tmp[self.horizon + split_value + 1:]

            self.X.append(self.X_train_tmp)
            self.y.append(self.y_train_tmp)
            self.X_array.append(self.X_train_array_tmp)
            self.y_array.append(self.y_train_array_tmp)
            self.X_train.append(self.X_train_tmp)
            self.y_train.append(self.y_train_tmp)
            self.X_train_array.append(self.X_train_array_tmp)
            self.y_train_array.append(self.y_train_array_tmp)
            # self.X_test.append(self.X_test_tmp)
            # self.y_test.append(self.y_test_tmp)
            # self.X_test_array.append(self.X_test_array_tmp)
            # self.y_test_array.append(self.y_test_array_tmp)

        for i, f in enumerate(self.test_file):
            # read the csv file into a pandas dataframe
            df = pd.read_csv(self.data_path + f)

            if i % 100 == 0:
                print(i, "out of", len(self.test_file))

            # windowed dataset creation
            columns = df[self.training_features].to_numpy()
            self.X_tmp, self.y_tmp = self.windowed_dataset(columns)
            split_value = int(self.X_tmp.shape[0] * self.train_split_factor)
            # self.y_train_tmp = self.y_tmp[:split_value]
            self.y_test_tmp = self.y_tmp[split_value:]
            # self.X_train_tmp = self.X_tmp[:split_value]
            self.X_test_tmp = self.X_tmp[split_value:]

            # unidimensional dataset creation
            self.X_array_tmp = df[self.target_name].to_numpy()
            if len(self.target_name) == 1:
                self.X_array_tmp = self.X_array_tmp.reshape(-1, 1)
            split_value = int(self.X_array_tmp.shape[0] * self.train_split_factor)

            # self.X_train_array_tmp = self.X_array_tmp[:split_value]
            # self.y_train_array_tmp = self.X_array_tmp[self.horizon + 1:self.horizon + split_value + 1]

            if self.horizon:
                self.X_test_array_tmp = self.X_array_tmp[split_value: -self.horizon - 1]
            else:
                self.X_test_array_tmp = self.X_array_tmp[split_value:-1]
            self.y_test_array_tmp = self.X_array_tmp[self.horizon + split_value + 1:]

            self.X.append(self.X_test_tmp)
            self.y.append(self.y_test_tmp)
            self.X_array.append(self.X_test_array_tmp)
            self.y_array.append(self.y_test_array_tmp)
            # self.X_train.append(self.X_train_tmp)
            # self.y_train.append(self.y_train_tmp)
            # self.X_train_array.append(self.X_train_array_tmp)
            # self.y_train_array.append(self.y_train_array_tmp)
            self.X_test.append(self.X_test_tmp)
            self.y_test.append(self.y_test_tmp)
            self.X_test_array.append(self.X_test_array_tmp)
            self.y_test_array.append(self.y_test_array_tmp)

        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)
        self.X_array = np.concatenate(self.X_array)
        self.y_array = np.concatenate(self.y_array)
        self.X_train = np.concatenate(self.X_train)
        self.y_train = np.concatenate(self.y_train)
        self.X_train_array = np.concatenate(self.X_train_array)
        self.y_train_array = np.concatenate(self.y_train_array)
        self.X_test = np.concatenate(self.X_test)
        self.y_test = np.concatenate(self.y_test)
        self.X_test_array = np.concatenate(self.X_test_array)
        self.y_test_array = np.concatenate(self.y_test_array)

        if self.verbose:
            print("Data size ", self.X.shape)

        if self.verbose:
            print("Training size ", self.X_train.shape, self.X_train_array.shape)
            print("Training labels size", self.y_train.shape, self.y_train_array.shape)
            print("Test size ", self.X_test.shape, self.X_test_array.shape)
            print("Test labels size", self.y_test.shape, self.y_test_array.shape)

    def dataset_normalization(self, methods=["minmax"], scale_range=(0, 1)):
        """
        Normalize the data column according to the specify parameters.
        :param methods: list of strings: normalization methods to apply to each column.
                        Options: ['minmax', 'standard', None], Default = ["minmax"]
        :param scale_range: list of tuples: scale_range for each scaler. Default=(0,1) for each MinMax scaler
        :return: None
        """
        super(TransferDatasetInterface, self).dataset_normalization(methods, scale_range)

    def metadata_creation(self):
        """
        Add metadata to the dataset features. To implement according to the data format.
        :return: None
        """
        pass

    def windowed_dataset(self, dataset):
        """

        :param dataset: np.array: features of the dataset
        :return: X: np.array: windowed version of the features
                 y: np.array: windowed version of the labels
        """
        return super(TransferDatasetInterface, self).windowed_dataset(dataset)

    def dataset_shuffle(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=0)
