"""
Interface of a Dataset class with shared functionalities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.utils import shuffle
from pytorch_forecasting import TimeSeriesDataSet
import copy
import torch


class TransferDatasetWrapper:
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
        self.name = trainingfiles[0] + '-' + testfiles[-1]
        "string: name pof the experiment"

        self.X_train = None
        """TimeSeriesDataSet: Training set"""
        self.X_val = None
        """TimeSeriesDataSet: Validation set"""
        self.X_test = None
        """TimeSeriesDataSet: Test set"""

        self.training_features = training_features
        """list of strings: columns names of the features for the training"""
        self.target_name = target_name
        """list of strings: Columns names of the labels to predict"""
        self.channels = len(self.training_features)
        """int: number of input dimensions"""

        self.data_path = './saved_data/'
        """string: directory path of the dataset"""

        self.train_split_factor = train_split_factor
        """float: training/Test split factor"""

        self.normalization = None
        """list of strings: list of normalization methods to apply to features columns"""
        self.X_scalers = {}
        """dict: dictionary of scaler used for the features"""

        self.input_window = input_window
        """int:  input sequence, number of timestamps of the time series used for training the model"""
        self.stride = 1
        """int: stride for the windowed dataset creation"""
        self.output_window = output_window
        """int: index of the first future timestamp to predict"""
        self.horizon = horizon
        """int: index of the first future timestamp to predict"""

        self.verbose = 1
        """int: level of verbosity of the dataset operations"""

        columns = self.training_features.copy()
        columns.append("time_idx")
        columns.append("group_ids")
        self.X_train_df = pd.DataFrame(columns=columns)
        """DataFrame: Training set"""
        self.X_test_df = pd.DataFrame(columns=columns)
        """DataFrame: Test set"""
        self.X_train_df = self.X_train_df.astype({'time_idx': np.int32})
        self.X_test_df = self.X_test_df.astype({'time_idx': np.int32})

        # Input files
        self.data_file = trainingfiles + testfiles
        """list: dataset file names"""
        self.training_file = trainingfiles
        """list: training dataset names"""
        self.test_file = testfiles
        """list: test dataset names"""

        self.train_dataloader = None
        """"""
        self.val_dataloader = None
        """"""
        self.test_dataloader = None
        """"""

    def data_save(self, name):
        """
        Save the dataset using pickle package
        :param name: string: name of the output file
        :return: None
        """
        pass

    def data_load(self, name):
        """
        Load the dataset using pickle package
        :param name: string: name of the inout file
        :return: None
        """
        pass

    def data_summary(self):
        """
        Print a summary of the dataset
        :return: None
        """
        pass

    def dataset_creation(self, methods=["minmax"], scale_range=(0, 1)):
        """
        Create all the datasets components with the training and test sets split.
        :return: None
        """
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")

        for i_f, f in enumerate(self.training_file):
            # read the csv file into a pandas dataframe
            df = pd.read_csv(self.data_path + f)

            df = df[self.training_features]
            df['time_idx'] = df.index
            df['group_ids'] = str(i_f)
            split_value = int(df.shape[0] * self.train_split_factor)
            df = df.iloc[:split_value]

            self.X_train_df = pd.concat([self.X_train_df, df], ignore_index=True)

        for i_f, f in enumerate(self.test_file):
            if i_f % 100 == 0:
                print(i_f, "out of", len(self.test_file))

            # read the csv file into a pandas dataframe
            df = pd.read_csv(self.data_path + f)

            df = df[self.training_features]
            df['time_idx'] = df.index
            df['group_ids'] = str(len(self.training_file) + i_f)
            split_value = int(df.shape[0] * self.train_split_factor)
            df = df.iloc[split_value:]

            self.X_test_df = pd.concat([self.X_test_df, df], ignore_index=True)

        if self.verbose:
            print("Data normalization")
        if methods is not None and self.channels != len(methods):
            print("ERROR: You have to specify a scaling method for each feature")
            exit(1)

        self.X_scalers = {}
        if methods is not None:
            self.normalization = methods
            for i, feature in zip(range(self.channels), self.training_features):
                if self.normalization[i] is not None:
                    if self.normalization[i] == "standard":
                        self.X_scalers[i] = StandardScaler()
                    elif self.normalization[i] == "minmax":
                        self.X_scalers[i] = MinMaxScaler(scale_range)
                    # window dataset
                    self.X_train_df[feature] = self.X_scalers[i].fit_transform(
                        self.X_train_df[feature].values.reshape(-1, 1))
                    self.X_test_df[feature] = self.X_scalers[i].transform(
                        self.X_test_df[feature].values.reshape(-1, 1))

        max_prediction_length = self.output_window + self.horizon
        training_cutoff = self.X_train_df["time_idx"].max() - max_prediction_length

        if len(self.target_name) == 1:
            # self.X_train_df.cuda()
            # self.X_test_df.cuda()
            self.X_train = TimeSeriesDataSet(
                self.X_train_df[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=self.target_name[0],
                group_ids=["group_ids"],
                min_encoder_length=self.input_window // 2,  # keep encoder length long (as it is in the validation set)
                max_encoder_length=self.input_window,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_categoricals=[],
                # static_categoricals=["group_ids"],
                time_varying_unknown_reals=self.target_name,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                # target_normalizer=None,
                # categorical_encoders=None,
                # scalers=None
            )

            self.X_test = TimeSeriesDataSet(
                self.X_test_df[:-self.horizon - 1],
                time_idx="time_idx",
                target=self.target_name[0],
                group_ids=["group_ids"],
                min_encoder_length=self.input_window // 2,  # keep encoder length long (as it is in the validation set)
                max_encoder_length=self.input_window,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_categoricals=[],
                # static_categoricals=["group_ids"],
                time_varying_unknown_reals=self.target_name,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                # target_normalizer=None,
                # categorical_encoders=None,
                # scalers=None
            )
        else:
            self.X_train = TimeSeriesDataSet(
                self.X_train_df[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=self.target_name,
                group_ids=["group_ids"],
                min_encoder_length=self.input_window // 2,  # keep encoder length long (as it is in the validation set)
                max_encoder_length=self.input_window,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_categoricals=[],
                # static_categoricals=["group_ids"],
                time_varying_unknown_reals=self.target_name,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True
            )

            self.X_test = TimeSeriesDataSet(
                self.X_test_df,
                time_idx="time_idx",
                target=self.target_name,
                group_ids=["group_ids"],
                min_encoder_length=self.input_window // 2,  # keep encoder length long (as it is in the validation set)
                max_encoder_length=self.input_window,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_categoricals=[],
                # static_categoricals=["group_ids"],
                time_varying_unknown_reals=self.target_name,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True
            )

        self.X_val = TimeSeriesDataSet.from_dataset(self.X_train, self.X_train_df, predict=True,
                                                    stop_randomization=True)

        if self.verbose:
            print("Training size ", self.X_train_df.shape)
            print("Test size ", self.X_test_df.shape)

    def loader_creation(self, batch_size=128):
        """
            Create all the datasets components with the training and test sets split.
            :return: None
            """
        self.train_dataloader = self.X_train.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        self.val_dataloader = self.X_val.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
        self.test_dataloader = self.X_test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
