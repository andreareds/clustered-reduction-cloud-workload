from models.model_interface import ModelInterface
import copy
from pathlib import Path
import warnings
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MultivariateDistributionLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import MQF2DistributionLoss, NormalDistributionLoss


# modify n_trials and self.p['epochs']

# class DeepAR(ModelProbabilisticDL):
class TFT(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Probabilistic class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.temp_model = None
        self.parameter_list = {'hidden_size': (8, 512),
                               'lstm_layers': [2],
                               'dropout': (0.1, 0.3),
                               'attention_head_size': (1, 4),
                               'lr': (0.001, 0.1),
                               'hidden_continuous_size': (8, 128),
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50]
                               }
        """dict: Dictionary of hyperparameters search space"""
        self.p = {'hidden_size': 252,
                  'lstm_layers': 2,
                  'dropout': 0.11,
                  'attention_head_size': 2,
                  'lr': 0.002,
                  'hidden_continuous_size': 21,
                  'epochs': 2000,
                  'patience': 50,
                  'batch_size': 256,
                  'gradient_clip_val': 0.28
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""
        self.trainer = None
        self.talos_path = 'talos/tl/TFT/'

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: predictions of the quantiles of the samples X
        """
        predictions = self.model.predict(X, mode='quantiles')  # .cuda()

        return predictions

    def fit_predict(self, X):
        """
        Training the model on self.ds.X_train and self.ds.y_train and predict on samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: predictions of the quantiles of the samples X
        """
        self.fit()
        return self.predict(X)

    def fit(self):
        """
        Training of the model
        :return: None
        """
        self.trainer.fit(
            self.model,
            train_dataloaders=self.ds.train_dataloader,
            val_dataloaders=self.ds.val_dataloader
            # ckpt_path=self.talos_path
        )
        return self.model

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train_array
        :return: np.array: predictions: predictions of the quantiles of the samples trainval set
        """
        return self.predict(self.ds.train_dataloader)  # .to(torch.device('cuda'))

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training set
        :param y: nparray: Validation set
        :return: None
        """
        self.trainer.fit(
            self.model,
            train_dataloaders=X,
            val_dataloaders=y
            # ckpt_path=self.talos_path
        )

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=50, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger(self.talos_path)  # logging results to a tensorboard
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath=self.talos_path,
                                              filename=self.name + '-{epoch:02d}-{val_loss:.2f}')

        self.trainer = pl.Trainer(
            max_epochs=self.p['epochs'],
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=self.p['gradient_clip_val'],
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            logger=logger,
            default_root_dir=self.talos_path
        )

        if len(self.ds.target_name) == 1:
            self.model = TemporalFusionTransformer.from_dataset(
                self.ds.X_train,
                learning_rate=self.p['lr'],
                hidden_size=self.p['hidden_size'],
                attention_head_size=self.p['attention_head_size'],
                dropout=self.p['dropout'],
                hidden_continuous_size=self.p['hidden_continuous_size'],
                output_size=7,  # 7 quantiles by default
                loss=QuantileLoss(),
                reduce_on_plateau_patience=4,
            )
        else:
            loss = NormalDistributionLoss()  # MQF2DistributionLoss(prediction_length=self.ds.output_window +
            # self.ds.horizon)
            #loss.requires_grad = True

            self.model = TemporalFusionTransformer.from_dataset(
                self.ds.X_train,
                learning_rate=self.p['lr'],
                hidden_size=self.p['hidden_size'],
                attention_head_size=self.p['attention_head_size'],
                dropout=self.p['dropout'],
                hidden_continuous_size=self.p['hidden_continuous_size'],
                loss=loss,  # .to(torch.device('cuda')),
                reduce_on_plateau_patience=4,
            )

        # if self.verbose == 1:
        #     print(f"Number of parameters in network: {self.temp_model.size() / 1e3:.1f}k")

    def hyperparametrization(self):
        """
        Search the best parameter configuration using talos
        :return: None
        """
        # create study
        study = optimize_hyperparameters(
            self.ds.train_dataloader,
            self.ds.val_dataloader,
            model_path=self.talos_path,
            n_trials=10000,
            max_epochs=self.p['epochs'],
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=self.parameter_list['hidden_size'],
            hidden_continuous_size_range=self.parameter_list['hidden_continuous_size'],
            attention_head_size_range=self.parameter_list['attention_head_size'],
            learning_rate_range=self.parameter_list['lr'],
            dropout_range=self.parameter_list['dropout'],
            # trainer_kwargs=dict(limit_train_batches=30),
            # reduce_on_plateau_patience=4,
            use_learning_rate_finder=True
        )

        # save study results - also we can resume tuning at a later point in time
        with open(self.talos_path + "test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)

        # show best hyperparameters
        if self.verbose == 1:
            print(study.best_trial.params)

        best_res = pd.DataFrame(study.best_trial.params, index=[0])
        best_res.to_csv('param/p_' + self.name)

        # best_model_path = self.trainer.checkpoint_callback.best_model_path
        # print(best_model_path)
        #
        # print(vars(self.trainer))
        # print("HERE")
        #
        # self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        pass

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        pass
