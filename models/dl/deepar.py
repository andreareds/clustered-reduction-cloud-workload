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
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss, MultivariateNormalDistributionLoss
import optuna


# modify n_trials and self.p['epochs']

class DeepARModel(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Probabilistic class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.temp_model = None
        self.parameter_list = {'hidden_size': [32, 64, 128, 256, 512],
                               'dropout': [0.1, 0.2, 0.3],
                               'rnn_layers': [1, 2, 3, 4],
                               'lr': [0.001, 0.01, 0.1],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               }

        """dict: Dictionary of hyperparameters search space"""
        self.p = {'hidden_size': 32,
                  'dropout': 0.3,
                  'rnn_layers': 1,
                  'lr': 0.01,
                  'epochs': 2000,
                  'patience': 50,
                  'batch_size': 256
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""
        self.trainer = None
        self.talos_path = 'talos/tl/DeepAR/'

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
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            logger=logger,
            default_root_dir=self.talos_path
        )

        if len(self.ds.target_name) == 1:
            self.model = DeepAR.from_dataset(
                self.ds.X_train,
                learning_rate=self.p['lr'],
                hidden_size=self.p['hidden_size'],
                rnn_layers=self.p['rnn_layers'],
                dropout=self.p['dropout'],
                reduce_on_plateau_patience=4,
            )
        else:
            self.model = DeepAR.from_dataset(
                self.ds.X_train,
                learning_rate=self.p['lr'],
                hidden_size=self.p['hidden_size'],
                rnn_layers=self.p['rnn_layers'],
                dropout=self.p['dropout'],
                # loss=(MultivariateNormalDistributionLoss(rank=30), MultivariateNormalDistributionLoss(rank=30)),
                reduce_on_plateau_patience=4,
            )

        # if self.verbose == 1:
        #     print(f"Number of parameters in network: {self.temp_model.size() / 1e3:.1f}k")

    def hyperparametrization(self):
        """
        Search the best parameter configuration using talos
        :return: None
        """

        study = optuna.create_study(study_name=self.name, direction="minimize",
                                    sampler=optuna.samplers.GridSampler(self.parameter_list))
        study.optimize(self.__optuna_objective, n_trials=10000, show_progress_bar=True)
        print('\nBEST PARAMS: \n{}'.format(study.best_params))
        print('\nBEST VALUE:\n{}'.format(study.best_value))
        df = study.trials_dataframe()
        df.to_csv(self.talos_path + self.name + '.csv')

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
        # self.model = DeepAR.load_from_checkpoint(best_model_path)

    def __optuna_objective(self, trial):
        """
        Custom Function of the Optuna which represented a single execution of a trial
        :param dict trial: Represents the hypermeters being examined
        :return float mse: MeanSquaredError of the Model's Predictive Performance on the Validation Set
        """
        rnn_layers = trial.suggest_categorical('rnn_layers', self.parameter_list['rnn_layers'])
        hidden_size = trial.suggest_categorical('hidden_size', self.parameter_list['hidden_size'])
        dropout = trial.suggest_categorical('dropout', self.parameter_list['dropout'])
        learning_rate = trial.suggest_categorical('lr', self.parameter_list['lr'])
        # optimizer = trial.suggest_categorical('optimizer', self.parameter_list['optimizer'])
        print('Trial Params: {}'.format(trial.params))

        if len(self.ds.target_name) == 1:
            self.model = DeepAR.from_dataset(
                self.ds.X_train,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                rnn_layers=rnn_layers,
                dropout=dropout,
                reduce_on_plateau_patience=4,
            )
        else:
            self.model = DeepAR.from_dataset(
                self.ds.X_train,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                rnn_layers=rnn_layers,
                dropout=dropout,
                reduce_on_plateau_patience=4,
            )

        # self.model.optimizer_cls = opt
        self.model.optimizer_kwargs = {"lr": learning_rate}
        self.model.lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

        preds = self.fit_predict(self.ds.train_dataloader)

        val_preds = self.predict(self.ds.val_dataloader)
        actuals = [y[0] for x, y in iter(self.ds.val_dataloader)][0]

        if len(self.ds.target_name) == 1:
            return torch.mean(self.model.loss.loss(val_preds, actuals))
        else:
            return torch.mean(torch.cat(self.model.loss.loss(val_preds, actuals)))

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
