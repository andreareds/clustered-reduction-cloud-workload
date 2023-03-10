"""
Hybrid Bayesian Neural Network model with bigger hyperparameters space
Inherits from ModelProbabilisticDL class
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import talos
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from sklearn.metrics import mean_squared_error
from util import plot_training
import tensorflow_probability as tfp
import os
import pandas as pd
from datetime import datetime

import pickle
from models.model_probabilistic import ModelProbabilistic


class HBNN(ModelProbabilisticDL):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.temp_model = None
        self.parameter_list = {
            'conv_dim': [[16], [32], [64], [128], [32, 16], [64, 16], [64, 32],
                                            [64, 64], [32, 32], [16, 16], [128, 32], [128, 64],
                                            [128, 64, 32], [128, 64, 16], [128, 32, 16], [64, 32, 16],
                                            [128, 64, 64], [128, 32, 32]],
            # 'conv_dim': [[16], [32], [64], [128]],
            'conv_kernel': [3, 5, 7, 11, 13],
            'conv_activation': ['relu', 'tanh'],
            'lstm_dim': [[16], [32], [64], [128], [32, 16], [64, 16], [64, 32],
                         [64, 64], [32, 32], [16, 16], [128, 32], [128, 64],
                         [128, 64, 32], [128, 64, 16], [128, 32, 16], [64, 32, 16],
                         [128, 64, 64], [128, 32, 32]],
            # 'lstm_dim': [[16], [32], [64], [128]],
            'first_dense_dim': [8, 16, 32, 64],
            'first_dense_activation': [keras.activations.relu],
            'mlp_units': [[64, 32], [64, 64], [32, 32], [32, 16], [16, 16]],
            'dense_kernel_init': ['he_normal', 'glorot_uniform'],
            'batch_size': [128, 256, 512, 1024],
            'epochs': [2000],
            'patience': [50],
            'optimizer': ['adam', 'rmsprop'],
            'lr': [1e-2, 1E-3, 1E-4, 1E-5],
            'momentum': [0.9, 0.99],
            'decay': [1E-3, 1E-4, 1E-5],
        }
        """dict: Dictionary of hyperparameters search space"""
        self.p = {'conv_dim': [64],
                  'conv_kernel': 7,
                  'conv_activation': 'relu',
                  'lstm_dim': [32, 32],
                  'first_dense_dim': 32,
                  'first_dense_activation': keras.activations.relu,
                  'mlp_units': [64, 64],
                  'dense_kernel_init': 'he_normal',
                  'batch_size': 1024,
                  'epochs': 2000,
                  'patience': 20,
                  'optimizer': 'adam',
                  'lr': 1E-4,
                  'momentum': 0.99,
                  'decay': 1E-3
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        tf.keras.backend.clear_session()
        input_shape = self.ds.X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        x = Lambda(lambda x: x)(input_tensor)
        for dim in self.p['conv_dim']:
            x = Conv1D(filters=dim, kernel_size=self.p['conv_kernel'],
                       strides=1, padding="causal",
                       activation=self.p['conv_activation'],
                       input_shape=input_shape)(x)

        for dim in self.p['lstm_dim'][:-1]:
            x = LSTM(dim, return_sequences=True)(x)

        x = LSTM(self.p['lstm_dim'][-1])(x)

        for dim in self.p['mlp_units']:
            x = Dense(dim, activation="relu")(x)

        x = self.VarLayer('var', self.p['first_dense_dim'],
                          self.__prior,
                          self.__posterior,
                          1 / self.ds.X_train.shape[0],
                          self.p['first_dense_activation'])(x)

        distribution_params = layers.Dense(units=2 * self.ds.y_train.shape[2])(x)
        outputs = tfp.layers.IndependentNormal(self.ds.y_train.shape[2])(distribution_params)

        self.temp_model = Model(inputs=input_tensor, outputs=outputs)

        if self.p['optimizer'] == 'adam':
            opt = Adam(learning_rate=self.p['lr'], decay=self.p['decay'])
        elif self.p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=self.p['lr'], momentum=self.p['momentum'])
        self.temp_model.compile(loss=self.__negative_loglikelihood,
                                optimizer=opt,
                                metrics=["mse", "mae"])

    def __negative_loglikelihood(self, targets, estimated_distribution):
        """
        Negative log-likelihood custom function
        :param targets: np.array: labels
        :param estimated_distribution: np.array: prediction
        :return:
        """
        return -estimated_distribution.log_prob(targets)

    def __prior(self, kernel_size, bias_size, dtype=None):
        """
        Prior probability distribution function
        :param kernel_size: int: kernel size
        :param bias_size: int: bias size
        :param dtype: data type
        :return: keras model as a multivariate normal distribution of the specified size
        """
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    def __posterior(self, kernel_size, bias_size, dtype=None):
        """
        Posterior probability distribution function
        :param kernel_size: int: kernel size
        :param bias_size: int: bias size
        :param dtype: data type
        :return: keras model as a multivariate normal distribution of the specified size
        """
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n)),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    @tf.keras.utils.register_keras_serializable()
    class VarLayer(tfp.layers.DenseVariational):
        """
        Variational Dense Layer inherits from tfp.layers.DenseVariational
        """

        def __init__(self, name, units, make_prior_fn, make_posterior_fn, kl_weight, activation, **kwargs):
            """
            Constructor of the variational dense layer
            :param name: string: name of the layer
            :param units: int: number of neurons
            :param make_prior_fn: func: priori probability function
            :param make_posterior_fn: func: posteriori probability function
            :param kl_weight: float: kl weight
            :param activation: keras.activation.function: activation function of the layer
            :param kwargs: dict: extra arguments (see tfp.layers.DenseVariational documentation)
            """
            super().__init__(units=units, make_prior_fn=make_prior_fn, make_posterior_fn=make_posterior_fn, name=name,
                             kl_weight=kl_weight, activation=activation, **kwargs)

        def get_config(self):
            """
            configuration of the layer
            :return: None
            """
            config = super(HBNN.VarLayer, self).get_config()
            config.update({
                'name': self.name,
                'units': self.units,
                'activation': self.activation})
            return config

        def call(self, inputs):
            """
            Method necessary for talos implementation to make this class callable
            :param inputs: parameters
            :return:
            """
            return super(HBNN.VarLayer, self).call(inputs)
