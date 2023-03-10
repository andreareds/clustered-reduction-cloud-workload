### Code inspired by https://github.com/tejaslodaya/timeseries-clustering-vae

from vrae.vrae import VRAE
import numpy as np
import torch
import pickle

# import plotly
from torch.utils.data import TensorDataset

import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import os
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dload = './model_dir'  # download directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

LIB_PATH = "../../BigQuery/preprocessed/"
# LIB_PATH = "../dataset/preprocessed/"

files = glob.glob(LIB_PATH + "*.csv")
print("Datasets number:", len(files))

mySeries = []
namesofMySeries = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    split_factor = int(0.8 * df.shape[0])
    df = df[['avgcpu', 'avgmem']].iloc[:split_factor]
    mySeries.append(df.values)
    namesofMySeries.append(f[:-4])

for i in range(len(mySeries)):
    scaler = MinMaxScaler()
    mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])

X_train = np.array(mySeries)  # , axis=1)
# X_train = X_train[:, :, np.newaxis]
print(X_train.shape)

hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 128
learning_rate = 0.0005
n_epochs = 500
dropout_rate = 0.2
optimizer = 'Adam'  # options: ADAM, SGD
cuda = True  # options: True, False
print_every = 30
clip = True  # options: True, False
max_grad_norm = 5
loss = 'MSELoss'  # options: SmoothL1Loss, MSELoss
block = 'LSTM'  # options: LSTM, GRU

sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]

X_train = X_train

train_dataset = TensorDataset(torch.from_numpy(X_train).to(device))

vrae = VRAE(sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=hidden_size,
            hidden_layer_depth=hidden_layer_depth,
            latent_length=latent_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            cuda=cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss=loss,
            block=block).to(device)

vrae.fit(train_dataset, save=True)