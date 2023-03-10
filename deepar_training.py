import glob
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import pandas as pd
import warnings
import util.save_results
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.dl.deepar import DeepARModel
from pytorch_forecasting import TimeSeriesDataSet
from util import transferdatasetwrapper
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
win = 288
h = 2
model_name = 'DeepAR'
talos_path = 'talos/tl/' + model_name + '/'

path = './saved_data/machines/'
filenames = [f for f in os.listdir(path) if f.endswith('.csv')]
random.seed(10)
runs = 1
training_size = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500, 1000, 1500, 2000]


for size in training_size:
    train_times, inf_times = [], []
    for it in range(runs):
        mses, maes = [], []
        random.shuffle(filenames)
        trainingnames = filenames[:size]
        testnames = filenames[size:]
        experiment_name = model_name + '-ts-' + str(size) + '-it-' + str(it)

        ds = transferdatasetwrapper.TransferDatasetWrapper(trainingfiles=trainingnames, testfiles=testnames,
                                                           input_window=win, output_window=1,
                                                           horizon=h, training_features=['avgcpu', 'avgmem'],
                                                           target_name=['avgcpu', 'avgmem'], train_split_factor=0.8)

        ds.data_path = path
        ds.dataset_creation(['minmax', 'minmax'])

        model = DeepARModel(experiment_name)
        model.verbose = 1
        model.ds = ds
        model.create_model()
        model.talos_path = talos_path
        model.ds.loader_creation(model.p['batch_size'])

        training_start = datetime.now()
        model.fit()
        training_time = datetime.now() - training_start
        print("Training complete in ", training_time)
        train_times.append(training_time)
    
        inference_start = datetime.now()
        preds = model.predict(model.ds.test_dataloader)
        inference_time = (datetime.now() - inference_start) / preds[0].shape[0]
        print("Inference complete in ", inference_time)
        inf_times.append(inference_time)

        util.save_results.save_quantile_csv(preds[0][-ds.X_test_df.shape[0]:, 2, :],
                                            preds[1][-ds.X_test_df.shape[0]:, 2, :],
                                            ds.X_test_df.values[:, 0],
                                            ds.X_test_df.values[:, 1],
                                            model.name)
    df_train = pd.DataFrame({'training_time': train_times})
    df_train.to_csv('profiling/train_time' + model.name + '.csv')
    df_inf = pd.DataFrame({'inference_time': inf_times})
    df_inf.to_csv('profiling/inf_time' + model.name + '.csv')

# %%
