import glob
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from models.dl.hbnn import HBNN
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from models.dl.model_interface_dl import ModelInterfaceDL
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from util import dataset, plot_training, save_results, transferdataset, multidataset
from datetime import datetime

win = 288
h = 2
model_name = 'HBNN'
cluster_method = 'som'  # 'vrae' #'tskmeans' #'som'
talos_path = 'talos/tl/' + model_name + '/'
path = './saved_data/machines/'
fulllist = [f for f in os.listdir(path) if f.endswith('.csv')]
cluster_number = pd.read_csv('Cluster/' + cluster_method + '/stats.csv').shape[0]
filenames = []
for i in range(cluster_number):
    df = pd.read_csv('Cluster/' + cluster_method + '/Cluster_' + str(i) + '.csv').Series + '.csv'
    filenames.append(df.values)

print("CLUSTER METHOD:", cluster_method.upper())
print("Number of clusters", cluster_number)

random.seed(10)
runs = 1
training_size = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500, 1000, 1500, 2000]

for size in training_size:
    print("Experiment with", size, "training samples.")
    train_times, inf_times = [], []
    for it in range(runs):
        for j in filenames:
            random.shuffle(j)
        mses, maes = [], []
        trainingnames, testnames = [], []
        min_rep = size // cluster_number
        count = 0
        indexes = [0 for i in range(cluster_number)]
        while count < size:
            for c in range(cluster_number):
                index = count // cluster_number
                try:
                    trainingnames.append(filenames[c][index])
                    indexes[c] += 1
                    count += 1
                    if count == size:
                        break
                except:
                    continue

        for c in range(cluster_number):
            testnames.append(filenames[c][indexes[c]:])
        testnames = list(np.concatenate(testnames))
        for f in fulllist:
            if f not in trainingnames and f not in testnames:
                testnames.append(f)
        print('Dataset in the training set:', len(trainingnames))
        print('Dataset in the test set:', len(testnames))

        experiment_name = model_name + '-' + cluster_method + '-ts-' + str(size) + '-it-' + str(it)

        ds = transferdataset.TransferDatasetInterface(trainingfiles=trainingnames, testfiles=testnames,
                                                      input_window=win, output_window=1,
                                                      horizon=h, training_features=['avgcpu', 'avgmem'],
                                                      target_name=['avgcpu', 'avgmem'], train_split_factor=0.8)

        ds.data_path = path
        ds.dataset_creation()
        ds.dataset_normalization(['minmax', 'minmax'])  # , 'minmax', 'minmax'])
        ds.dataset_shuffle()

        model = HBNN(experiment_name)

        model.talos_path = talos_path
        model.ds = ds
        # model.p = p
        model.create_model()

        training_start = datetime.now()
        model.fit()
        training_time = datetime.now() - training_start
        print("Training complete in ", training_time)
        train_times.append(training_time)

        if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF':
            train_mean = model.evaluate()
            inference_start = datetime.now()
            preds = model.predict(ds.X_test)
            inference_time = (datetime.now() - inference_start) / preds.shape[0]
            print("Inference complete in ", inference_time)
            inf_times.append(inference_time)

            if h > 0:
                preds = preds[:-h]

            if len(ds.target_name) <= 1:
                labels = ds.y_test_array[h:len(preds) + h].reshape(-1, 1)
                train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
            else:
                labels = ds.y_test_array[h:len(preds) + h]
                train_labels = ds.y_train_array[h:len(train_mean) + h]

            print("MSE", mean_squared_error(labels, preds))
            print("MAE", mean_absolute_error(labels, preds))

            plot_training.plot_series(np.arange(0, len(preds)), labels, preds, label1="ground truth",
                                      label2="prediction", title=model.name, bivariate=len(ds.target_name) > 1)

            if len(ds.target_name) <= 1:
                train_mean = np.array(train_mean).reshape(-1, 1)
                train_mean = np.concatenate(train_mean, axis=0)
                # train_labels = np.concatenate(ds.y_train.reshape(-1, 1), axis=0)
                if isinstance(model, ModelInterfaceDL):
                    preds = np.array(preds).reshape(-1, 1)
                    preds = np.concatenate(preds, axis=0)
                    labels = np.concatenate(labels, axis=0)

            save_results.save_output_csv(preds, labels, 'avg', model.name,
                                         bivariate=len(ds.target_name) > 1)
            save_results.save_output_csv(train_mean, train_labels, 'avg', 'train-' + model.name,
                                         bivariate=len(ds.target_name) > 1)

            save_results.save_params_csv(model.p, model.name)

        else:
            inference_start = datetime.now()
            prediction_mean, prediction_std = [], []
            step = ds.X_test.shape[0] // 10000
            print(step)
            for i in range(ds.X_test.shape[0] // 10000):
                try:
                    tmp_prediction_mean, tmp_prediction_std = model.predict(ds.X_test[step * i:step * (i + 1)])
                    prediction_mean.append(tmp_prediction_mean)
                    prediction_std.append(tmp_prediction_std)
                except:
                    if ds.X_test[step * i:].shape[0] != 0:
                        tmp_prediction_mean, tmp_prediction_std = model.predict(ds.X_test[step * i:])
                        prediction_mean.append(tmp_prediction_mean)
                        prediction_std.append(tmp_prediction_std)
            prediction_mean = np.concatenate(prediction_mean)
            prediction_std = np.concatenate(prediction_std)
            inference_time = (datetime.now() - inference_start) / prediction_mean.shape[0]
            print("Inference complete in ", inference_time)
            inf_times.append(inference_time)

            print(prediction_mean.shape, prediction_std.shape)

            if len(ds.target_name) <= 1:
                b = np.concatenate(ds.y_test[:len(prediction_mean)], axis=0).reshape(-1, 1)
                if isinstance(model, ModelProbabilistic) and not isinstance(model, ModelProbabilisticDL):
                    b = ds.y_test_array
            else:
                b = ds.y_test[:len(prediction_mean)]
 
            save_results.save_uncertainty_csv(prediction_mean, prediction_std, b,
                                              'avg',
                                              model.name, bivariate=len(ds.target_name) > 1)


    df_train = pd.DataFrame({'training_time': train_times})
    df_train.to_csv('profiling/train_time' + model.name + '.csv')
    df_inf = pd.DataFrame({'inference_time': inf_times})
    df_inf.to_csv('profiling/inf_time' + model.name + '.csv')
    del model
    del ds
