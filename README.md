# Clustering-Enhanced Transfer Learning for Cloud Workload Probabilistic Forecast

The transfer learning paradigm has improved the generalisation capabilities of deep learning models, transferring knowledge to unseen (but related) tasks and data distributions. In this paper, we apply it to workload prediction in cloud computing by using pretrained forecasting models to predict the future resource demand of newly available machines in cloud computing cells. We investigate the enhancement of transfer learning capabilities employing clustering algorithms for wisely selecting training data, offering accuracy guarantees when the computational resources are limited. We train Bayesian Neural Networks (BNNs) and state-of-the-art probabilistic models to predict machine-level future resource demand distribution and evaluate them on virtual machines in the Google Cloud data centre. Experiments show that selecting the training data via clustering approaches such as Self Organising Maps allows the model to achieve the same accuracy with lower computational cost compared to a random selection of the data. Moreover, BNNs are able to capture uncertainty aspects that influence the quality of service, which state-of-the-art methods cannot do. %Moreover, state-of-the-art methods achieve high accuracy but cannot capture uncertainty aspects that influence the QoS, like the BNNs. All the considered models achieve prediction time performance suitable for real-world scenarios. 

## Python Dependencies
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* tensorflow-probability    0.14.0
* torch                     1.12.1                 
* torchaudio                0.12.1              
* torchmetrics              0.10.0                   
* torchvision               0.13.1  
* pytorch                   1.12.1      
* pytorch-forecasting       0.10.3                   
* pytorch-lightning         1.7.7  
* optuna                    2.10.1
* minisom                   2.3.0
* tslearn                   0.5.2 
* scikit-learn              1.1.3


## Project Structure
* **cluster**: for each clustering approach, contains the list of the machines in each cluster.
* **hyperparams**: contains for each deep learning model the list of optimal hyperparameters found with Talos and Optuna.
* **lightning_logs**: contains logs from PyTorch training phases
* **models**: contains the definition of the predictive models. One can train the model from scratch using the optimal parameters found with Talos or optuna, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **profiling**: contains measurements of the time for training and inference phases.
* **res**: contains the results of the prediction
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the training phase.
* **talos**: contains the results of the hyperparameter search from Talos
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

#### Train HBNN

For the random version:

```bash
python hbnn_training.py
```

For the clustered-based version:

```bash
python clustered_hbnn_training.py
```

#### Train DeepAR


For the random version:

```bash
python deepar_training.py
```

For the clustered-based version:

```bash
python clustered_deepar_training.py
```

#### Train TFT


For the random version:

```bash
python tft_training.py
```

For the clustered-based version:

```bash
python clustered_tft_training.py
```
