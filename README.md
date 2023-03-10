# Clustering-Enhanced Transfer Learning for Cloud Workload Probabilistic Forecast

The transfer learning paradigm has improved the generalisation capabilities of deep learning models, transferring knowledge to unseen (but related) tasks and data distributions. In this paper, we apply it to workload prediction in cloud computing by using pretrained forecasting models to predict the future resource demand of newly available machines in cloud computing cells. We investigate the enhancement of transfer learning capabilities employing clustering algorithms for wisely selecting training data, offering accuracy guarantees when the computational resources are limited. We train Bayesian Neural Networks (BNNs) and state-of-the-art probabilistic models to predict machine-level future resource demand distribution and evaluate them on virtual machines in the Google Cloud data centre. Experiments show that selecting the training data via clustering approaches such as Self Organising Maps allows the model to achieve the same accuracy with lower computational cost compared to a random selection of the data. Moreover, BNNs are able to capture uncertainty aspects that influence the quality of service, which state-of-the-art methods cannot do. %Moreover, state-of-the-art methods achieve high accuracy but cannot capture uncertainty aspects that influence the QoS, like the BNNs. All the considered models achieve prediction time performance suitable for real-world scenarios. 
