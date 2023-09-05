
#### to accelaerate the classifier. could be commented out
from sklearnex import patch_sklearn 
patch_sklearn()
#############################################

from map_evaluation import P_wrapper, Evaluator
import os


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



import datetime 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), 
              "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



data_dir = './data'
data_dirs = [
    'har', 
    'mnist', 
    'fashionmnist', 
     'reuters', 
     ]
datasets_real = {}

for d in data_dirs:
    dataset_name = d

    X = np.load(os.path.join(data_dir, d,  'X.npy'))
    y = np.load(os.path.join(data_dir, d, 'y.npy'))
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    train_size = min(int(n_samples*0.9), 100)
    test_size = 500 # inverse
    
    dataset =\
        train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
    datasets_real[dataset_name] = dataset

    ## clip dataset[1] and dataset[3] to test_size if they are larger
    if dataset[1].shape[0] > test_size:
        dataset[1] = dataset[1][:test_size]
        dataset[3] = dataset[3][:test_size]
        
classifiers = {
    'Logistic Regression': linear_model.LogisticRegression(n_jobs=-1),
    'Random Forests': RandomForestClassifier(n_jobs=-1, random_state=999), 
    'Neural Network': MLPClassifier([200,200, 200], random_state=999) ,  
    'SVM': SVC(probability=True),
}


DMs = {
            'SSNP' : P_wrapper(ssnp=1),
            'DBM': P_wrapper(NNinv_Keras=1),
            'DeepView': P_wrapper(deepview=1),       
            }

evaluater = Evaluator()


date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

### PLEASE CHECK THE SAVE NAME BEFORE RUNNING!
save_name = 'global_metrics' + date + '.csv'
load_data = False
data_to_evaluate = datasets_real
data_path = 'lolcal_metrics_results'


res = evaluater.evaluate_all(classifiers, DMs, data_to_evaluate, read_from_file=load_data, save_name=save_name, save_path=data_path)