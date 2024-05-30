#################################################################
# Import libraries
#################################################################
import random
import string

# Modelling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import graphviz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import sys
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import PegasosQSVC, VQC
# from qboost.qboost import QBoostClassifier
# from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from dwave.system.composites import EmbeddingComposite
import time
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics

# import psycopg2
# from psycopg2.extras import RealDictCursor
# from imblearn.under_sampling import RandomUnderSampler
from timeit import default_timer as timer
from datetime import *
from collections import Counter
# from connect import connect

# Garbage collection
import gc

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from numpy import mean

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


def loadData(path):
    data = pd.read_csv(path)
    # print(data)
    return data


def trainPQSVC(X, y, data_dimension):
    from qiskit.circuit.library import ZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    print('Training PegaSOS QSVC Model... ')
    # Number of qubits is equal to the number of features
    num_qubits = data_dimension  # Test with high data dimension to check if this works
    # Number of steps performed during the training procedure
    tau = 100  # Test with other number of steps to check difference in data
    # Regularization parameter
    C = 1000

    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # algorithm_globals.random_seed = 12345  # Already declared in the global variables
    # Define the Pegasos QSVC classifier
    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
    # training - fit classifier to data
    start = timer()
    pegasos_qsvc.fit(X, y)
    end = timer()
    train_time = end - start

    return pegasos_qsvc, train_time


# def callback_graph(weights, obj_func_eval):
#     clear_output(wait=True)
#     objective_func_vals.append(obj_func_eval)
#     plt.title("Objective function value against iteration")
#     plt.xlabel("Iteration")
#     plt.ylabel("Objective function value")
#     plt.plot(range(len(objective_func_vals)), objective_func_vals)
#     plt.show()

# Declare VQC
def vqc_nat(feature_map, ansatz):
    from qiskit.algorithms.optimizers import COBYLA

    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=30),
        # callback=callback_graph,
    )


# Training VQC
def trainVQC(X, y, data_dimension):
    print('Training VQC Model... ')
    feature_map = ZZFeatureMap(data_dimension, reps=1)
    ansatz = RealAmplitudes(data_dimension, reps=3)
    vqc = vqc_nat(feature_map, ansatz)
    start = timer()
    # fit classifier to data
    vqc.fit(X, y)
    end = timer()
    train_time = end - start
    return vqc, train_time


def predictModel(model, x):
    print('Prediction on model :', model)
    start = timer()
    predictions = model.predict(x)
    end = timer()
    predict_time = end - start
    return predictions, predict_time


def reportResult(y, predictions, model, predict_type, train_time, predict_time):
    print("Reporting Result ...")
    result_time = datetime.now().strftime("%Y%m%d%H%M%S")

    print(result_time)
    report = classification_report(y, predictions)

    print(report)

    with open('/mnt/harddisk/devincentiis/smartcity/main_project/results/' + str(result_time) + '_' + str(
            model) + '_' + str(predict_type) + '_report.txt', 'a') as f:
        f.write('Training time :' + str(train_time) + '\n')
        f.write(report)
        f.write('Prediction time :' + str(predict_time) + '\n')
    f.close()
    print("Saved...")


def main():
    path = '/mnt/harddisk/devincentiis/smartcity/main_project/data/train70_reduced.csv'
    tpath = '/mnt/harddisk/devincentiis/smartcity/main_project/data/test30_reduced.csv'

    dataframe = loadData(path)

    df1 = loadData(tpath)

    df = pd.concat([dataframe, df1])

    non_numeric_columns = df.select_dtypes(exclude=[int, float]).columns.tolist()

    non_numeric_columns.remove('target')

    print(non_numeric_columns)

    df = df.drop(columns=non_numeric_columns)

    df['target'] = df['target'].replace('legitimate', -1).replace('dos', 1).replace('malformed', 1).replace(
        'bruteforce', 1).replace('slowite', 1).replace('flood', 1)

    columns_with_same_val = df.columns[(df == df.iloc[0]).all()]

    df = df.drop(columns=columns_with_same_val)

    shp = df.shape
    cols = list(df.columns.values)

    X = df[cols[0:shp[1] - 1]]

    y = df['target']

    print(X.head())
    print(y.head())

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    print(scaled_X)

    pca = PCA()
    pca.fit(scaled_X)

    n_components = 6  # Choose the desired number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(scaled_X)

    y_pca = y.values.ravel()

    print(X_pca)
    print(y_pca)

    x_train, x_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size=0.3, shuffle=True, random_state=32)

    # data_dimension = X.shape[1]
    data_dimension = 6

    # model, train_time = trainQboost(x_train, y_train)
    # model, train_time = trainPQSVC(x_train, y_train, data_dimension)
    model, train_time = trainVQC(x_train, y_train, data_dimension)

    predictions, predict_time = predictModel(model, x_test)

    reportResult(y_test, predictions, str(model)[1:6], 'training', train_time, predict_time)


if __name__ == '__main__':
    main()
