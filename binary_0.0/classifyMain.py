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
from sklearn.tree import export_graphviz
from IPython.display import Image
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
from qboost.qboost import QBoostClassifier
from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dwave.system.composites import EmbeddingComposite
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC

# import psycopg2
# from psycopg2.extras import RealDictCursor
from imblearn.under_sampling import RandomUnderSampler
from timeit import default_timer as timer
from datetime import *
from collections import Counter
from connect import connect

from qiskit import BasicAer
from sklearn.datasets import make_blobs

# from qmeans import qkmeans
# from qiskit import IBMQ
from qiskit import Aer

from qiskit_ibm_provider import IBMProvider

################################################################
# Import Pickle and Joblist to test Serialization of Model
################################################################
import pickle
import joblib

#################################################################
# QBOOST classification parameters
#################################################################


DW_PARAMS = {'num_reads': 3000,
             'auto_scale': True,
             # "answer_mode": "histogram",
             'num_spin_reversal_transforms': 10,
             # 'annealing_time': 10,
             # 'postprocess': 'optimization',
             }
NUM_WEAK_CLASSIFIERS = 35
TREE_DEPTH = 3
dwave_sampler = DWaveSampler(token="DEV-98f903479d1e03bc59d7ba92376a492f76f7c906")
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.5

# End Parameters

#################################################################
# QISKIT classification parameters
#################################################################

# IBMQ.save_account('806f68c7762a09f1e982010af9627f8f12e51d7016f9a20ed72584b673811a1a'
#                  'e268d2b3398bfc6a5517f8a9385d920ec3aae5ba0a1979aaa9fe2dc1ef97e89a')
# IBMQ.load_account()
# backend = Aer.get_backend('aer_simulator')

# New way v2 to load account and access backend
provider = IBMProvider()
backend = Aer.get_backend('aer_simulator')


# End Parameters


#################################################################
# Load csv into memory
#################################################################


def loadData(path):
    data = pd.read_csv(path)
    # print(data)
    return data


#################################################################
# Analyze dataset (find out if the dataset needs to be balanced)
# Find out the number of 1 and 0 labels
#################################################################


def analyzeData(data):
    #  Number of label 1's
    zeros = data['label'].value_counts()[0]
    ones = data['label'].value_counts()[1]
    print('Data analysis...')
    print('zeros :' + str(zeros))
    print('ones :' + str(ones))
    return zeros, ones


#################################################################
# Select the features of the dataset (independent features)
#################################################################


def featureList(data):
    shape = data.shape
    cols = list(data.columns.values)
    # print(shape[0], shape[1], cols)
    alist = cols[0:shape[1]]
    # Remove date and time from training features (cols starts from index 2)
    flist = cols[2:shape[1] - 2]
    # cols.pop(-1)
    label = [cols[-2]]
    print(alist, flist, label)
    return alist, flist, label


#################################################################
# Data scaling
#################################################################


def scaleData(data, flist):
    # Discretize high and low values in Iot_Weather.csv
    #print(data.shape[1])

    # define the min-max scaler
    scaler = MinMaxScaler()

    # Check the shape of data; and process as per the shape; if full feature data
    # then just perform full scaling, else just plain transform
    if data.shape[1] > 1:
        #print(data)
        data['time'] = data['time'].str.replace(':', '')
        data['time'] = [int(x) for x in data['time']]
        data['date'] = data['date'].str.replace("-", "")
        # Change the months to numbers
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_num = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        for i in range(12):  # "i" is index to access month and month_num
            data['date'] = data['date'].str.replace(month[i], month_num[i])
        # Scale the data to be within with 0 and 1
        #print(data)
        data[flist] = scaler.fit_transform(data[flist])
        return data
    else:
        data[flist] = scaler.fit_transform(data[flist])
        return data


#################################################################
#  Data Transformation (clean, transform, and other operations on data)
#################################################################


"""def transformData(xtrain, xtest, ytrain, ytest, flist, cols):
    # print(alldata)
    # print(label)
    # label = data[flist[-1:]].replace(0, -1)  # Change the 0 label to -1 for Qboost algorithm
    # label = np.where(data[flist[-1:]], 1, -1)  # Change the 0 label to -1 for Qboost algorithm
    # label = label.replace(0, -1)  # Change the 0 label to -1 for Qboost algorithm
    # x_train, x_test, y_train, y_test = train_test_split(alldata, label, test_size=0.3, shuffle=True, random_state=23)
    print('x_train :', len(x_train))
    print('y_train :', len(y_train))
    print('x_test :', len(x_test))
    print('ytest :', len(y_test))
    x_train, y_train = downSample(x_train, y_train)
    x_test, y_test = downSample(x_test, y_test)
    return x_train, y_train, x_test, y_test"""


#################################################################
#  SVC model
#################################################################


def trainSVC(x, y):
    print('Training SVC Model ... ')
    svc = SVC(kernel='linear', C=1.0)
    start = timer()
    svc.fit(x, y)
    end = timer()
    train_time = end - start
    print('SVC training time in seconds :', end - start)
    return svc, train_time

#################################################################
#  RF model
#################################################################


def trainRF(x, y):
    print('Training Random Forest Model ... ')

    # param_dist = {'n_estimators': randint(1, 500), 'max_depth': randint(1, 20)}

    # Create a random forest classifier
    # rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    """rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=5,
                                     cv=5)"""

    # Fit the random search object to the data
    # rand_search.fit(x, y)

    # Create a variable for the best model
    # best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    # print('Best hyperparameters:', rand_search.best_params_)
    print('Best parameters based on 5-fold cross validation = max_depth: 9, '
          'n_estimators: 381')

    rf = RandomForestClassifier(max_depth=9, n_estimators=381)

    start = timer()
    rf.fit(x, y)
    end = timer()
    train_time = end - start
    print('RF training time in seconds :', end - start)
    return rf, train_time


#################################################################
#  Adaboost model
#################################################################


def trainAdaboost(x, y):
    print('Training Adaboost Model ... ')

    svc = SVC(probability=True, kernel='linear')
    abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)
    start = timer()
    abc.fit(x, y)
    end = timer()
    train_time = end - start
    print('Adaboost training time in seconds :', end - start)
    return abc, train_time


#################################################################
#  Extra Tree model
#################################################################


def trainET(x, y):
    print('Training Extratree Model ... ')

    etc = ExtraTreesClassifier(n_estimators=100, random_state=15)
    start = timer()
    etc.fit(x, y)
    end = timer()
    train_time = end - start
    print('Extra Tree training time in seconds :', end - start)
    return etc, train_time


#################################################################
#  QBoost model
#################################################################


def trainQboost(x, y):
    print('Training QBOOST Model... ')
    qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    start = timer()
    qboost.fit(x, y, emb_sampler, lmd=lmd, **DW_PARAMS)
    end = timer()
    train_time = end - start
    print('QBoost training time in seconds :', train_time)
    return qboost, train_time


#################################################################
#  PegaSOS data Preparation
#################################################################


def dataPrepPegasos():
    # START Data LOAD from real dataset
    path = '../data/percent2.csv'
    data = loadData(path)

    # Extract the feature names and label name
    alist, flist, label = featureList(data)
    # Dimension of data
    # data_dimension = len(flist)
    # Separate the dependent and independent features
    independent_data = data[flist]
    # print(independent_data)
    dependent_data = data[label]
    dependent_data = dependent_data.replace(0, -1)  # change 0 to -1 for quantum algorithm
    # print(dependent_data)

    independent_data = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(independent_data)

    # print(independent_data, dependent_data)
    # exit(0)

    x_train_down, y_train_down = downSample(independent_data, dependent_data)

    # Split data for machine learning
    x_train, x_test, y_train, y_test = train_test_split(x_train_down, y_train_down, test_size=0.2,
                                                        shuffle=True, random_state=23)

    # Prepare data for model fitting
    # x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
    #    prepData(x_train, x_test, y_train, y_test, flist, label)

    #x_train_scale_flatten = x_train[flist].values
    #x_test_scale_flatten = x_test[flist].values
    y_train_flatten = y_train.values.ravel()
    y_test_flatten = y_test.values.ravel()
    print(x_train)
    print(y_train_flatten)
    print(x_test)
    print(y_test_flatten)

    # print('Down-sampling the data ...')
    # x_train_down, y_train_down = downSample(x_train_scale_flatten, y_train_scale_flatten)
    # x_test_down, y_test_down = downSample(x_test_scale_flatten, y_test_scale_flatten)
    # exit(0)

    return x_train, x_test, y_train_flatten, y_test_flatten


#################################################################
#  PegaSOS QSVC Model
#################################################################


def trainPQSVC():

    #xtrainf, xtestf, ytrainf, ytestf = dataPrepPegasos()
    #print('Data Description:')
    #print(xtrainf.shape, xtestf.shape, ytrainf.shape, ytestf.shape)

    # example dataset
    print('Create Data')
    X, y = make_blobs(n_samples=65000, n_features=20, centers=2, random_state=32, shuffle=True)
    print(X, y)

    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    print(X, y)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, train_size=0.7, shuffle=False
    )
    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

    # number of qubits is equal to the number of features
    num_qubits = X.shape[1]
    print('Num Features: ', X.shape[1])

    # number of steps performed during the training procedure
    tau = 100

    # regularization parameter
    C = 1000

    algorithm_globals.random_seed = 43

    # The block of code mentioned below is adapted from Medium
    # https://medium.com/@raijeku/how-to-use-quantum-machine-learning-in-your-ai-projects-80720d547c96

    # Pegasos backend parameter
    # pegasos_backend = QuantumInstance(
    #     BasicAer.get_backend("statevector_simulator"),
    #     seed_simulator=algorithm_globals.random_seed,
    #     seed_transpiler=algorithm_globals.random_seed,
    # )
    # feature_map = ZZFeatureMap(feature_dimension=xtrainf.shape[1], reps=1, entanglement='full')
    # qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=pegasos_backend)
    # pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel)

    # MEDIUM block ends here

    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
    # exit(0)

    print('Pegasos Training ...')
    # training
    start = timer()
    pegasos_qsvc.fit(train_features, train_labels)
    end = timer()
    print('Training time: %3f' % (end - start))

    print('Pegasos Testing ...')
    # testing
    start = timer()
    predictions = pegasos_qsvc.predict(test_features)
    end = timer()
    # pegasos_score = pegasos_qsvc.score(xtestf, ytestf)
    print(classification_report(test_labels, predictions))
    print('Prediction time: %3f' % (end - start))

    # print(pegasos_score)

    exit(0)


#################################################################
#  SVC model prediction
#################################################################


def predictModel(model, x):
    print('Prediction on model :', model)
    start = timer()
    predictions = model.predict(x)
    end = timer()
    predict_time = end - start
    return predictions, predict_time


#################################################################
#  Create confusion matrix
#################################################################
def makeConfMatrix(y_test, prediction):
    conf_matrix = metrics.confusion_matrix(y_test, prediction)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.show()
    plt.savefig('../results/conf_matrix.png')

#################################################################
#  Save the model
#################################################################


def saveModel(model, name):
    # -- Save the model using Pickle --
    # model_p = 'model.pkl'
    # pickle.dump(model, open(model_p, 'wb'))
    # -- Save the model using Joblib --
    model_file = name + '.pkl'
    joblib.dump(model, model_file)
    print('Model Saved :', model)


#################################################################
#  Load the model
#################################################################


def loadModel(model_file):
    # -- Loading Model from Pickle ----
    # modelQboost = pickle.load(open(model_file, 'rb'))
    # -- Loading Model from Joblib ----
    qboost = joblib.load(model_file)
    print(qboost)
    return qboost


#################################################################
#  Report Result
#################################################################


def reportResult(y, predictions, model, predict_type, train_time, predict_time):
    print("Reporting Result ...")
    result_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report = classification_report(y, predictions)
    with open('../results/' + str(result_time) + '_' + str(model) + '_' + str(predict_type) + '_report.txt', 'a') as f:
        f.write('Training time :' + str(train_time) + '\n')
        f.write(report)
        f.write('Prediction time :' + str(predict_time) + '\n')
    f.close()
    print("Saved...")


#################################################################
#  Report Results for Model Load
#################################################################


def updateDB(values):
    # Define the SQL queries here
    sql = "insert into scmobility.iotweather(attackid, attackdate, attacktime, severity, " \
          "categories, attacktype) values(%s, %s, %s, %s, %s, %s);"
    connect(sql, values)
    return 0


#################################################################
#  Report Results for Model Load
#################################################################
# def repLoadModelResult()
#    # --------------------------------------------------------
#    # Edit the data to remove all but 1 row.
#    # Just pass 1 record to the model
#    # --------------------------------------------------------
#    data = data[0].reshape(1, -1)
#    print(data)
#    # break
#
#    print("Data mining Quantum in corso...")
#    start = time.time()
#    # quantum_predicted = loadModelQboostPickle.predict(data)
#    quantum_predicted = loadModelQboostJoblib.predict(data)
#    print("Completato")
#    end = time.time()
#    print("Tempo impiegato Quantum: ", end - start)
#    # tempo_quantum = end - start
#    # print(tempo_quantum)
#    with np.printoptions(threshold=np.inf):
#        print("Prediction Val:", quantum_predicted)


#################################################################
# Down-Sample to balance
#################################################################


def downSample(x, y):
    # Reduce the data size to match the minority class
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=32)
    # fit and apply the Under Sampling Transform
    x_under, y_under = undersample.fit_resample(x, y)
    # a, b = analyzeData(x_under)
    # c, d = analyzeData(y_under)
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    return x_under, y_under


#################################################################
# Create model name to be used in the report
#################################################################


# def findModelName(model):
#    for part in model:
#        if part.isalnum():
#            modname = part
#            break
#        else:
#            modname = str(model)
#    return modname


#################################################################
# Over-Sample to balance
#################################################################


def overSample(x, y):
    return x, y

#################################################################
# Delete rows from Predictions where they are NOT ATTACK
#################################################################


def delRow(data, num, col):
    return data[data[:, col] != num, :]

#################################################################
# Update rows from Predictions where they are ATTACK
#################################################################


def updateRow(data, col):
    data[col] = 'Attack State'
    return 0

#################################################################
# Generate a random alphanumeric string in Python
#################################################################


def random_alphanumeric_string(length):
    return ''.join(
        random.choices(
            string.ascii_letters + string.digits,
            k=length
        )
    )

#################################################################
# Create the primary key for insertion into the database
#################################################################


def createKey(res):
    key = []
    if len(res.shape) == 1:
        date_object = datetime.strptime(res[0].strip(), '%d-%b-%y').date()
        date_str = date_object.strftime("%y%m%d")
        time_object = datetime.strptime(res[1].strip(), '%H:%M:%S').time()
        time_str = time_object.strftime("%H%M%S")
        key = date_str.strip() + time_str.strip() + random_alphanumeric_string(5).strip()
    else:
        for data in res:
            date_object = datetime.strptime(data[0].strip(), '%d-%b-%y').date()
            date_str = date_object.strftime("%y%m%d")
            time_object = datetime.strptime(data[1].strip(), '%H:%M:%S').time()
            time_str = time_object.strftime("%H%M%S")
            keyval = date_str.strip() + time_str.strip() + random_alphanumeric_string(5).strip()
            key = np.append(key, keyval)
    # print(key)
    # exit(0)
    return key

#################################################################
# Create model name to be used in the report
#################################################################


def prepData(*data):
    if len(data) > 3:
        """ Map data from multiple arguments to variables """
        x_train = data[0]
        x_test = data[1]
        y_train = data[2]
        y_test = data[3]
        flist = data[4]
        label = data[5]

        # Scale data for the training data
        print('Scaling Data ...')
        x_train_scale = scaleData(x_train, flist)
        x_test_scale = scaleData(x_test, flist)
        y_train_scale = scaleData(y_train, label)
        y_test_scale = scaleData(y_test, label)

        # Replace 0 with -1 in labels and flatten the values into arrays
        print('Replacing 0 with -1; Qboost requirement ...')
        y_train_scale = y_train_scale.replace(0, -1)
        y_test_scale = y_test_scale.replace(0, -1)
        x_train_scale_flatten = x_train_scale[flist].values
        x_test_scale_flatten = x_test_scale[flist].values
        y_train_scale_flatten = y_train_scale.values.ravel()
        y_test_scale_flatten = y_test_scale.values.ravel()
        # print('x_train :', len(x_train))
        # print('y_train :', len(y_train))
        # print('x_test :', len(x_test))
        # print('ytest :', len(y_test))

        print('Down-sampling the data ...')
        x_train_down, y_train_down = downSample(x_train_scale_flatten, y_train_scale_flatten)
        x_test_down, y_test_down = downSample(x_test_scale_flatten, y_test_scale_flatten)
        return x_train_down, x_test_down, y_train_down, y_test_down
    else:
        print('Preparing data for Prediction...')
        x_test = data[0]
        print('Count of Pred :', len(x_test))

        # Find length of test data
        test_data_length = len(x_test)
        # print(test_data_length)

        # print(x_test)
        # Columns of the dataframe
        cols = list(x_test.columns.values)

        #  Using the whole dataset to help in scaling
        main_data = data[1]
        main_data = main_data[cols]

        # Appending the test data in front of the main dataset
        # x_test = x_test.append(main_data, ignore_index=True)
        x_test = pd.concat([x_test, main_data], ignore_index=True)

        # Scale data for the test data
        print('Scaling Data ...')
        x_test_scale = scaleData(x_test, cols)
        print(len(x_test_scale))

        # Select the scaled test data from the whole dataset
        x_test_scale = x_test_scale.head(test_data_length)
        # print(x_test_scale)

        # Flatten the test values into arrays
        x_test_scale_flatten = x_test_scale.values

        # print(x_test_scale_flatten)
        # print(len(x_test_scale_flatten))
        # exit(0)
        return x_test_scale_flatten


def makeData():
    # START Data LOAD from real dataset
    path = '../data/IoT_Weather.csv'
    data = loadData(path)

    # Extract the feature names and label name
    alist, flist, label = featureList(data)
    # Dimension of data
    data_dimension = len(flist)
    # Separate the dependent and independent features
    independent_data = data[alist]
    # print(independent_data)
    dependent_data = data[label]
    dependent_data = dependent_data.replace(0, -1)  # change 0 to -1 to fir to quantum algorithm
    # print(dependent_data)

    # Split data for machine learning
    x_train, x_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.2,
                                                        shuffle=True, random_state=23)

    # Prepare data for model fitting
    x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
        prepData(x_train, x_test, y_train, y_test, flist, label)

    return x_train_prep, x_test_prep, y_train_prep, y_test_prep, data_dimension


#################################################################
# Main function
#################################################################


def main():
    args = sys.argv[1:]
    print(args)
    location = '../data/IoT_Weather.csv'
    # location = '../data/percent18.csv'
    category = Path(location).stem
    # Train models or predict from saved models
    if args[0] == 'training':
        if args[1] == 'pqsvc':
            trainPQSVC()
        predict_type = args[0]
        #  dataset path
        # path = '../data/IoT_Weather.csv'
        path = '../data/percent18.csv'
        data = loadData(path)

        # Data analysis
        zeros, ones = analyzeData(data)
        # print('zeros :', zeros)
        # print('ones :', ones)

        # Extract the feature names and label name
        alist, flist, label = featureList(data)
        # Separate the dependent and independent features
        independent_data = data[alist]
        # print(independent_data)
        dependent_data = data[label]
        dependent_data = dependent_data.replace(0, -1)  # change 0 to -1 to fir to quantum algorithm
        # print(dependent_data)

        # Split data for machine learning
        x_train, x_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.2,
                                                            shuffle=True, random_state=23)
        # print(x_train)
        # print(x_test)
        # print(y_train)
        # print(y_test)

        # Prepare data for model fitting
        print('Preparing data for Training & Testing...')
        x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            prepData(x_train, x_test, y_train, y_test, flist, label)
        print('Data preparation complete.')

        # print(x_train_prep)
        # print(x_test_prep)
        # print(y_train_prep)
        # print(y_test_prep)

        # Training SVC model with training dataset
        # model, train_time = trainSVC(x_train_prep, y_train_prep)
        # saveModel(model)
        # predictions, predict_time = predictModel(svc, x_train_prep)
        # reportResult(y_train_prep, predictions, model[0:3], predict_type, train_time, predict_time)

        # Training qboost model with training dataset
        print('Training QBoost model on the 5% dataset...')
        model, train_time = trainQboost(x_train_prep, y_train_prep)
        # model, train_time = trainSVC(x_train_prep, y_train_prep)
        # model, train_time = trainRF(x_train_prep, y_train_prep)
        # model, train_time = trainAdaboost(x_train_prep, y_train_prep)
        # model, train_time = trainET(x_train_prep, y_train_prep)
        print('Model Training complete.')

        # Saving model
        saveModel(model, 'qboost')

        # SVC Testing
        # predictions, predict_time = predictModel(svc, x_train_prep)
        # reportResult(y_train_prep, predictions, 'svc', predict_type, train_time, predict_time)

        # Model testing with training dataset
        print('Testing model for training prediction values...')
        predictions, predict_time = predictModel(model, x_train_prep)
        print('Prediction phase complete.')
        reportResult(y_train_prep, predictions, str(model)[1:4], predict_type, train_time, predict_time)

        # Model testing with Training and Testing dataset
        predict_type = 'test'  # testing in default settings

        # SVC Testing
        # predictions, predict_time = predictModel(svc, x_test_prep)
        # reportResult(y_test, predictions, 'svc', predict_type, train_time, predict_time)

        # QBOOST Testing
        print('Testing model for TEST prediction values...')
        predictions, predict_time = predictModel(model, x_test_prep)
        print('TEST Prediction phase complete.')
        reportResult(y_test_prep, predictions, str(model)[1:4], predict_type, train_time, predict_time)
        print('Results reporting complete.')
        makeConfMatrix(y_test_prep, predictions)
    # Predict from values: D
    # Datasets of the form (<date>, <time>, <temperature>, <pressure>, <humidity>)
    elif args[0] == 'prediction':
        predict_type = args[0]

        # Load Main Dataset for scaling
        path = '../data/IoT_Weather.csv'
        data = loadData(path)

        #  Test dataset path
        path = '../data/test.csv'
        testdata = loadData(path)

        # Extraction of features not needed; although data scaling is needed
        x_test_prep = prepData(testdata, data)

        # Load model
        model = loadModel('qboost.pkl')

        # Predict with model
        predictions, predict_time = predictModel(model, x_test_prep)
        print()
        print(predictions)

        # Create array for DB update
        # Convert the np array to dataframe
        df_predictions = pd.DataFrame(predictions, columns=['label'])

        # Concat dataframes horizontally
        result_data = pd.concat([testdata, df_predictions], axis=1)

        # Convert Dataframe to array for insertion
        result_data = result_data.values

        #print(result_data)
        print(len(result_data.shape))

        # Create new key for the DB table
        res = createKey(result_data)

        # Insert a new first column with the created keys
        result_data = np.insert(result_data, 0, res, axis=1)

        # Delete rows from result where the results are NOT ATTACK
        result_data = delRow(result_data, -1, (result_data.shape[1] - 1))   # TO DO: Analyse and remove this to permit
        # all data

        # Process the result_data for DB upload
        # Store the results to be uploaded in an array
        result_data[:, 3] = 'No Info'  # TO DO: Change this to include a summary of data that entered
        result_data[:, -2] = category
        result_data[:, -1] = 'Attack State'  # TO DO: Change this to reflect all data from the dataset

        # Delete 5th (0, 1, 2, 3, 4<-) column(axis=1)  from the result_data np array
        result_data = np.delete(result_data, obj=4, axis=1)

        #result_data = np.array([str(data).strip() for data in result_data])

        print(result_data)

        # print("Predicted Value (x[" + str(randomtestnum) + "]):" + str(predictions[0]))
        # print("Test Value (y[" + str(randomtestnum) + "]):" + str(y_test[randomtestnum]))
        # print(testdata)
        # print(predictions)

        updateDB(result_data)
    # Only Predict data
    elif args[0] == 'randomtest':
        predict_type = 'test'

        predict_type = args[0]
        #  dataset path
        path = '../data/IoT_Weather.csv'
        data = loadData(path)

        # Extract the feature names and label name
        alist, flist, label = featureList(data)
        """ Separate the dependent and independent features """
        independent_data = data[alist]
        # print(independent_data)
        dependent_data = data[label]
        dependent_data = dependent_data.replace(0, -1)  # change 0 to -1 to fir to quantum algorithm
        # print(dependent_data)

        # Split data for machine learning
        x_train, x_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.2,
                                                            shuffle=True, random_state=23)

        # Apply down sampling on the original unscaled dataframe
        # to access the original data in order to update the DB -
        # Maintains same index as Down Sampling is done using same random number seed
        xdata, ytest = downSample(x_test, y_test)

        # Prepare data for model fitting
        x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            prepData(x_train, x_test, y_train, y_test, flist, label)

        # --------------------------------------------------------
        # Edit the data to remove all but 1 row.
        # Just pass 1 record to the model
        # --------------------------------------------------------
        # Generate random number from the test data
        randomtestnum = random.randint(1, len(x_test_prep))

        # select one record from the test data with index as the random number
        data = x_test_prep[randomtestnum].reshape(1, -1)
        # print(data)
        print(randomtestnum)

        # Load the model
        model = loadModel('qboost.pkl')

        # Predict with the model
        predictions, predict_time = predictModel(model, data)

        print("Predicted Value (x[" + str(randomtestnum) + "]):" + str(predictions[0]))
        print("Test Value (y[" + str(randomtestnum) + "]):" + str(y_test_prep[randomtestnum]))
        # reportResult(y_test, predictions, str(model)[1:4], predict_type, 0, predict_time)

        print(xdata.iloc[randomtestnum])
        #result_data = xdata.iloc[randomtestnum].to_numpy()
        result_data = xdata.iloc[randomtestnum].values

        print(result_data)

        # Create the key for insertion into DB
        res = createKey(result_data)

        # Process the result_data for DB upload
        # Store the results to be uploaded in an array
        result = np.array([str(res), str(result_data[0]).strip(), str(result_data[1]).strip(),
                           str('No Info').strip(), str(category).strip(), str(result_data[6].strip())])

        print(result)
        print(len(result.shape))

        # Delete 4th (0, 1, 2, 3 <-)  from the result_data np array
        # result_data = np.delete(result_data, [2, 3])
        # result_data[0] = str(result_data[0]).strip()
        # result_data[1] = str(result_data[1]).strip()
        # result_data[2] = str('No Info').strip()
        # result_data[3] = str(category).strip()
        # result_data[4] = str(result_data[4].strip())

        # print(result_data)
        # print(len(result_data.shape))

        if predictions[0] == 1:
            if str(result[5]).strip() != 'normal':
                updateDB(result)
            else:
                print('False Positive. No Offense.')
        else:
            print('No offense.')


if __name__ == "__main__":
    main()
