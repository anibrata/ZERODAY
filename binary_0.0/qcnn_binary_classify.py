#################################################################
# Import libraries
#################################################################
import os
import re
import sys
import random
import string
import time
from datetime import *
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
# import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import randint
# Modelling
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
#from qboost.qboost import QBoostClassifier
#from dwave.system.samplers import DWaveSampler
#from dwave.system.composites import EmbeddingComposite
# import psycopg2
# from psycopg2.extras import RealDictCursor
from imblearn.under_sampling import RandomUnderSampler
from timeit import default_timer as timer
from collections import Counter

# from connect import connect
################################################################
# Import Pickle and Joblist to test Serialization of Model
################################################################
import pickle
import joblib
################################################################
from IPython.display import clear_output
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# from ibm_quantum_widgets import *
from qiskit_aer import AerSimulator
# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector

#####################################################################

from sklearn.manifold import TSNE
import seaborn as sns

#########################################################
# tab2Img from deepInsight
from tab2img.converter import Tab2Img

from IPython.display import clear_output

#########################################################
# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# import json
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

# Declare environment variables
algorithm_globals.random_seed = 12345
objective_func_vals = []  # to store loss function values


# We now define a two qubit unitary as defined in [3] - Circuit
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


# Define the convolutional layer
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


# Define pooling circuit
def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


# Define the pooling layer
def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index: (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# Define the callback function for the loss function evaluation
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


#################################################################
# Load csv into memory
#################################################################


def loadData(path):
    data = pd.read_csv(path)
    # print(data)
    return data


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
    # print(data.shape[1])

    # define the min-max scaler
    scaler = MinMaxScaler()

    # Check the shape of data; and process as per the shape; if full feature data
    # then just perform full scaling, else just plain transform
    if data.shape[1] > 1:
        # print(data)
        data['time'] = data['time'].str.replace(':', '')
        data['time'] = [int(x) for x in data['time']]
        data['date'] = data['date'].str.replace("-", "")
        # Change the months to numbers
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_num = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        for i in range(12):  # "i" is index to access month and month_num
            data['date'] = data['date'].str.replace(month[i], month_num[i])
        # Scale the data to be within with 0 and 1
        # print(data)
        data[flist] = scaler.fit_transform(data[flist])
        return data
    else:
        data[flist] = scaler.fit_transform(data[flist])
        return data


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


def makeData(path):
    # START Data LOAD from real dataset
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

    print('Data preparation complete')

    return x_train_prep, x_test_prep, y_train_prep, y_test_prep, data_dimension


def main():
    # 5% of main dataset
    path = '../data/percent18.csv'

    x_train_prep, x_test_prep, y_train_prep, y_test_prep, data_dimension = makeData(path)

    # Create a new LENx8 array
    img_array = np.zeros((x_train_prep.shape[0], 8))

    # Copy main array to zero array
    for i in range(x_train_prep.shape[0]):
        img_array[i, :3] = x_train_prep[i]

    # Create a new LENx8 array
    img_array_test = np.zeros((x_test_prep.shape[0], 8))

    # Copy main array to zero array
    for i in range(x_test_prep.shape[0]):
        img_array_test[i, :3] = x_test_prep[i]

    print(img_array, y_train_prep, img_array_test, y_test_prep)

    # Let's draw this circuit and see what it looks like
    # params = ParameterVector("θ", length=3)
    # circuit = conv_circuit(params)
    # circuit.draw("mpl")

    # Create convolutional layer
    # circuit = conv_layer(4, "θ")
    # circuit.decompose().draw("mpl")

    # Draw pooling circuit
    # params = ParameterVector("θ", length=3)
    # circuit = pool_circuit(params)
    # circuit.draw("mpl")

    # Create pooling layer
    # sources = [0, 1]
    # sinks = [2, 3]
    # circuit = pool_layer(sources, sinks, "θ")
    # circuit.decompose().draw("mpl")

    # Define Feature MAP
    feature_map = ZFeatureMap(8)
    #Draw feature MAP Network
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    # Draw the circuit
    # circuit.draw("mpl")

    # Create the classifier
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=1),  # Set max iterations here
        callback=callback_graph,
        # initial_point=initial_point,
    )

    plt.rcParams["figure.figsize"] = (12, 6)
    start = timer()
    classifier.fit(img_array, y_train_prep)
    end = timer()
    train_time = end - start

    # score classifier
    print(f"Accuracy from the train data : {np.round(100 * classifier.score(img_array, y_train_prep), 2)}%")

    start = timer()
    y_predict = classifier.predict(img_array_test)
    end = timer()
    predict_time = end - start
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(img_array_test, y_test_prep), 2)}%")

    # Create report file with details
    reportResult(y_test_prep, y_predict, 'qcnn', 'test', train_time, predict_time)


if __name__ == '__main__':
    main()
