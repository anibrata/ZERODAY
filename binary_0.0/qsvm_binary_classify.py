import matplotlib.pyplot as plt
import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.svm import SVC

# import methods from classifyMain program
from dataprep import *

#####################
# Qiskit Deprecated #
#####################


def plot_features(ax, features, labels, class_label, marker, face, edge, label):
    # A train plot
    ax.scatter(
        # x coordinate of labels where class is class_label
        features[np.where(labels[:] == class_label), 0],
        # y coordinate of labels where class is class_label
        features[np.where(labels[:] == class_label), 1],
        marker=marker,
        facecolors=face,
        edgecolors=edge,
        label=label,
    )


def plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total):
    plt.figure(figsize=(5, 5))
    plt.ylim(0, 2 * np.pi)
    plt.xlim(0, 2 * np.pi)
    plt.imshow(
        np.asmatrix(adhoc_total).T,
        interpolation="nearest",
        origin="lower",
        cmap="RdBu",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
    )
    # A train plot
    plot_features(plt, train_features, train_labels, 0, "s", "w", "b", "A train")
    # B train plot
    plot_features(plt, train_features, train_labels, 1, "o", "w", "r", "B train")
    # A test plot
    plot_features(plt, test_features, test_labels, 0, "s", "b", "w", "A test")
    # B test plot
    plot_features(plt, test_features, test_labels, 1, "o", "r", "w", "B test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("Ad hoc dataset")
    plt.show()


# Define Quantum Kernel
def quantum_kernel(adhoc_dimension):
    adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension,
                                     reps=2, entanglement="linear")
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
    return adhoc_kernel

#############################################################
# Classification using Scikit Learn SVC - Traditional
#############################################################


# Kernel as a callable function
def svc_callable(train_features, train_labels, test_features, test_labels, adhoc_kernel):
    adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
    adhoc_svc.fit(train_features, train_labels)
    adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)
    print(f"Callable kernel classification test score: {adhoc_score_callable_function}")


# Precomputed kernel matrix
def scv_precomputed(train_features, train_labels, test_features, test_labels, adhoc_kernel):
    adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
    adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.asmatrix(adhoc_matrix_train),
                  interpolation="nearest", origin="upper", cmap="Blues")
    axs[0].set_title("Ad hoc training kernel matrix")
    axs[1].imshow(np.asmatrix(adhoc_matrix_test),
                  interpolation="nearest", origin="upper", cmap="Reds")
    axs[1].set_title("Ad hoc testing kernel matrix")
    plt.show()
    adhoc_svc = SVC(kernel="precomputed")
    adhoc_svc.fit(adhoc_matrix_train, train_labels)
    adhoc_score_precomputed_kernel = adhoc_svc.score(adhoc_matrix_test, test_labels)
    print(f"Precomputed kernel classification test score: {adhoc_score_precomputed_kernel}")


# Classification with QSVC
def qsvc_nat(train_features, train_labels, test_features, test_labels, adhoc_kernel):
    print(train_features)
    print(train_labels)
    print(test_features)
    print(test_labels)
    qsvc = QSVC(quantum_kernel=adhoc_kernel)
    start = timer()
    qsvc.fit(train_features, train_labels)
    end = timer()
    train_time = end - start
    #
    # qsvc_score = qsvc.score(test_features, test_labels)
    # print(f"QSVC classification test score: {qsvc_score}")
    #
    # Use Predict Model function from classifyMain
    predictions, predict_time = predictModel(qsvc, test_features)
    return train_time, predict_time, predictions


def main():
    algorithm_globals.random_seed = 12345

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

    print(x_train_prep)
    print(x_test_prep)
    print(y_train_prep)
    print(y_test_prep)
    print(data_dimension)
    # exit(0)

    # FINISH Data load from real dataset

    # adhoc_dimension = 2
    # train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    #     training_size=10,
    #     test_size=5,
    #     n=adhoc_dimension,
    #     gap=0.3,
    #     plot_data=False,
    #     one_hot=False,
    #     include_sample_total=True,
    # )

    # plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total)
    qk = quantum_kernel(data_dimension)
    # SVC using kernel as callable function
    # svc_callable(train_features, train_labels, test_features, test_labels, qk)
    # SVC using precomputed kernel matrices
    # scv_precomputed(train_features, train_labels, test_features, test_labels, qk)

    # QSVC using qiskit native
    train_time, predict_time, predictions = qsvc_nat(x_train_prep, y_train_prep,
                                                     x_test_prep, y_test_prep, qk)
    # Create report file with details
    reportResult(y_test_prep, predictions, 'qsvc', 'test',
                 train_time, predict_time)


if __name__ == '__main__':
    main()
