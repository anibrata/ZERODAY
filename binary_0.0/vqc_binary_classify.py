import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.circuit.library import EfficientSU2

# import methods from classifyMain program
from dataprep import *

algorithm_globals.random_seed = 42
# create empty array for callback to store evaluations of the objective function
# (as global variable)
objective_func_vals = []


def vqc_nat(feature_map, ansatz):
    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=30),
        callback=callback_graph,
    )


# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    #plt.show()


def main():
    # num_inputs = 2
    # num_samples = 20
    # X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
    # y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
    # y = 2 * y01 - 1  # in {-1, +1}
    # y_one_hot = np.zeros((num_samples, 2))
    # for i in range(num_samples):
    #     y_one_hot[i, y01[i]] = 1
    #
    # for x, y_target in zip(X, y):
    #     if y_target == 1:
    #         plt.plot(x[0], x[1], "bo")
    #     else:
    #         plt.plot(x[0], x[1], "go")
    # plt.plot([-1, 1], [1, -1], "--", color="black")
    # plt.show()

    x_train_prep, x_test_prep, y_train_prep, y_test_prep, data_dimension = makeData()

    # construct feature map, ansatz, and optimizer (num_inputs ~ data_dimension)
    feature_map = ZZFeatureMap(data_dimension, reps=1)
    ansatz = RealAmplitudes(data_dimension, reps=3)

    # Alternatively, different ansatz can also be used
    # ansatz = EfficientSU2(num_qubits=data_dimension, reps=3)

    # variational quantum classifier
    vqc = vqc_nat(feature_map, ansatz)

    ###################################################
    # Training
    plt.rcParams["figure.figsize"] = (12, 6)

    start = timer()
    # fit classifier to data
    vqc.fit(x_train_prep, y_train_prep)
    end = timer()
    train_time = end - start

    # return to default figsize
    plt.rcParams["figure.figsize"] = (6, 4)

    # score classifier
    vqc.score(x_train_prep, y_train_prep)
    # plt.show()

    ###################################################
    #TESTING / PREDICTING
    start = timer()
    # evaluate data points
    predictions = vqc.predict(x_test_prep)
    end = timer()
    predict_time = end - start

    # Create report file with details
    reportResult(y_test_prep, predictions, 'vqc', 'test',
                 train_time, predict_time)

    # plot results
    # red == wrongly classified
    # for x, y_target, y_p in zip(X, y_one_hot, y_predict):
    #     if y_target[0] == 1:
    #         plt.plot(x[0], x[1], "bo")
    #     else:
    #         plt.plot(x[0], x[1], "go")
    #     if not np.all(y_target == y_p):
    #         plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
    # plt.plot([-1, 1], [1, -1], "--", color="black")
    # plt.show()


if __name__ == '__main__':
    main()
