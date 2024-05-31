# Main program to train the model and test it
# The training and validation would be done by using a 5/10-fold cross validation

# Import statements
# import pandas as pd
# import numpy as np
# import os
import time
import sys
import logging
from datetime import *
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, classification_report
from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
# from fastcore.basics import *
# from fastcore.parallel import *
# from os import cpu_count
import pickle
# import subprocess
import seaborn as sns

from fastai.tabular.all import df_shrink
# import gc
# from time import sleep

from zerodaydata import *
from preprocess import *

# Group 5: Custom imports Qboost Library
from qboost.qboost import *
from dwave.system import DWaveSampler, EmbeddingComposite


def save_model(model):
    # Save model using Pickle
    model_p = 'rf_model.pkl'
    pickle.dump(model, open(model_p, 'wb'))


def reportResult(y, predictions, model, train_time, predict_time):
    # Save the results in a text file
    print("Reporting Result ...")
    result_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report = classification_report(y, predictions)
    with open('../results/' + str(result_time) + '_' + str(model) + '_report.txt', 'a') as f:
        f.write('Training time :' + str(train_time) + '\n')
        f.write(report)
        f.write('Prediction time :' + str(predict_time) + '\n')
    f.close()
    print("Saved...")


def train_classify_RF(X, y, cond_cv):
    print("Inside train_classify_RF()")
    # Train and validate the dataset using a 5-fold cross validation

    # Create a Random Forest classifier with 50 decision trees (as per Sarhan et al. 2023)
    rf_classifier = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None)

    if cond_cv:
        # Use 5-fold cross validation to evaluate the model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Start cross-validation ...")
        # Perform cross validation and get the average accuracy,
        scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')
        print("Cross-validation complete ...")
        average_accuracy = np.mean(scores)
        accuracy_std = np.std(scores)

        print('Average Accuracy (Cross-Val): %.3f' % average_accuracy)
        print('Average Standard Deviation (Cross-Val): %.3f' % accuracy_std)

    # Start the timer
    start = timer()
    # Fit the classifier on the entire training dataset
    rf_classifier.fit(X, y)
    # Stop the timer
    end = timer()
    train_time = end - start
    # Show total training time taken - This includes cross validation time.
    print("Training Time  (Training Data): ", train_time, " seconds")

    # Start the timer
    start = timer()
    # Make predictions on the entire training dataset
    y_train_pred = rf_classifier.predict(X)
    # Stop the timer
    end = timer()
    pred_time = end - start
    # Show total training time taken - This includes cross validation time.
    print("Prediction Time (Training Data): ", pred_time, " seconds")

    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y, y_train_pred)
    print(cm)
    # Plot confusion matrix using seaborn
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # # plt.show()
    #
    # # plt.show()
    # plt.savefig('../results/conf_train_matrix.png')
    #
    # # Plot the confusion matrix
    # """metrics.plot_confusion_matrix(rf_classifier, X, y, cmap='Blues')
    # plt.savefig('../results/conf_train_matrix.png')"""
    #
    # # Plot the ROC curve
    # """metrics.plot_roc_curve(rf_classifier, X, y)
    # plt.savefig('../results/roc_train_curve.png')"""
    #
    # RocCurveDisplay.from_predictions(y, y_train_pred)
    # # plt.show()
    # plt.savefig('../results/roc_train_curve.png')

    # Plot the precision-recall curve
    """metrics.plot_precision_recall_curve(rf_classifier, X, y)
    plt.savefig('../results/prc_train_curve.png')"""

    # display = PrecisionRecallDisplay.from_predictions(
    #     y, y_train_pred, name="RF-gini", plot_chance_level=True
    # )
    # _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.savefig('../results/prc_train_curve.png')

    # Calculate the Accuracy, Precision, Recall, F1-Score, and AUC
    auc_score = roc_auc_score(y, y_train_pred)
    print("Training AUC-Score:", auc_score)

    return rf_classifier, train_time


# Test the model on the test dataset (Zero day dataset)
def test_classify_RF(model, X, y):
    print("Inside test_classify_RF()")
    # Start the timer
    start_time = timer()
    # Make predictions on the test dataset
    y_test_pred = model.predict(X)
    # End the timer
    end_time = timer()

    pred_time = end_time - start_time

    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y, y_test_pred)
    print(cm)
    # Plot confusion matrix using seaborn
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # # plt.show()
    #
    # # plt.show()
    # plt.savefig('../results/conf_test_matrix_rf.png')
    #
    # # Plot the ROC curve
    # RocCurveDisplay.from_predictions(y, y_test_pred)
    # plt.savefig('../results/roc_test_curve_rf.png')
    #
    # # Plot the precision-recall curve
    # display = PrecisionRecallDisplay.from_predictions(
    #     y, y_test_pred, name="RF-gini", plot_chance_level=True
    # )
    # _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.savefig('../results/prc_test_curve_rf.png')

    # Calculate the Accuracy, Precision, Recall, F1-Score, and AUC
    auc_score = roc_auc_score(y, y_test_pred)
    print("Test AUC-Score:", auc_score)

    print("Test Time taken: ", pred_time, "seconds")

    return y_test_pred, pred_time


""" START Insert methods for QBoost Cross Validation and QBoost Test """

""" Training Qboost and prediction with QBoost """
""" This model has returned the best lambda value as  lam = 0.08506944444444445 """
""" Once the lambda value is fixed, the model can be used for test """
""" So, validation is not needed every time. It will be run once and then 
 the best lambda value would be used for the test """


def train_classify_Qboost(X, y, cv):
    print('Inside train_classify_Qboost()')
    """ Evaluate the QBoost model for the value of lambda to be used """
    n_features = np.size(X, 1)
    print('Number of features:', n_features)
    print('Number of training samples:', len(X))

    """ Create block to override cross-validation """
    if cv:
        print('Carrying out cross validation. Crossval: ', cv)
        """ Use cross validation to find out the lambda value """
        # See Boyda et al. (2017), Eq. (17) regarding normalization
        normalized_lambdas = np.linspace(0.0, 1.75, 10)
        lambdas = normalized_lambdas / n_features
        print('Performing cross-validation using {} '
              'values of lambda, this make take several minutes...'.format(len(lambdas)))
        clf_qboost, lam = qboost_lambda_sweep(X, y, lambdas, verbose=True)
        print('Best Classifier: ', clf_qboost)
        print('Best lambda value: ', lam)
        # print('Best features: ', bfeatures)
    else:
        # lam = 0.07142857142857142
        # lam = 0.085079365079365
        lam = 0.003003003003003

    print('Lambda value: ', lam)

    """ Use the best lambda value for the QBoost model training """
    """ Start Timer for QBoost training """
    start = timer()
    qboost = QBoostClassifier(X, y, lam)
    """ End Timer """
    end = timer()
    train_time = end - start
    print('QBoost Training time in seconds :', train_time)

    return qboost, train_time


def test_classify_Qboost(model, X, y):
    print('Inside test_classify_Qboost()')
    print('Number of test samples:', len(X))

    dir = 'PCA-num'
    if not os.path.exists(dir):
        os.makedirs(dir)

    """ Predict with Qboost and evaluate the model """
    """ Start timer for QBoost prediction """
    start = timer()
    y_pred = model.predict_class(X)

    print(type(y_pred))
    print('y_pred: ', y_pred)

    """ End timer """
    end = timer()
    pred_time = end - start
    print('QBoost Prediction time in seconds :', pred_time)

    print(y_pred)

    # exit(0)

    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y, y_pred)
    print(cm)
    # Plot confusion matrix using seaborn
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # # plt.show()
    #
    # # Dynamically include the new directory "dir" in the path while saving a file
    # plt.savefig(f'../results/{dir}/conf_test_matrix_QB.png')
    #
    # # plt.show()
    # # plt.savefig('../results/conf_test_matrix_QB.png')
    #
    # # Plot the ROC curve
    # RocCurveDisplay.from_predictions(y, y_pred)
    # plt.savefig('../results/roc_test_curve_QB.png')
    #
    # # Plot the precision-recall curve
    # display = PrecisionRecallDisplay.from_predictions(
    #     y, y_pred, name="QBoost", plot_chance_level=True
    # )
    # _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.savefig('../results/prc_test_curve_QB.png')

    # Calculate the Accuracy, Precision, Recall, F1-Score, and AUC
    auc_score = roc_auc_score(y, y_pred)
    print("Test AUC-Score:", auc_score)

    print("Test Time taken: ", pred_time, "seconds")

    return y_pred, pred_time


""" END Insert methods for Qboost complete """


def train_test_qboost(X, y, xtest, ytest, lam):
    print('Lambda value: ', lam)

    """ Use the best lambda value for the QBoost model training """
    """ Start Timer for QBoost training """
    start = timer()
    qboost = QBoostClassifier(X, y, lam)
    """ End Timer """
    end = timer()
    train_time = end - start
    print('QBoost Training time in seconds :', train_time)

    print('Number of test samples:', len(xtest))

    """ Predict with Qboost and evaluate the model """
    """ Start timer for QBoost prediction """
    start = timer()
    y_pred = qboost.predict_class(xtest)

    print(type(y_pred))
    print('y_pred: ', y_pred)

    """ End timer """
    end = timer()
    pred_time = end - start
    print('QBoost Prediction time in seconds :', pred_time)
    print(y_pred)

    # Calculate confusion matrix
    cm = metrics.confusion_matrix(ytest, y_pred)
    print(cm)

    """# Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # plt.show()
    plt.savefig('../results/conf_test_matrix_QB.png')

    # Plot the ROC curve
    RocCurveDisplay.from_predictions(y, y_pred)
    plt.savefig('../results/roc_test_curve_QB.png')

    # Plot the precision-recall curve
    display = PrecisionRecallDisplay.from_predictions(
        y, y_pred, name="QBoost", plot_chance_level=True
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    plt.savefig('../results/prc_test_curve_QB.png')"""

    # Calculate AUC
    auc_score = roc_auc_score(ytest, y_pred)
    print("Test AUC-Score:", auc_score)

    print("Test Time taken: ", pred_time, "seconds")

    return y_pred, pred_time, train_time


# Define the main function
def main():
    # Save the original stdout and stderr
    #original_stdout = sys.stdout
    #original_stderr = sys.stderr

    # Specify the log file
    #log_file_path = 'output.log'

    # If the file exists zero the contents and Open the log file in write mode
    #log_file = open(log_file_path, 'w')

    # Redirect stdout and stderr to the log file
    #sys.stdout = log_file
    #sys.stderr = log_file

    # Capture the first argument
    #args = sys.argv[1:]

    # Create a list variable to number of features to be used
    # num = [25, 50, 75, 100, 125, 150, 175, 200, 225]
    num = [175]
    # print("Number of Features: ", num)

    # Check the logic here: it seems that the number of features is not matching the pca boolean variables in the
    # create_data function ***

    for val in num:
        print("Number of Features: " + str(val))

        # Call the function
        X_train, y_train, X_test, y_test = create_data("zero", val, pca_analysis=False, pca_hardcode=True)

        print(X_train, y_train, X_test, y_test)

        # Delete X and y
        """del X, y, X_train, y_train
        sleep(5)
        gc.collect()"""

        # Call the zeroday_create_data function
        # X_test, y_test = zeroday_create_data(zeroday, args[0])

        print(type(X_train), type(y_train), type(X_test), type(y_test))

        # Train and test QBoost
        print("Qboost")
        # Call the train_classify to train and validate model
        model, train_time = train_classify_Qboost(X_train, y_train, False)  # use False to run without cv (harcoded cv value)
        # Show total memory usage
        print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")
        # Call the test_classify function
        y_test_pred, pred_time = test_classify_Qboost(model, X_test, y_test)
        # Create Results File
        reportResult(y_test, y_test_pred, "qboost", train_time, pred_time)

        """lam = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055]
        for i in lam:
            y_test_pred, pred_time, train_time = train_test_qboost(X_train, y_train, X_test, y_test, i)
            reportResult(y_test, y_test_pred, "qboost_short", train_time, pred_time)"""

        print("Random Forest")
        # Train and test Random Forest
        # Call the train_and_validate function
        model, train_time = train_classify_RF(X_train, y_train, True)  # use False to run without cv
        # Show total memory usage
        print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")
        # Call the test_model function
        y_test_pred, pred_time = test_classify_RF(model, X_test, y_test)
        # Create Results File
        reportResult(y_test, y_test_pred, "rf", train_time, pred_time)

        print("**************************************************************")

        # exit(0)
        # break

    """ 
    Program stops with error:
    Process finished with exit code 137 (interrupted by signal 9:SIGKILL)
    Resolve this issue
    ********************
    Code is working now as expected; if memory issue occurs then please check for it again
    """
    print("Done")

    # Close the log file and restore original stdout and stderr
    #log_file.close()
    #sys.stdout = original_stdout
    #sys.stderr = original_stderr


# Call main function
if __name__ == "__main__":
    main()
