#######################################################################################
# This file is the main file of the project. This is to call the Qboost to 
# perform binary classfication of the CICIoT 2023 Data.
# The target is to run the algorithm on the whole CICIoT 2023 Dataset, but due to 
# restrictions of the project, we are running the algorithm on a subset of the dataset
# After this step, we will run the algorithm on the whole dataset in the SerLab server.
# Author: Anibrata Pal
# Date: 15/09/2023
# Version: 1.0
# Department: SerLab
#######################################################################################
# 
#                       MQTT            CICIot
# Dependent variable    | "target"      | "label"
# Value of the DV       | "legitimate"  | "BenignTraffic"


##############################################################################
##
## TO DO: MODIFY THE CODE TO WORK BOTH BINARY AND MULTI-CLASS BASED ON ARGS
##
## UPDATE: Not done yet
##
##############################################################################

# Group 1: System-related imports
from collections import Counter
import sys
import os
import os.path
import platform
import random
import string
from sys import exit
from pathlib import Path

# Group 2: Data visualization and plotting
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# Group 3: Data preprocessing and scaling
import pandas as pd
from numpy import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Group 4: Machine learning and model evaluation
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# Group 5: Custom imports Qboost Library
from qboost.qboost import *
from qboost import QBoostClassifier
from dwave.system import DWaveSampler, EmbeddingComposite

# Group 6: Timer libraries
import time
from timeit import default_timer as timer
from datetime import *

# Group 7: DB
from connect import connect

# Group 8: Save Models
import pickle
import joblib

# Declare gobal variables: different classes of the dataset
classes = {"BenignTraffic": 0, "DDoS-ICMP_Flood": 1, "DDoS-UDP_Flood": 2, "DDoS-TCP_Flood": 3, "DDoS-PSHACK_Flood": 4, 
            "DDoS-SYN_Flood": 5, "DDoS-RSTFINFlood": 6, "DDoS-SynonymousIP_Flood": 7, "DoS-UDP_Flood": 8, "DoS-TCP_Flood": 9, 
            "DoS-SYN_Flood": 10,  "Recon-PingSweep": 11, "Mirai-greeth_flood": 12, "Mirai-udpplain": 13, "Mirai-greip_flood": 14, 
            "DDoS-ICMP_Fragmentation": 15, "MITM-ArpSpoofing": 16, "DDoS-ACK_Fragmentation": 17, "DDoS-UDP_Fragmentation": 18,
            "DNS_Spoofing": 19, "Recon-HostDiscovery": 20, "Recon-OSScan": 21, "Recon-PortScan": 22, "DoS-HTTP_Flood": 23, 
            "VulnerabilityScan": 24, "DDoS-HTTP_Flood": 25, "DDoS-SlowLoris": 26, "DictionaryBruteForce": 27, 
            "BrowserHijacking": 28, "SqlInjection": 29, "CommandInjection": 30, "XSS": 31, "Backdoor_Malware": 32, 
            "Uploading_Attack": 33, }
# Global variable for lambda
lam = 0.08506944444444445


# Read from a list of csv files existing in a folder into a numpy dataframe
def readAllCSVFiles(folderPath):
    """
    Reads all CSV files in a folder and returns a single pandas DataFrame.

    Parameters:
        folderPath (str): The path of the folder containing the CSV files.

    Returns:
        pd.DataFrame: A single DataFrame containing the data from all CSV files.
    """
    dataFrames = []
    for filename in os.listdir(folderPath):
        if filename.endswith(".csv"):
            filePath = os.path.join(folderPath, filename)
            df = pd.read_csv(filePath)
            dataFrames.append(df)
    combinedDataFrame = pd.concat(dataFrames, ignore_index=True)
    """ Print the list of files"""
    print(os.listdir(folderPath))
    return combinedDataFrame

# Read two specific csv files into a pandas dataframe
def readCSVFile(folderPath):
    #df1 = pd.read_csv(folderPath + "/test30_augmented.csv")
    #df2 = pd.read_csv(folderPath + "/train70_augmented.csv")
    df1 = pd.read_csv(folderPath + "/test30_reduced.csv")
    df2 = pd.read_csv(folderPath + "/train70_reduced.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    """ Free memory """
    del df1, df2
    return df

# Analyze the dataframe to find out the characteristics of the data
def analyzeData(df):
    """ Find the number of rows and columns in the dataFrame """
    print(df.shape)

    """ Find the number of rows which have the value in the last column "legitimate" and which do not have """
    # print(dataFrame[dataFrame['target'] == 'legitimate'].shape[0])
    # print(dataFrame[dataFrame['target'] != 'legitimate'].shape[0])

    """ Change the target value from "legitimate" to "1" and 'not "legitimate"' to "-1" """
    """df['label'] = df['label'].replace('BenignTraffic', 0)
    df.loc[df['label'] != 0, 'label'] = 1""" # Commented binary classification
    
    # Tweaked this part of the code to permit multi-classes in labels
    for key, value in classes.items():
        df.loc[df['label'] == key, 'label'] = value
    
    # Replace both the target values at once using replace and regex (Works but less efficient as it uses regex)
    # df['target'] = df['target'].replace('legitimate', 1).replace(to_replace={'^((?!legitimate).)*$': -1}, regex=True)

    zeros = df[df['label'] == 0].shape[0]
    ones = df[df['label'] == 1].shape[0]
    print('zeros, ones:', zeros, ones)
    return zeros, ones, df

""" Show the datapoints of each column in a graph """
def showData(df):
    df.hist()
    plt.show()


""" Analyze dataframe to find out catagorical and numerical features. Remove categorical features and use only numerical features """
def dataClean(df):
    """ Find out the columns with categorical values and list them """
    catCol = df.columns[df.dtypes == 'object'].to_list()

    """ Remove the item target from the list of categorical columns """
    catCol.remove('label')    
    
    """ Remove categorical columns (as a whole, it removes the target column also) """
    # df = df.select_dtypes(exclude=[int, float]).columns.tolist()

    """ Remove the categorical columns now from the df in situ """
    df = df.drop(columns=catCol)

    """ Find out the columns which have only one unique value """
    uniqueValCol = df.columns[df.nunique() == 1]

    """ Remove the columns that have only one unique vales from the df in situ """
    df = df.drop(columns=uniqueValCol)

    # The lines below are commented not to remove the low impact columns
    """ Use correlation to find unimportant features and remove them """
    # corrNotImpCols = correlations(df)
    # df = df.drop(columns=corrNotImpCols)
    
    """ Find out the max values from each column to avoid 'inf' -> infinity value problem """
    # print(df.max(axis=0))

    """ Some columns above have Infinity value and cannot be processed as it is, so they need to be replaced with NaN
        and then imputed """
    """ Find out the columns which have NaN / Missing values - None here."""
    nan_values = df.columns[df.isna().any()]
    print('NaN Values: ', nan_values)

    """ Check if the dataframe has infinity value and set it as NaN when it is infinity """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    """ Imputing using median for all NaN values """
    df.fillna(df.median(), inplace=True)

    """ If the unique values in a column in the dataset is 1, then remove the column """
    # df = df.drop(columns=df.columns[df.nunique() == 1])

    """ Find the distribution of data in each column """
    #print(df.describe())

    return df

""" Implement a function to check if the dataset is balanced or not; comparison with """
def isBalanced(zeros, ones, size, X, y):
    if abs(zeros - ones) <= 0.005 * size:
        print("Dataset is balanced; nothing to do, going to next step ... ")
        return X, y
    else:
        x_bal, y_bal = balanceData(X, y)
        return x_bal, y_bal

""" Normalize the dataset using MinMaxScalar from sklearn """
def normalizeData(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

""" Normalize the dataset based on binning """
def normalizeDataBinning(df):
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

""" Change the target values from 0 to -1 """
def changeTarget(df):
    df.loc[df['label'] == 0, 'label'] = -1
    return df

""" Separate the target and independent features """
def create_xy(df):
    # Separate Dependent and Independent columns
    shp = df.shape  # Shape of the dataframe
    cols = list(df.columns.values)  # List of features
    # print(cols)

    # Independent data
    X = df[cols[0:shp[1] - 1]]
    # print(X)

    # Dependent data
    y = df['label']
    # print(y)

    # Check
    print('checking data shape before ML !!')
    print('Data Shape: ', X.shape[0], X.shape[1], y.shape[0])

    return X, y

""" Implement the correlation coefficient between each feature and the target variable """
def correlations(data):
    correlations = data.corr()['label'].sort_values(ascending=True)  # Sort the values in ascending order of correlation
    # Extract the correlation values which are between -0.1 and 0.1
    correlations = correlations[(correlations >= -0.1) & (correlations <= 0.1)]
    # create a numpy array from correlations
    feature_names = np.array(correlations.index)
    # print(feature_names)
    return feature_names

""" Implement PCA and use both threshold and elbow methods to find out the number of components """
def pca(X):
    print('Performing PCA ...')
    # Create a PCA object
    pca = PCA()

    # Fit PCA on your dataset
    pca.fit(X)

    # Get the explained variance ratio / cumulative 
    explained_variance_ratio = pca.explained_variance_ratio_

    elbow_index = elbow(explained_variance_ratio)
    threshold_index = threshold(explained_variance_ratio)

    if elbow_index <= threshold_index:
        n_components = elbow_index + 1
    else:
        n_components = threshold_index + 1

    """ Hardcoding the number of components for the MQTT dataset """
    # n_components=16

    """ Hardcoding the number of components for the CICIoT dataset """
    # n_components=16

    print('Number of components:', n_components)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print('PCA done ... ')
    return X_pca

""" Implement the threshold method """
def threshold(explained_variance_ratio):
    # Find the best number of components by the Threshold method
    threshold_val = 0.99  # Define the desired threshold (e.g., 99% variance explained)

    # Calculate the cumulative sum of the explained variance ratio
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find the number of components above the threshold
    threshold_index = np.argmax(cumulative_variance >= threshold_val) + 1

    # Plot the threshold line
    """plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
    plt.axhline(y=threshold_val, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.show()"""
    print("Number of components above the threshold:", threshold_index)

    return threshold_index

""" Implement Elbow method to find the optimal number of components """
def elbow(explained_variance_ratio):
    # Finding the best number of components by Elbow method
    # Calculate the difference in explained variance between components
    explained_variance_diff = np.diff(explained_variance_ratio)
    """plt.plot(range(1, len(explained_variance_diff) + 1), explained_variance_diff)
    plt.xlabel('Number of Components')
    plt.ylabel('explained_variance_ratio')
    plt.title('explained_variance_ratio vs. Number of Components')
    plt.show()"""

    # Find the index of the elbow point (maximum difference)
    elbow_index = np.argmax(explained_variance_diff) + 1

    print("Number of components at the elbow point:", elbow_index)

    return elbow_index

""" Training Qboost and prediction with QBoost """
""" This model has returned the best lambda value as  lam = 0.08506944444444445 """
def classifyQboost(X, y, crossval):
    """ Split the dataset into training and testing sets in a ratio of 80:20 """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    """ Evaluate the QBoost model for the value of lambda to be used """
    n_features = np.size(X, 1)
    """ Free memory """
    del X, y    
    print('Number of features:', n_features)
    print('Number of training samples:', len(X_train))
    print('Number of test samples:', len(X_test))
    
    """ Create block to override cross-validation """
    if crossval == str(1):
        print('Carrying out cross validation. Crossval: ', crossval)
        """ Use cross validation to find out the lambda value """    
        # See Boyda et al. (2017), Eq. (17) regarding normalization
        normalized_lambdas = np.linspace(0.0, 1.75, 10)
        lambdas = normalized_lambdas / n_features
        print('Performing cross-validation using {} '
                  'values of lambda, this make take several minutes...'.format(len(lambdas)))
        clf_qboost, lam, bfeatures = qboost_lambda_sweep(X_train, y_train, lambdas, verbose=True)
        print('Best Classifier: ', clf_qboost)
        print('Best lambda value: ', lam)
        print('Best features: ', bfeatures)
    else:
        #lam = 0.07142857142857142
        lam = 0.08506944444444445   # declared in global variable

    """ Use the best lambda value for the QBoost model training """
    """ Start Timer for QBoost training """
    start = timer()
    qboost = QBoostClassifier(X_train, y_train, lam)
    """ End Timer """
    end = timer()
    train_time = end - start
    print('QBoost Training time in seconds :', train_time)

    """ Predict with Qboost and evaluate the model """
    """ Start timer for QBoost prediction """
    start = timer()
    y_pred = qboost.predict_class(X_test)
    """ End timer """
    end = timer()
    pred_time = end - start
    print('QBoost Prediction time in seconds :', pred_time)

    report = classification_report(y_test, y_pred)
    print(report)

    return qboost, train_time, pred_time, y_pred, y_test


""" Training Qboost and save multiple models, one for each class - Ove vs Rest Strategy """
def trainQboost_multiclass(X, y):
    """ Split the dataset into training and testing sets in a ratio of 80:20 """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # For every class carry out training and prediction
    for key, value in classes.items():
        # Change labels of dependent variables for OvR strategy
        y_train_multi = changeLabel(y_train, value)
        # y_test_multi = changeLabel(y_test, value)

        # Define QBoost model for training
        """ Use the best lambda value for the QBoost model training """
        """ Start Timer for QBoost training """
        start = timer()
        model = QBoostClassifier(X_train, y_train_multi, lam)
        """ End Timer """
        end = timer()
        train_time = end - start
        print('QBoost Training time in seconds :', train_time)

        # Delete y_train_multi to remove any memory leaks, and label issue
        del y_train_multi

        # Save Model
        save_model(model, key)

    """ Delete the DF to save memory """
    del X_train, X_test, y_train, y_test

""" Testing the multiclass Qboost model """
def testQboost_multiclass(X, y):
    """ Split the dataset into training and testing sets in a ratio of 80:20 """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)  # same random state used to get same samples

    # For every model carryout prediction
    for file in os.listdir():
        if file.endswith('.pkl'):
            model = joblib.load(file)
            accuscore = 0
            # For every class - prediction is done to find out the best classifier for the class
            for key, value in classes.items():
                # Prepare Test data for specific binary classification
                y_test_multi = changeLabel(y_test, value)
                """ Predict with Qboost and evaluate the model """
                """ Start timer for QBoost prediction """
                start = timer()
                y_predictions = model.predict_class(X_test)
                """ End timer """
                end = timer()
                pred_time = end - start
                # print('Intermediate QBoost Prediction time in seconds :', pred_time)

                # Evaluate the model; compare the accuracy score for all the classes; the best is the one with the highest score
                score = accuracy_score(y_test_multi, y_predictions)
                if score > accuscore:
                    accuscore = score
                    report = classification_report(y_test_multi, y_predictions, output_dict=False) # Output dict -> False to print to file
                    prediction_time = pred_time
                    y_conf = y_test_multi
                    y_pred_conf = y_predictions
            print('Prediction done with Model - ', file)
            # print(report)
            reportResult(report, model, prediction_time)
            print('Final QBoost prediction time in seconds :', prediction_time)

            """ Calculate the confusion matrix """
            confusion_matrix = confusionMatrix(y_conf, y_pred_conf)
            print(confusion_matrix)
            # predict.update(key, metrics.append(score))

""" Change the labels of the dependent variables so that there are only 2 classes """
def changeLabel(y_multiclass, value):
    # Set value of Target Class to 1
    y_multiclass_updated = np.where((y_multiclass == value), 1, -1)
    return y_multiclass_updated

""" Save the model """
def save_model(model, key):
    model_file = key + '.pkl'
    joblib.dump(model, model_file)
    print('Model Saved :', model)

""" Implement confusion matrix """
def confusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # The lines below are commented; will be printed only when the matrix figure is needed
    """cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()
    plt.savefig('../results/conf_matrix.png')"""
    return cm

""" Implement the SMOTE and RandomUnderSampling to balance the dataset """
def balanceData(X, y):
    """ Simple RandomUnderSampling """
    """print("Dataset is not balanced; balancing dataset ... ")
    # Reduce the data size to match the minority class
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=17)
    # fit and apply the Under Sampling Transform
    x_under, y_under = undersample.fit_resample(X, y)"""

    """ Implement SMOTE with K nearest neighbour and RandomUnderSampler """
    # Nearest neighbor values for SMOTE
    k_values = [1, 2, 3, 4, 5, 6, 7]
    s_values = [0.5, 0.6, 0.7, 0.8]
    
    # Declare score = 0
    fscore = 0
    fs  = 0
    fk = 0

    # Create a KFold object with n_splits=10
    """for s in s_values:
        for k in k_values:
            # Declare the models for SMOTE over and Random Undersampling
            # define pipeline
            model = DecisionTreeClassifier()
            over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=s, random_state=15)
            steps = [('over', over), ('under', under), ('model', model)]
            pipeline = Pipeline(steps=steps)
            # evaluate pipeline
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, y, scoring='recall', cv=cv, n_jobs=-1)
            score = mean(scores)
            if fscore < score:
                fscore = score
                fs = s
                fk = k
                print('HIGHEST SCORE: > k=%d, s=%.1f, Mean Recall: %.3f' % (k, s, score))
            print('> k=%d, s=%.1f, Mean Recall: %.3f' % (k, s, score))
            print('*******************************************')"""

    """ Hardocode s and k values as the experiment was run and the best values were found """
    """ Same reason lines 378 to 397 were commented """
    fs = 0.8
    fk = 7

    # Apply SMOTE and Random Under Sampling to recreate a balanced dataset
    X, y = apply_smote_under(X, y, fs, fk)
    
    # Create a module to find out the best values of s and k and return those
    return X, y

# Apply SMOTE oversampling and Random Under Sampling
def apply_smote_under(X, y, s, k):
    over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=s)
    X, y = over.fit_resample(X, y)
    X, y = under.fit_resample(X, y)
    return X, y

# Report the results in a file
def reportResult(report, model, predict_time):
    #result_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # Check if there are any result.txt file in the directory, if yer then delete it and then open it in append mode
    # Add logic later -  time constraint.
    if os.path.exists('/home/anibrata/Anibrata/PROJECTS/CODE/SMART_CITY_SECURITY_PRIVACY/SMARTCITY_SOC_JSS/results/result.txt'): 
        with open('/home/anibrata/Anibrata/PROJECTS/CODE/SMART_CITY_SECURITY_PRIVACY/SMARTCITY_SOC_JSS/results/result.txt', 'a') as f:
            f.write('Model : ' + str(model) + '\n')
            # f.write('Training time :' + str(train_time) + '\n')
            f.write('Prediction time :' + str(predict_time) + '\n')
            f.write(report)
            f.write('\n')
            f.close()
    else:
        with open('/home/anibrata/Anibrata/PROJECTS/CODE/SMART_CITY_SECURITY_PRIVACY/SMARTCITY_SOC_JSS/results/result.txt', 'w') as f:
            f.write('Model : ' + str(model) + '\n')
            #f.write('Training time :' + str(train_time) + '\n')
            f.write('Prediction time :' + str(predict_time) + '\n')
            f.write(report)
            f.write('\n')
        f.close() 
        """ print("Saved...") """
        

# Main section
if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)

    """ Using CICIoT Dataset """
    folderPath = "/home/anibrata/Anibrata/PROJECTS/CODE/SMART_CITY_SECURITY_PRIVACY/SMARTCITY_SOC_JSS/data"
    #folderPath = "~/PROJECTS/SMARTCITY_JSS/data"
    df = readAllCSVFiles(folderPath)
    print(df)

    """ Using MQTT Dataset """
    #folderPath = "/home/anibrata/Anibrata/PROJECTS/CODE/SMART_CITY_SECURITY_PRIVACY/SMARTCITY_SOC_JSS/MQTT/data"
    #folderPath = "~/PROJECTS/SMARTCITY_JSS/data"
    # df = readCSVFile(folderPath)
    # print(df)

    print('Test1')

    zeros, ones, df = analyzeData(df) # zeros and ones not used for multi-class

    print('Test2')

    df = dataClean(df)

    print('Test3')

    """ Normalize data """
    """ Using Binning """
    #df = normalizeDataBinning(df)  # Works similar to MinMaxScalar and so not used, it's costlier

    """ Change the target values from 0 to -1 """
    # df = changeTarget(df) # Commented for multi-class

    print('Test4')

    """ Create X and y """
    X, y = create_xy(df)

    print('Test5')

    """ Free memory """
    del df

    # Check if the dataset is balanced or not, if not then balance it
    """ If the difference between zeros and ones is greater than 0.1% of the size of the dataset then balance the dataset """
    #X, y = isBalanced(zeros, ones, X.shape[0], X, y)  # Commented for multi class, no balancing needed for now.

    """ Using MinMaxScalar """
    X = normalizeData(X)
    
    if args:
        if args[0] == 'pca':
            """ Dimentionality Reduction using PCA """
            X = pca(X)

    """ Convert the pandas series into numpy array """
    y = y.values.ravel()

    """ Train and Predict with Qboost """
    # model, train_time, pred_time, predictions, y_test = classifyQboost(X, y, args[1])

    """ Training with QBoost and save multi-class models """
    trainQboost_multiclass(X, y)

    """ Predict multi-class with Qboost binary classifier using OvR strategy """
    testQboost_multiclass(X, y)

    """ Calculate the confusion matrix """
    """ confusion_matrix = confusionMatrix(y_test, predictions)
    print(confusion_matrix) """

    print('Reported Results and Done !!!')

