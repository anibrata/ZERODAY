#######################################################################
# This code is for procesing of data for Zero day attacks
# Author: Pal, Anibrata
# Date: 20/12/2023
#######################################################################
import sys

import pandas as pd
import numpy as np


import gc
from time import sleep

import psutil
from fastai.tabular.core import df_shrink

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def read_parquet(path):
    df = pd.read_parquet(path)
    return df


def create_zero_main(df):
    print(df.shape)
    # Select all rows where labels are like "Zero"
    zero_rows = df[df['Label'].str.startswith('Zero')]

    # Randomly select the same number of rows where labels are like "Ben"
    ben_rows = df[df['Label'].str.startswith('Ben')].sample(len(zero_rows), replace=True)

    # Select the rows where labels are not like Zero or are not the  Ben labelled rows selected above from df
    rdf = df[~df['Label'].str.startswith('Zero') & ~df.index.isin(ben_rows.index)]

    # Concatenate the selected rows into a new DataFrame in random order
    cdf = pd.concat([zero_rows, ben_rows]).sample(frac=1).reset_index(drop=True)

    # Shuffle the dataset and reset the index for rdf
    rdf = rdf.sample(frac=1).reset_index(drop=True)

    del zero_rows, ben_rows
    sleep(5)
    gc.collect()

    return rdf, cdf


def balance_dataset(df):
    # Generalized function
    # Check if the difference between the number of rows with Label Benign and number of
    # rows without the same Label is less than 1% of the total number of rows.
    total_rows = df.shape[0]
    benign_rows = df[df['Label'] == 'Benign'].shape[0]
    non_benign_rows = df[df['Label'] != 'Benign'].shape[0]
    difference = abs(benign_rows - non_benign_rows)

    X_balanced = []
    y_balanced = []

    if difference < (total_rows * 0.01):
        print("The difference between the number of rows with Label Benign and "
              "number of rows without the same Label is less than 1% of the total "
              "number of rows. balancing not needed")
    else:
        print("The difference between the number of rows with Label Benign and "
              "number of rows without the same Label is greater than or equal to 1% "
              "of the total number of rows.")
        print("Balancing the data")
        print(benign_rows, non_benign_rows)

        # Change the Label of all Non-Benign rows to Malicious
        df['Label'] = df['Label'].astype(str)
        df.loc[df['Label'] != 'Benign', 'Label'] = 'Malicious'
        print(df)
        # Drop the Label column for RandomUnder sampling or SMOTE
        X = df.drop('Label', axis=1)  # Features (excluding the 'label' column)
        y = df['Label']  # Target variable ('label' column)

        print("% of Benign rows: ", benign_rows / total_rows * 100)
        s_values = [0.83, 0.86, 0.9, 0.93, 0.96, 0.99]
        k_values = [1, 2, 3, 4, 5]

        if benign_rows > non_benign_rows:
            print(" Benign rows are more than Malicious rows")
            # Use Random Under sampler to balance the dataset
            print("Use RandomUndersampler to reduce the number of benign rows to be "
                  "similar to the number of non-benign rows.")
            bal_flag = 0  # Set balance flag =0 when benign rows are more than non-benign rows
            # Balance the dataset
            X_balanced, y_balanced = balance_data(X, y, s_values, k_values, bal_flag)
        else:
            print(" Malicious rows are more than Benign rows")
            bal_flag = 1  # Set balance flag =1 when malicious rows are more than benign rows and
            # malicious rows can't be reduced.
            # Use SMOTE to balance the dataset
            # Balance the dataset
            X_balanced, y_balanced = balance_data(X, y, s_values, k_values, bal_flag)

        # The undersampled DataFrame is now ready to be used
        # print(X_balanced, y_balanced)

    return X_balanced, y_balanced


""" Implement the SMOTE and RandomUnderSampling to balance the dataset """


def balance_data(X, y, s_values, k_values, bal_flag):
    """ Implement SMOTE with K nearest neighbour and RandomUnderSampler """
    # Declare score = 0
    fscore = 0
    fs = 0
    fk = 0

    # Create a KFold object with n_splits=10
    for s in s_values:
        for k in k_values:
            # Declare the models for SMOTE over and Random Undersampling
            # define pipeline using Decision Tree Sampler to check the quality of the model.
            model = DecisionTreeClassifier()
            over = SMOTE(sampling_strategy=s, k_neighbors=k)
            if bal_flag == 0:
                print("Using both SMOTE and Random Undersampling")
                under = RandomUnderSampler(sampling_strategy=s, random_state=42)
                steps = [('over', over), ('under', under), ('model', model)]
            else:
                print("Using only SMOTE")
                steps = [('over', over), ('model', model)]
            pipeline = Pipeline(steps=steps)
            # evaluate pipeline
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, y, scoring='recall', cv=cv, n_jobs=-1)
            score = np.mean(scores)
            if fscore < score:
                fscore = score
                fs = s
                fk = k
                print('HIGHEST SCORE: > k=%d, s=%.1f, Mean Recall: %.3f' % (k, s, score))
            print('> k=%d, s=%.1f, Mean Recall: %.3f' % (k, s, score))
            print('*******************************************')

    # Apply SMOTE and/or Random Under Sampling to recreate a balanced dataset
    X, y = apply_smote_under(X, y, fs, fk, bal_flag)
    return X, y


# Apply SMOTE oversampling and Random Under Sampling
def apply_smote_under(X, y, s, k, bal_flag):
    over = SMOTE(sampling_strategy=s, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=s, random_state=42)
    if bal_flag == 0:
        print("Using both SMOTE and Random Undersampling")
        X, y = over.fit_resample(X, y)
        X, y = under.fit_resample(X, y)
    else:
        print("Using only SMOTE")
        X, y = over.fit_resample(X, y)
    return X, y


def create_zero_trojan_main(df):
    # Create an attack dataframe that includes both zeroday and trojan.
    # This is to check if the model is generalized to capture both zeroday and trojan attacks
    print(df.shape)
    return df, df


def balance_dataset_specific(X, y):
    s = 0.99
    k = 5
    # over = SMOTE(sampling_strategy=s, k_neighbors=k)
    # Do RandomOverSampling to balance the dataset
    over = RandomOverSampler(sampling_strategy=s, random_state=42)

    # Print the type of data in the Label column
    # print("Type of Data in Label column: ", df['Label'].dtypes)

    # Check the unique values in the Label column
    # print("Unique values in Label column: ", df['Label'].unique())

    # Print datatypes of each column in the dataframe
    # print(df.dtypes)

    # Te part below is not needed, the labels should be changed into
    # 1 for Benign and -1 for Malicious
    # To resolve the error "TypeError: Cannot setitem on a Categorical with a new category
    # (Malicious), set the categories first" , update the categories first
    """df['Label'] = df['Label'].cat.add_categories(['Malicious'])

    # Change the Label of all Non-Benign rows to Malicious
    df.loc[df['Label'] != 'Benign', 'Label'] = 'Malicious'

    # Print number of Benign records
    print("Number of Benign records: ", len(df[df['Label'] == 'Benign']))

    # Print number of Malicious records
    print("Number of Malicious records: ", len(df[df['Label'] == 'Malicious']))"""

    """# Change the type of the Label column to string 
    df['Label'] = df['Label'].astype(str)"""

    # Convert the "Label" column to integers (1 for Benign, -1 for Not Benign)
    # df['Label'] = df['Label'].apply(lambda x: 1 if x == 'Benign' else -1)

    """# Set "Benign" to 1 and the rest to -1 dynamically
    unique_categories = df['Label'].unique()
    category_mapping = {category: 1 if category == 'Benign' else -1 for category in unique_categories}
    df['Label'] = pd.Categorical(df['Label'], categories=category_mapping.keys()).codes"""

    # Print the datatype of the Label column
    # print("Type of Data in Label column: ", df['Label'].dtypes)

    # print(df['Label'])

    # Print the total number of records in RDF
    print("Total number of records in RDF: ", len(X))

    # Drop the Label column for Random Under sampling or SMOTE
    # X = df_shrink(df.drop('Label', axis=1))  # Features (excluding the 'label' column)
    # X = df.drop('Label', axis=1)
    # y = df['Label']  # Target variable ('label' column)

    # Show total memory usage
    print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")

    # Print the shapes of X and y, also print the memory usage, also print unique values in y
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    # print("X memory usage: ", X.memory_usage().sum() / 1024**2, "MB")
    # print("y unique values: ", y.unique())
    # print("y type: ", type(y))
    # print("y memory usage: ", y.memory_usage().sum() / 1024 ** 2, "MB")"""

    print("Test/Before Oversampling")
    # Show total memory usage
    # print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")

    # print("Comment oversampling to reduce the proicessing time in order to check "
    #      "an issue with the QBoost lam parameter which is showing different results"
    #      " with different values. Also commented the delete dataframe block and "
    #      "changed the return dataframes.")
    # Apply Oversampling
    X_over, y_over = over.fit_resample(X, y)

    del X, y
    sleep(5)
    gc.collect()

    return X_over, y_over
    # return X, y


def normalizeData(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df


def create_data(val, num, pca_analysis, pca_hardcode):
    # Set n_components to 0
    n_components = 0
    # Show total memory usage
    print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")

    # Read the parquet file
    path = '../data/final_df.parquet'
    df = df_shrink(read_parquet(path))

    # Show total memory usage
    print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")

    if pca_analysis:
        # Drop the Label column for Random Under Sampling or SMOTE
        X = df_shrink(df.drop('Label', axis=1))  # Features (excluding the 'label' column)

        # Convert the "Label" column to integers (1 for Benign, -1 for Not Benign)
        # Target variable ('label' column)
        y = df['Label'].apply(lambda x: 1 if x == 'Benign' else -1)

        print("X:", X)
        print("y:", y)

        # Normalize data
        X = normalizeData(X)
        """ Dimensionality Reduction using PCA """
        n_components = pca_components(X)

        # Delete X and y
        del X, y
        sleep(5)
        gc.collect()
    elif pca_hardcode:
        n_components = num
    else:
        print("Skipping PCA")

    # Check if the first argument provided is "zero" then call zero_day_main()
    # otherwise call zero_trojan_main()
    if val == "zero":
        # check if the argv argument is "zero"
        # if sys.argv[1] == "zero":
        train, test = create_zero_main(df)  # Call zero_day_main()

        # Release memory
        del df
        sleep(5)
        gc.collect()

        # Show total memory usage
        print("Total memory usage: ", psutil.virtual_memory().used / (1024 ** 2), "MB")

        print("**********************")
        print("Creating Training data ... ")
        print("Training Dataset Shape: ", train.shape)

        X_train = df_shrink(train.drop('Label', axis=1))
        # Convert the "Label" column to integers (1 for Benign, -1 for Not Benign)
        y_train = train['Label'].apply(lambda x: 1 if x == 'Benign' else -1)

        # Balance the training dataset
        X_train, y_train = balance_dataset_specific(X_train, y_train)

        # Normalize the values of the features dataframe using MinMaxScaler
        print("Normalizing Training data ... ")
        X_train = normalizeData(X_train)

        if pca_analysis or pca_hardcode:
            # Apply PCA on the training dataset
            print("Applying PCA on Training data ... ")
            X_train_pca = pca(n_components, X_train)
        else:
            X_train_pca = X_train

        # Convert the pandas series into numpy array
        y_train = y_train.values.ravel()

        print("**********************")
        print("Creating Testing data ... ")
        print("Zeroday Dataset Shape: ", test.shape)

        X_test = df_shrink(test.drop('Label', axis=1))
        # Convert the "Label" column to integers (1 for Benign, -1 for Not Benign)
        y_test = test['Label'].apply(lambda x: 1 if x == 'Benign' else -1)

        # Normalize the values of the features dataframe using MinMaxScaler
        print("Normalizing Testing data ... ")
        X_test = normalizeData(X_test)

        if pca_analysis or pca_hardcode:
            # Apply PCA on the training dataset
            print("Applying PCA on Testing data ... ")
            X_test_pca = pca(n_components, X_test)
        else:
            X_test_pca = X_test

        # Convert the pandas series into numpy array
        y_test = y_test.values.ravel()

    else:
        # THIS PART HAVE TO BE DEVELOPED LATER TO INCLUDE TROJAN DATA ALSO IN THE TESTING DATA
        # AND REMOVE THE TROJAN DATA FROM THE TRAINING DATA TO TEST ZER-DAY ATTACK FOR BOTH
        # ZERODAY AND TROJAN
        # cdf, rdf = create_zero_trojan_main(df)  # Call zero_trojan_main() which creates a new df
        # where Zeroday and Trojan are used combined as zero day.
        # X, y = balance_dataset_specific(cdf)
        return [], [], [], []

    return X_train_pca, y_train, X_test_pca, y_test


def correlations(data):
    """ Implement the correlation coefficient between each feature and the target variable """
    corr = data.corr()['Label'].sort_values(ascending=True)  # Sort the values in ascending order of correlation
    print(corr)
    # Extract the correlation values which are between -0.1 and 0.1
    corr = corr[(corr >= -0.1) & (corr <= 0.1)]
    # create a numpy array from correlations
    feature_names = np.array(corr.index)
    print(feature_names)
    return feature_names


def threshold(explained_variance_ratio):
    """ Implement the threshold method """
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


def elbow(explained_variance_ratio):
    """ Implement Elbow method to find the optimal number of components """
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


def pca_components(X):
    """ Implement PCA and use both threshold and elbow methods to find out the number of components """

    print('Performing PCA ...')
    # Create a PCA object
    model_pca = PCA()
    # Fit PCA on your dataset
    model_pca.fit(X)
    # Get the explained variance ratio / cumulative
    explained_variance_ratio = model_pca.explained_variance_ratio_
    elbow_index = elbow(explained_variance_ratio)
    threshold_index = threshold(explained_variance_ratio)
    if elbow_index <= threshold_index:
        n_components = elbow_index + 1
    else:
        n_components = threshold_index + 1

    print('Number of components:', n_components)
    return n_components


def pca(n_components, X):
    model_pca = PCA(n_components=n_components)
    X_pca = model_pca.fit_transform(X)
    print('PCA done ... ')
    return X_pca
