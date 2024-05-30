############################################################################################
# This code is for preprocessing of the CCCS CIC 2020 dataset
# Author: Pal, Anibrata
# Date: 19/12/2023
# Object: Create the compressed parquet file from the CIC 2020 dataset
############################################################################################

# Import statements

import pandas as pd
import numpy as np
import os
# from sklearn.tree import DecisionTreeClassifier
# from fastcore.basics import *
# from fastcore.parallel import *
# from os import cpu_count

# import subprocess

from fastai.tabular.all import df_shrink
import gc
from time import sleep

# read files
# import glob

import csv


# Define a function to read filenames in a directory
def read_filenames_in_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory_path)

    # Display the list of files
    print(f"Files in the directory '{directory_path}':")
    for file_name in files:
        print(file_name)

    return files, directory_path


"""
    Define a function to read CSV files in a directory
    Read from the csv files into a single file.
    Args:
        files (list): A list of file paths for the csv files.
    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from all the csv files.
"""


def file_read(files, directory):
    # Read from the csv files into a single file
    # csv_files = glob.glob('/home/anibrata/Anibrata/PROJECTS/ZERO_DAY_DETECTION/Datasets/
    # CCCS-CIC-AndMal-2020/Static_Analysis/attack/*.csv')

    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Loop through each CSV file and read it into a DataFrame
    for file in files:
        # Create an empty DataFrame
        df1 = pd.DataFrame()
        # Get the full path of the CSV file
        csv_file_path = directory + '/' + file

        # Open the CSV file in read mode
        with open(csv_file_path, 'r') as csv_file:
            # Create a CSV reader object
            csv_reader = csv.reader(csv_file)

            # Read the first row of the CSV file
            header = next(csv_reader)

            # Print the number of features in the CSV file
            num_features = len(header)
            print(f"Number of features in {file}: {num_features}")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(directory + '/' + file, sep=',', encoding='utf-8')
        df = df_shrink(df)
        # Add a header to the DataFrame
        df.columns = [f"F{i}" for i in range(num_features)]
        # print(df.shape)

        # Add a new column with Label Name, which is the attack name
        label_name = os.path.splitext(os.path.basename(file))[0]
        # print(os.path.splitext(os.path.basename(file))[0])

        # Concatenate the label with the DataFrame (to avoid errors,
        # since the warning sign for fragmented DF shows)
        df = pd.concat([df, df1], axis=1)
        if label_name.startswith("Ben"):
            label_name = 'Benign'
        df['Label'] = label_name
        # print(df.shape)
        # print(df)

        # Append the DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        # print(combined_df)

        del df, df1
        sleep(5)
        gc.collect()

    # Display the final df
    # print(combined_df)

    # return combined_df
    return combined_df


""" 
    Data Pre-processing
    Check if:
        1. Any feature has the same values for all rows
        2. Any "nan" values - done
        3. Any categorical values
        4. Any missing values
    Then, find out the unique values to understand the extreme values, and then:
        1. Remove columns/features with the same values
        2. Impute nan with median values
        3. Impute missing values with median values
        4. Remove the categorical features
"""


def resolve_nan(df):
    # Display columns with NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    print("Columns with NaN values:")
    print(nan_columns)

    # Store unique values for columns with NaN values in a dictionary
    unique_values_dict = {}
    for column in nan_columns:
        unique_values = df[column].unique()
        unique_values_dict[column] = unique_values

        # Check if the unique column has values other than NaN
        if np.any(~pd.isna(unique_values)):
            # Check if the columns containing NaN values have more than 20% NaN values
            if df[column].isna().sum() / df.shape[0] > 0.2:
                # Drop the column
                print(f"Dropping column '{column}' because it has more than 20% NaN values.")
                df = df.drop(column, axis=1)
            else:
                # Fill NaN values with median
                print("Fill NaN values with median")
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
    return df


def remove_duplicate(df):
    # Remove duplicate rows
    # duplicate = df.duplicated().sum()
    # if duplicate > 0:
    #     print(df, duplicate, "fully duplicate rows to remove")
    # df = df.drop_duplicates(inplace=True)
    # df.reset_index(inplace=True, drop=True)

    # Identify constant columns
    constant_columns = df.columns[df.nunique() == 1]
    # Remove the column Label from the list
    constant_columns_no_label = constant_columns[constant_columns != 'Label']
    print("Remove Constant Columns: ", constant_columns_no_label)
    # Remove constant columns
    df = df.drop(columns=constant_columns_no_label)
    print(df)
    return df


def remove_categorical(df):
    # Identify categorical columns
    # categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = df.select_dtypes(exclude='number')
    # Remove categorical columns except 'Label'
    categorical_columns = [col for col in categorical_columns if col != 'Label']
    print("Drop Categorical Columns: ", categorical_columns)
    df = df.drop(categorical_columns, axis=1)
    return df


def missing_value(df):
    # Identify missing values
    missing_values = df.isna().sum()
    print(f'missing_values: ', missing_values)
    # Impute missing values with median of the column except in the columns Label
    print("Impute missing values with median of the column")
    for column in df.columns:
        if column != 'Label':
            df[column].fillna(df[column].median(), inplace=True)
    print(df)
    return df


def main():
    files, directory = read_filenames_in_directory('../data/data')
    static_df = file_read(files, directory)
    # Print the loaded df
    print(static_df)

    # Resolve NaN
    static_df = resolve_nan(static_df)
    # Remove categorical columns
    static_df = remove_categorical(static_df)
    # Remove duplicate rows and columns
    static_df = remove_duplicate(static_df)
    # Impute missing values
    static_df = missing_value(static_df)

    # Display the final df
    print(static_df)

    # Check if there is an existing parquet file with name final_df.parquet
    if os.path.exists('../data/final_df.parquet'):
        print("Removing existing parquet file")
        os.remove('../data/final_df.parquet')
    # Save the final df as a parquet file
    static_df.to_parquet('../data/final_df.parquet')

    # Delete static_df
    del static_df
    sleep(5)
    gc.collect()
    print("Data Pre-processing Done, except normalization")


if __name__ == "__main__":
    main()
