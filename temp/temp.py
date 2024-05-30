import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def test_impute():
        # Example DataFrame with missing values
        data = {'A': [1, 2, None, 4],
                'B': [5, None, 7, 8],
                'C': [9, 10, 11, None]}

        df = pd.DataFrame(data)

        # Display the original DataFrame
        print("Original DataFrame:")
        print(df)

        # Find missing values
        missing_values = df.isna()

        # Impute missing values with median
        df_imputed = df.fillna(df.median())

        # Display the DataFrame after imputation
        print("\nDataFrame after imputing missing values with median:")
        print(df_imputed)


def test_string_start():
        # Example string
        my_string = "BenignExample"

        # Check if the string starts with "Ben"
        if my_string.startswith("Ben"):
                print("The string starts with 'Ben'")
        else:
                print("The string does not start with 'Ben'")


def change_label():

        # Example DataFrame
        data = {'Feature1': [1, 2, 3, 4],
                'Feature2': [5, 6, 7, 8],
                'Label': ['Benign', 'bills', 'Benign', 'balls']}

        df = pd.DataFrame(data)

        # Display the original DataFrame
        print("Original DataFrame:")
        print(df)

        # Change labels to 'Malicious' where the values are not 'Benign'
        df.loc[df['Label'] != 'Benign', 'Label'] = 'Malicious'

        # Display the DataFrame after changing labels
        print("\nDataFrame after changing labels:")
        print(df)


def change_label2():
        # Assuming your DataFrame is named df and the "category" column is named "label"
        # You can replace these names with your actual DataFrame and column names

        # Creating a sample DataFrame
        data = {'feature1': [1, 2, 3, 4],
                'label': ['Benign', 'DoS', 'Backdoor', 'DoS']}
        df = pd.DataFrame(data)

        # Set "Benign" to 1 and the rest to -1 dynamically
        unique_categories = df['label'].unique()
        category_mapping = {category: 1 if category == 'Benign' else -1 for category in unique_categories}
        df['label'] = pd.Categorical(df['label'], categories=category_mapping.keys()).codes

        # Display the modified DataFrame
        print(df)

def conf_matrix():
     # Generating a sample DataFrame
     data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'label': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]}  # Assuming binary classification (1 and 0)
     df = pd.DataFrame(data)

     # Splitting the DataFrame into features (X) and labels (y)
     X = df[['feature1']]
     y = df['label']

     # Splitting the data into training and testing sets
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Creating a RandomForestClassifier (you can replace this with your classifier)
     model = RandomForestClassifier(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)

     # Defining class names (you can replace this with your actual class names)
     class_names = ['Class 0', 'Class 1']

     # Plot non-normalized confusion matrix
     titles_options = [
         ("Confusion matrix, without normalization", "true")
     ]
     for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=class_names,
            cmap='Blues',
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

     plt.savefig('conf_matrix.png')


def split():
    # Generating dummy data
    np.random.seed(42)
    X = np.random.rand(100, 3)  # 100 samples with 3 features each
    y = np.random.randint(0, 2, size=100)  # Binary target variable

    # Using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the shapes of the resulting arrays
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # datatype of the above mentioned variables
    print(type(X_train), type(X_test), type(y_train), type(y_test))


def cfmatrix():
    # Example data
    y_true = np.array([1, 0, 1, 2, 0, 1, 2, 0, 2])
    y_pred = np.array([1, 0, 1, 2, 0, 2, 1, 0, 2])

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print(cm)

    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    # test_impute()
    # test_string_start()
    # change_label()
    # change_label2()
    # conf_matrix()
    # split()
    cfmatrix()


if __name__ == "__main__":
    main()
