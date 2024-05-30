import sys
import os
import numpy as np
import pandas as pd
import time
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.manifold import TSNE
import seaborn as sns
import gc

# import methods from classifyMain program
from classifyMain import *

pd.options.display.max_columns = None

kernel_mode = True
if kernel_mode:
    sys.path.insert(0, "../iterative-stratification")

sns.set(style="darkgrid")
gc.enable()
rand_seed = 1120

# Make ready data for ML
x_train_prep, x_test_prep, y_train_prep, y_test_prep, data_dimension = makeData()

# print(x_train_prep.shape)
# print(x_test_prep.shape)
# print(y_train_prep.shape)
# print(y_test_prep.shape)

# DeepInsight Transform - t-SNE 2D Embeddings
# Based on https://github.com/alok-ai-lab/DeepInsight, but with some corrections to the norm-2 normalization.

