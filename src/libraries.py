import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ( 
    accuracy_score, 
    classification_report, 
    confusion_matrix
    roc_auc_score, roc_curve
)

