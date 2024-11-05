import numpy as np
import pandas as pd

# Classifiers
from sklearn import ensemble, model_selection, metrics
import lightgbm
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def trainer(train_x, train_y, model_name, seed):

    classifier = None
    if model_name == 'LR':
        classifier = LogisticRegression(n_jobs=-1, random_state=seed)
    elif model_name == 'RF':
        classifier = RandomForestClassifier(n_jobs=-1, random_state=seed)
    elif model_name == 'NB':
        classifier = GaussianNB()
    elif model_name == 'SVC':
        classifier = SVC(probability=True, random_state=seed)
    elif model_name == 'DT':
        classifier = DecisionTreeClassifier(random_state=seed)
    elif model_name == 'KNN':
        classifier = KNeighborsClassifier()


    classifier.fit(train_x, train_y)

    return classifier
