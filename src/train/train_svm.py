import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def feature_selection(df, corr_threshold=0.8):
    
    print("Remove correlated features...")

    # Check corr_threshold
    if type(corr_threshold) is float:
        if 0 < corr_threshold and corr_threshold < 1:
            print("Correlation threshold:", corr_threshold)
        else:
            print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
            return -1
    else:
        print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
        return -1
    
    # Get all features
    all_features = df.columns.to_list()
    n_features   = df.shape[1]

    # Compute correlation matrix
    corr = df.corr(method='pearson')
    corr = corr.abs()

    # Keep only correlation values in the upper traingle matrix
    triu = np.triu(np.ones(corr.shape), k=1)
    triu = triu.astype(bool)
    corr = corr.where(triu)
    
    # Select columns which will be dropped from dataframe
    cols_to_drop = [column for column in corr.columns if any(corr[column] > corr_threshold)]

    n_cols_to_drop = len(cols_to_drop)
    p_cols_to_drop = 100 * (float(n_cols_to_drop) / float(n_features))
    p_cols_to_drop = np.round(p_cols_to_drop, decimals=1)
    print("Drop", n_cols_to_drop, "/", n_features, " features (", p_cols_to_drop, "%).")

    # Drop colums
    df = df.drop(cols_to_drop, axis=1)
    
    # Find names of features which have been dropped
    uncorrelated_features = df.columns.to_list()
    dropped_features      = list(set(all_features) - set(uncorrelated_features))
    
    return df, dropped_features

def train_svm(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    #set corraletion threshold
    corr_threshold= 0.8

    #feature selection
    print("Feature Selection")
    train_data_uncorr, dropped_features = feature_selection(train_data, corr_threshold)
    dropped_features= np.asarray(dropped_features)
    feature_names = train_data_uncorr.columns.to_numpy()

    #apply feature selection for validation and test datasets
    val_data_uncorr = val_data.drop(columns=dropped_features)
    test_data_uncorr = test_data.drop(columns=dropped_features)

    #concatenate train and validation data and label
    frames = [train_data_uncorr, val_data_uncorr]
    comb_data = pd.concat(frames, ignore_index= True)
    comb_label = np.append(train_labels, val_labels)

    print("Train")
    #set model
    clf = SVC(kernel = 'rbf', random_state = 0,probability=True,)
    #train model
    clf.fit(comb_data, comb_label)

    #predict class probability and select the higher one
    pred_labels_prob = clf.predict_proba(test_data_uncorr)
    pred_labels = np.argmax(pred_labels_prob, axis = 1)

    #calculate accuracy
    acc = accuracy_score(test_labels, pred_labels)
    print("predict", pred_labels)
    print(acc)

    #Plot permutation importance of the features
    perm_importance = permutation_importance(clf, test_data_uncorr, test_labels, n_repeats=30)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.title('Feature Importances')
    plt.barh(range(len(feature_names)), perm_importance.importances_mean[sorted_idx])
    plt.yticks(range(len(feature_names)), feature_names[sorted_idx], size= 3.5, stretch= 'extra-condensed')
    plt.xlabel("Permutation Importance")
    plt.show()
    
    return acc