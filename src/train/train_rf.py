from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import pandas as pd

from pprint import pprint
import matplotlib.pyplot as plt
from src.train.feature_selection import feature_selection

#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def train_rf(train_data, train_labels, val_data, val_labels, test_data, test_labels):

    corr_threshold= 0.6
    print("Feature Selection")
    train_data_uncorr, dropped_features = feature_selection(train_data, corr_threshold)
    dropped_features= np.asarray(dropped_features)
    feature_names = train_data_uncorr.columns.to_numpy()

    val_data_uncorr = val_data.drop(columns=dropped_features)
    test_data_uncorr = test_data.drop(columns=dropped_features)

    #concatenate train and validation data and label
    frames = [train_data_uncorr, val_data_uncorr]
    comb_data = pd.concat(frames, ignore_index= True)
    comb_label = np.append(train_labels, val_labels)

    #Number of trees
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    #Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    #Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #None if max_depth unbounded
    max_depth.append(None)
    #Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    #Create grid vor RandomizedSearchCV
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    #Create grid for GridSearchCV
    grid_search = {'n_estimators': [400, 800, 1200, 1800],
                'max_features': ['sqrt'],
                'max_depth': [10,30,50,70],
                'min_samples_split': [3, 4, 6],
                'min_samples_leaf': [3, 5, 7],
                'bootstrap': [True]}
    
    model = RandomForestClassifier(n_estimators= 400,random_state=0, max_features='sqrt', 
                         max_depth=10,min_samples_leaf= 3, min_samples_split=3, bootstrap= True )

    print("Fitting model")

    # Comment out one model to do RandomiuedSearchCV or GridSearchCV
    #model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=0, n_jobs = -1)
    #model_random = GridSearchCV(estimator=model, param_grid= grid_search, cv = 5, verbose=2, n_jobs= -1)

    #if model_random used, change to model_random.fit
    model.fit(comb_data, comb_label)

    #print("Best Parameters")
    #pprint(model_random.best_params_)

    #comment out if using RandomiuedSearchCV or GridSearchCV
    #print("Best Model")
    #best_model = model_random.best_estimator_
    #pprint(best_model.get_params())

    #change model to best_model if using RandomiuedSearchCV or GridSearchCV
    print("Making prediction")
    pred_labels = model.predict(test_data_uncorr)
    acc = accuracy_score(test_labels, pred_labels)

    #get feature importance values and save in lists
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    x_val = [x[0] for x in feature_importances]
    y_val = [x[1] for x in feature_importances]

    print("Accuracy:", acc)

    #plot feature names and importance values 
    plt.title('Feature Importances')
    plt.barh(range(len(y_val)), y_val, color='#8f63f4', align='center')
    plt.yticks(range(len(x_val)), x_val, size= 3.5, stretch= 'extra-condensed')
    plt.xlabel('Relative Importance')
    plt.show()
