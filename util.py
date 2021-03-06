from src.train.train_beta_vae import train_beta_vae
from src.train.train_rf import train_rf
from src.train.train_svm import train_svm
from src.train.train_cnn_baseline import train_cnn

import numpy as np

def get_training_function(model_name):
    if model_name == 'BetaVAE':
        return train_beta_vae
    elif model_name == 'RandomForest':
        return train_rf
    elif model_name == 'SVM':
        return train_svm
    elif model_name == "cnn":
        return train_cnn
      


def createIdentifier(model_name, params):
    return model_name + str(params[model_name])
