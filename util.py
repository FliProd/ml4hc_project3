from src.train.train_beta_vae import train_beta_vae
from scr.train.train_rf import train_rf
from scr.train.train_svm import train_svm

def get_training_function(model_name):
    if model_name == 'BetaVAE':
        return train_beta_vae
    elif model_name == 'RandomForest':
        return train_rf
    elif model_name == 'SVM':
        return train_svm
      


def createIdentifier(model_name, params):
    return model_name + str(params[model_name])
