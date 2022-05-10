from src.train.train_beta_vae import train_beta_vae

def get_training_function(model_name):
    if model_name == 'BetaVAE':
        return train_beta_vae


def createIdentifier(model_name, params):
    return model_name + str(params[model_name])
