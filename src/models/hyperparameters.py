
params = {
    'BetaVAE': {
        'beta': 5,
        'z_dim': 10,
        'reconstruction_loss_distr': 'bernoulli',
        'loss_reduction': 'sum',
        'classifier': 'SVM',
        'classifier_options': {
            'SVM': {'kernel':'rbf', 'gamma':'auto'}
        }
    }
}