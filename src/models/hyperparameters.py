
params = {
    'BetaVAE': {
        'beta': 7,
        'z_dim': 10,
        'reconstruction_loss_distr': 'gaussian',
        'loss_reduction': 'sum',
        'classifier': 'SVM',
        #'pretrained': True,
        'classifier_options': {
            'SVM': {'kernel':'rbf', 'gamma':'auto'},
            #'RFS': {'n_estimators': 50},
            #'KNN': {'n_neighbors': 2, 'weights': 'distance'}
        }
    }
}