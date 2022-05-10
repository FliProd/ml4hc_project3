import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.models.hyperparameters import params
from config import config
from util import get_training_function, createIdentifier
from src.data.data import get_img_dataset, get_radiomics_dataset



def main():    
    model_name = config['model']
    options = {**config, **params[model_name]}
    model_identifier = createIdentifier(model_name, params)
    options['model_identifier'] = model_identifier
    

    # load and split data into sentences with labels
    print('loading data')
    if model_name in options['image_models']:
        (train_dataset, val_dataset, test_dataset) = get_img_dataset(transform=options['transforms'])
    elif model_name in options['radiomics_models']:
        (train_data, train_labels, val_data, val_labels, test_data, test_labels) = get_radiomics_dataset()
        train_dataset = (train_data, train_labels)
        val_dataset = (val_data, val_labels)
        test_dataset = (test_data, test_labels)
    print('loaded', len(train_dataset), 'training images')
    print('loaded', len(test_dataset), 'test images')

    

    # call training function for model specified in config/config.py
    # pass hyperparameters as named parameters
    print('training model')
    model = get_training_function(model_name)(train_dataset=train_dataset, options=options)
    
    
    # uncomment to evaluate
    evaluate(model=model, test_dataset=test_dataset, model_identifier=model_identifier)
    

# tests the model on the test dataset
def evaluate(model, test_dataset, model_identifier):
    print('evaluating')
    y_true = np.empty((len(test_dataset)))
    y_pred = np.empty((len(test_dataset)))

    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True,  num_workers=4)

    for i, (imgs, labels) in enumerate(test_dataloader):
        y_pred = model.predict(imgs)
        y_true = labels

    acc = accuracy_score(y_true, y_pred)
    print('Accuracy:', acc)



if __name__ == "__main__":
    main()
    