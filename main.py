import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch
import shap

from src.models.hyperparameters import params
from config import config
from util import get_training_function, createIdentifier
from src.data.data import get_img_dataset, get_radiomics_dataset



def main():    
    model_name = config['model']
    options = {**config, **params[model_name]}
    model_identifier = createIdentifier(model_name, params)
    options['model_identifier'] = model_identifier
    radiomics = False

    # load and split data into sentences with labels
    print('loading data')
    if model_name in options['image_models']:
        (train_dataset, val_dataset, test_dataset) = get_img_dataset(transform=options['transforms'], pretrain=options['pretrain'])
    elif model_name in options['radiomics_models']:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = get_radiomics_dataset()
        train_dataset = (train_data, train_labels)
        val_dataset = (val_data, val_labels)
        test_dataset = (test_data, test_labels)
        radiomics = True

    print('loaded', len(train_dataset), 'training images')
    print('loaded', len(test_dataset), 'test images')

    
    if radiomics :
        get_training_function(model_name)(train_data, train_labels, val_data, val_labels, test_data, test_labels)
        
    
    else:
        # call training function for model specified in config/config.py
        # pass hyperparameters as named parameters
        print('training model')
        model = get_training_function(model_name)(train_dataset=train_dataset, options=options)
        
        
        # uncomment to evaluate
        evaluate(model=model, test_dataset=test_dataset, model_identifier=model_identifier)

        #uncomment for shap values
        shap_explainer(train_dataset, test_dataset, model)
    

# tests the model on the test dataset
def evaluate(model, test_dataset, model_identifier):
    print('evaluating')
    y_true = np.empty((len(test_dataset)))
    y_pred = np.empty((len(test_dataset)))

    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True,  num_workers=4)

    for i, (imgs, labels) in enumerate(test_dataloader):
        if config['model'] == 'cnn':
            preds = model(imgs)
            x, y_pred = torch.max(preds, dim=1)
        else:
            y_pred = model.predict(imgs)
        y_true = labels
        print(y_true)
        print(y_pred)

    acc = accuracy_score(y_true, y_pred)
    print('Accuracy:', acc)

def shap_explainer(train_dataset, test_dataset, model):
    bg = []
    for i,j in train_dataset:
        bg.append(i)
    bg = torch.stack(bg)
    e = shap.DeepExplainer(model, bg)

    for i, (image,target) in enumerate(test_dataset):
        image = image.reshape((1,3,128,128))

        preds = model(image)
        pred, out = torch.max(preds, dim=1)
        
        #https://github.com/slundberg/shap/issues/1243
        shap_values = e.shap_values(image)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(image.cpu().numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, test_numpy, labels = ["SHAP for class No","SHAP for class Yes"])



if __name__ == "__main__":
    main()
    