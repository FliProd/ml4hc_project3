from torchvision import transforms


"""
Training and evaluation settings
"""
config = dict()

"""
Data related settings 
"""
config['pos_img_dir'] = 'data/images/yes/'
config['neg_img_dir'] = 'data/images/no/'
config['label_dir'] = 'data/radiomics/'

config['transforms'] = [transforms.RandomRotation(90), transforms.RandomHorizontalFlip(), transforms.RandomPerspective(0.2), transforms.RandomAffine(45)]

config['batch_size'] = 32


"""
Model related settings 
Available models: BetaVAE, RandomForest, SVM
"""
config['model'] = 'SVM'

config['pytorch_models'] = ['BetaVAE']

config['image_models'] = ['BetaVAE']
config['radiomics_models'] = ['RandomForest','SVM']


"""
Path to serialized models & figures & stuff
"""
config['saved_model_path'] = 'models/'
config['figure_path'] = 'reports/figures/'
config['report_path'] = 'reports/'


"""
Other
"""
config['generate_beta_vae_umap_plot'] = True

# first do a run with pretrain true and finetune false to pretrain, then vice versa to fine tune
config['pretrain'] = False
config['finetune'] = False
