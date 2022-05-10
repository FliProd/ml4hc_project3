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
Available models: BetaVAE
"""
config['model'] = 'BetaVAE'

config['pytorch_models'] = ['BetaVAE']

config['image_models'] = ['BetaVAE']
config['radiomics_models'] = []


"""
Path to serialized models & figures & stuff
"""
config['saved_model_path'] = 'models/'
config['figure_path'] = 'reports/figures/'
config['report_path'] = 'reports/'