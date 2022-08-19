# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way
# http://places2.csail.mit.edu/download.html
# https://github.com/CSAILVision/places365
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2          # pip3 install opencv-python, conda install -c conda-forge opencv
from PIL import Image

from functools import partial

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from . import wideresnet
from ..datasets import utils as utils_ds
from . import utils_imagenet

data_folder = 'pred_diff/datasets/Data/places/'

# i)    create places data_folder
# ii)   download files from wget/curl http://data.csail.mit.edu/places/places365/test_256.tar
# iii)  unzip file into test_256 folder


######################################################################################################
# Load data
######################################################################################################
def load_places_data(n_selection=5):
    """Load bird images"""
    image_data_dir = os.getcwd() + '/' + data_folder

    places_test_dataset = datasets.ImageFolder(image_data_dir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    if n_selection == 5:
        index_selection = [1771, 1450, 308, 2618, 340]
    elif n_selection == 25:
        index_selection = [126130, 135158, 247693, 266012, 205593, 173293,  58088, 153721, 53945, 115992, 259191, 92403,
                           220415, 158460, 193886,  68328, 152953, 174840, 281697,  51790,  75818, 279577, 202316,
                           119106, 39156]
    elif n_selection == 50:
        index_selection = [180492,  35279,  90443,  45772,  83986, 282626, 241454, 120155, 188926,  91423, 178522,
                           271300, 296896, 134580, 158505, 327544, 43399, 228189,  67868, 164587, 129323,  41236,
                           260226, 122117, 183644,  68069, 196715,  44665, 262473, 226723, 288311,  63124, 83704,
                           176870, 241959, 285895, 258916, 173469, 172554, 283302, 325024, 286772, 304122, 255282,
                           240379, 57238, 212776, 142363, 78587, 122679]
    else:
        assert False, f'image selection not defined: n_selection = {n_selection}'

    temp = [np.array(places_test_dataset.__getitem__(i)[0]) for i in index_selection]
    imgs_selection = np.stack(temp)

    return places_test_dataset, imgs_selection


######################################################################################################
# Load model
######################################################################################################
# model sources
# https://github.com/CSAILVision/places365 for pytorch 0.2 (official)
# or https://github.com/HuaizhengZhang/scene-recognition-pytorch1.x/blob/master/model_zoo.md for pytorch 1.x (inofficial)

# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_model(device: torch.device):
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(data_folder+model_file, os.W_OK):
        # os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        # os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
        utils_ds.download_url(url='http://places2.csail.mit.edu/models_places365/' + model_file, folder_name='places')
        utils_ds.download_url(url='https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py' + model_file, folder_name='places')

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(data_folder+model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet

    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    model.predict = partial(utils_imagenet.predict, model, device=device, temperature=1.)
    return model, features_blobs


