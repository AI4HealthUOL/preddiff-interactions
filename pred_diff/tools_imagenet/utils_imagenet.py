import numpy as np
import os

import pickle
from torchvision import transforms, models, datasets
import torch
from functools import partial
import shutil
import tarfile
import code_base.RAP.vgg as models_rap

from skimage.segmentation import felzenszwalb, slic

from ..datasets import utils as utils_dataset
from ..imputers.imputer_base import ImputerBase

######################################################################################################
# Generate image masks
#####################################################################################################
def generate_superpixel_felzenszwalb(image: np.ndarray, n_segments=100):
    assert len(image.shape) == 3, 'provide only a single image'
    n_pixel = image.shape[1]**2
    min_segment_size = int(n_pixel/n_segments/3)
    seg = felzenszwalb(image.transpose(1, 2, 0), min_size=min_segment_size, scale=0.01)
    list_of_masks = [np.tile(seg == s, (3, 1, 1)) for s in range(seg.max())]
    return list_of_masks, seg


def generate_superpixel_slic(image: np.ndarray, n_segments=100):
    """image: shape: (3, n_pixel, n_pixel)"""
    assert len(image.shape) == 3, 'provide only a single image'
    # increase compactness to produce more squared superpixel
    seg = slic(image.transpose(1, 2, 0), n_segments=n_segments, compactness=20)
    list_of_masks = [np.tile(seg == s, (3, 1, 1)) for s in range(seg.max()+1)]
    return list_of_masks, seg


######################################################################################################
# Load data
######################################################################################################
# Organize image data
# from https://github.com/ecm200/caltech_birds/blob/master/notebooks/NABirds_Images_Dir_Sorting.ipynb

def load_train_test_split(dataset_path=''):
    train_images = []
    test_images = []
    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train:
                train_images.append(image_id)
            else:
                test_images.append(image_id)
    return train_images, test_images


def load_class_names(dataset_path=''):
    names = {}
    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])
    return names


def load_image_labels(dataset_path=''):
    labels = {}
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id
    return labels


def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path
    return paths


def load_vgg16_cub200_data(n_selection=5, download=True, organize_data=True, data_dir=None, return_targs=False):
    """Load bird images"""
    image_data_dir = os.getcwd() + '/pred_diff/datasets/Data/CUB_200_2011/' if data_dir is None else data_dir
    try:
        # data is already preprocessed
        test_data_file = image_data_dir + 'cub_test'
        with open(test_data_file, 'rb') as outfile:
            imgs_test = pickle.load(outfile)
        if return_targs:
            test_targs_file = image_data_dir + 'cub_test_targs'
            with open(test_targs_file, 'rb') as outfile:
                targs_test = pickle.load(outfile)
        with open(image_data_dir + 'cub_train', 'rb') as outfile:
            imgs_train = pickle.load(outfile)
        if return_targs:
            with open(image_data_dir + 'cub_train_targs', 'rb') as outfile:
                targs_train = pickle.load(outfile)
        if n_selection == 5:
            # with open(image_data_dir + 'cub_selection', 'rb') as outfile:
            #     imgs_selection = pickle.load(outfile)
            index_selection = [1771, 1450, 308, 2618, 340]
        elif n_selection == 25:
            index_selection = [160, 3195, 2886, 4396, 2343, 5546, 3054, 3269, 3154, 1260,  882, 3235, 2007,  447, 4616,
                               4543,  597, 5171, 4949, 2847, 4705, 4345, 92, 5651, 3903]
        elif n_selection == 50:
            index_selection = [1914, 3270, 3469,  160, 4291, 2298, 3276, 4468, 937, 1981, 2523, 5738,  529, 4970, 1971,
                               3267, 4734, 5556, 3241, 1009, 2392, 2846, 4985, 2067, 1373, 5240, 908, 1706, 3647, 3773,
                               663, 5327, 3973, 3733, 1127, 2449, 2033, 2814, 1667, 4665, 1438, 5775,  366,  903, 1177,
                               3803, 1704, 4364, 3454, 2028]
        else:
            assert False, f'image selection not defined: n_selection = {n_selection}'
        imgs_selection = imgs_test[index_selection]
        if return_targs:
            targs_selection = targs_test[index_selection]

    # set-up data from scratch
    except FileNotFoundError:
        if download is True:
            print(f'download CUB_200_2011 to {image_data_dir}')
            # if turned off, put the tar file into the image_data_dir
            # Download "All images and annotations" from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
            root_data_dir = image_data_dir + "CUB_200_2011/"

            url = 'https://tubcloud.tu-berlin.de/s/i7nMWipCPqKcAF4/download'
            utils_dataset.download_url(url=url, folder_name='CUB_200_2011')     # file name 'download'
            file_download = f'{image_data_dir}download'
            # uncompress .tar.gz file, 'download'
            with tarfile.open(file_download) as file:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(file, image_data_dir)
            os.remove(file_download)

        if organize_data is True:
            root_data_dir = image_data_dir + "CUB_200_2011/"
            # Expects unpacked data in image_data_dir/CUB_200_2011/
            # Organize image data
            # from https://github.com/ecm200/caltech_birds/blob/master/notebooks/NABirds_Images_Dir_Sorting.ipynb
            print('Start reorganizing CUB files...')
            train, test = load_train_test_split(dataset_path=root_data_dir)
            labels = load_image_labels(dataset_path=root_data_dir)
            image_paths = load_image_paths(dataset_path=root_data_dir)

            old_images_dir = 'images'
            new_images_dir = 'images_sorted'

            images_train_dir = os.path.join(root_data_dir, new_images_dir, 'train')
            images_test_dir = os.path.join(root_data_dir, new_images_dir, 'test')

            os.makedirs(os.path.join(root_data_dir, new_images_dir), exist_ok=True)
            os.makedirs(images_train_dir, exist_ok=True)
            os.makedirs(images_test_dir, exist_ok=True)

            for image in train:
                new_dir = os.path.join(images_train_dir, image_paths[image].split('/')[0])
                old_path_image = os.path.join(root_data_dir, old_images_dir, image_paths[image])
                new_path_image = os.path.join(images_train_dir, image_paths[image])
                os.makedirs(new_dir, exist_ok=True)
                # os.symlink(old_path_image, new_path_image)
                shutil.move(old_path_image, new_path_image)

            for image in test:
                new_dir = os.path.join(images_test_dir, image_paths[image].split('/')[0])
                old_path_image = os.path.join(root_data_dir, old_images_dir, image_paths[image])
                new_path_image = os.path.join(images_test_dir, image_paths[image])
                os.makedirs(new_dir, exist_ok=True)
                # os.symlink(old_path_image, new_path_image)
                shutil.move(old_path_image, new_path_image)

        print('Preprocessing files...')
        # preprocess and store in pickle file
        T = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Lambda(lambda x: np.array(x).transpose(2, 0, 1))])
        # expects images to be organized in folders corresponding to classes
        data = datasets.ImageFolder(image_data_dir + 'CUB_200_2011/images_sorted/test/', T)
        # load all images to pass them to preddiff
        imgs_test = []
        targs_test = []
        for item in data:
            imgs_test.append(item[0])
            targs_test.append(int(item[1]))
        imgs_test = np.stack(imgs_test)
        targs_test = np.array(targs_test)
        with open(image_data_dir + 'cub_test', 'wb') as outfile:
            pickle.dump(imgs_test, outfile)
        with open(image_data_dir + 'cub_test_targs', 'wb') as outfile:
            pickle.dump(targs_test, outfile)

        # fixed selection generated with np.random.random_integers(0, imgs_test.shape[0], 5)
        index_selection = [1771, 1450,  308, 2618,  340]
        imgs_selection = imgs_test[index_selection]
        targs_selection = targs_test[index_selection]
        with open(image_data_dir + 'cub_selection', 'wb') as outfile:
            pickle.dump(imgs_selection, outfile)
        with open(image_data_dir + 'cub_selection_targs', 'wb') as outfile:
            pickle.dump(targs_selection, outfile)

        data = datasets.ImageFolder(image_data_dir + 'CUB_200_2011/images_sorted/train/', T)
        # load all images to pass them to preddiff
        imgs_train = []
        targs_train = []
        for item in data:
            imgs_train.append(item[0])
            targs_train.append(int(item[1]))
        imgs_train = np.stack(imgs_train)
        targs_train = np.array(targs_train)
        with open(image_data_dir + 'cub_train', 'wb') as outfile:
            pickle.dump(imgs_train, outfile)
        with open(image_data_dir + 'cub_train_targs', 'wb') as outfile:
            pickle.dump(targs_train, outfile)
    
    if return_targs:
        return imgs_test, imgs_train, imgs_selection, targs_test, targs_train, targs_selection
    else:
        return imgs_test, imgs_train, imgs_selection


def load_imagenet(n_selection=5):
    """Load imagenet subset based on CIFAR-10 classes. """
    # imagenet NILS
    data_folder = 'pred_diff/datasets/Data/imagenet_samples/'
    image_data_dir_val = os.getcwd() + '/' + data_folder + 'val/'
    image_data_dir_train = os.getcwd() + '/' + data_folder + 'train/'

    val_dataset = datasets.ImageFolder(image_data_dir_val, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    train_dataset = datasets.ImageFolder(image_data_dir_train, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    if n_selection == 5:
        index_selection = [86, 98, 66, 87, 84]
    elif n_selection == 25:
        index_selection = [68, 38, 16,  5,  7, 87, 60, 20, 23, 98,  1, 84,  0, 49, 58, 30, 33, 56, 24, 78, 42, 82, 89,
                           76, 57]
    elif n_selection == 50:
        index_selection = [6, 87, 10, 73, 35, 36, 77, 39, 80, 27, 64, 94, 21, 16, 68, 48, 56, 17, 85, 24, 54, 89, 90,
                           40, 41,  2, 12, 57, 74, 42, 31, 86, 55, 14, 95, 52, 13, 79, 33, 19, 78, 49,  7, 93, 88, 99,
                           9, 65, 91, 44]
    else:
        assert False, f'image selection not defined: n_selection = {n_selection}'

    temp = [np.array(val_dataset.__getitem__(i)[0]) for i in index_selection]
    imgs_selection = np.stack(temp)

    rng = np.random.default_rng()
    index_selection_train = rng.choice(train_dataset.__len__(), 1000, replace=False)  # replace=False: unique indices
    temp = [np.array(train_dataset.__getitem__(i)[0]) for i in index_selection_train]
    imgs_train = np.stack(temp)

    return imgs_train, imgs_selection


######################################################################################################
# Load model
######################################################################################################
def predict(self, test_data, temperature=1., device=None):
    # normalization used for training of pretrained torchvision models
    if isinstance(test_data, torch.Tensor) is False:
        test_data = torch.Tensor(test_data)
    if test_data.max() > 1:
        test_data = test_data.float() / 255.0
    else:
        test_data = test_data.float()

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    test_data = (test_data - mean) / std

    test_data = test_data.to(device)

    with torch.no_grad():
        logits = self.forward(test_data)

    preds = torch.softmax(logits / temperature, dim=1)
    preds = preds.cpu().numpy()
    test_data = test_data.cpu()
    return preds


def load_vgg16_cub200_model(args, temperature):
    device = torch.device("cuda:{}".format(args.model_cuda_device) if args.model_cuda_device != -1 else "cpu")
    if torch.cuda.is_available() is False:
        print('no cuda available')
        device = torch.device('cpu')

    args.model = "vgg16_cub200"
    assert args.model in ["vgg16_imagenet", "vgg16_cub200"], "model not available"

    if args.model == "vgg16_imagenet":
        model = models.vgg16(pretrained=True).to(device)
    elif args.model == "vgg16_rap":
        model = models_rap.vgg16(pretrained=True).to(device)    
    elif args.model.startswith("vgg16_cub200"):
        if(args.model == "vgg16_cub200_rap"):
            model = models_rap.vgg16()
        else:
            model = models.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, 200)
        try:
            model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        except:
            print(f'download cub model to {args.model_checkpoint}')
            # download model from tu cloud
            url = 'https://tubcloud.tu-berlin.de/s/A2g5mWzYGdAnky5/download'
            utils_dataset.download_url(url=url, folder_name='vgg_model')  # file name 'download'
            old_file_name = os.getcwd() + '/pred_diff/datasets/Data/vgg_model/download'
            new_dir = '../checkpoints/'
            new_file_name = new_dir + 'vgg_caltech_full_net_trained.pth'
            os.makedirs(new_dir, exist_ok=True)
            # os.symlink(old_path_image, new_path_image)
            shutil.move(old_file_name, new_file_name)
            os.rmdir(os.getcwd() + '/pred_diff/datasets/Data/vgg_model/')

        model = model.to(device)

    model.eval()

    model.predict = partial(predict, model, device=device, temperature=temperature)
    return model


######################################################################################################
# routines
######################################################################################################
def visualize_imputations(imputer: ImputerBase, image_selection: np.ndarray, superpixel='slic', n_segments=50):
    n_imputations = 5
    dic_imputations = {'n_example_imputations': n_imputations,
                       'n_segments': n_segments}
    for i, image in enumerate(image_selection):
        # superpixel
        if superpixel=='slic':
            list_of_masks, seg = generate_superpixel_slic(image=image, n_segments=n_segments)
        elif superpixel=='felzenszwalb':
            list_of_masks, seg = generate_superpixel_felzenszwalb(image=image, n_segments=n_segments)

        mask_superpixel = list_of_masks[int(len(list_of_masks) / 2)]

        # prepare imputations
        imputations, _ = imputer.impute(test_data=image[np.newaxis], mask_impute=mask_superpixel, n_imputations=n_imputations)

        image_imputed = np.array([image[np.newaxis].copy() for _ in range(n_imputations)])
        image_imputed[:, :, mask_superpixel] = imputations[:, :, mask_superpixel]
        dic_imputations.update({f'image_original_{i}': image,
                                f'image_imputed_{i}': image_imputed,
                                'imputer': imputer.imputer_name,
                                f'segments_{i}': seg,
                                f'mask_{i}': mask_superpixel
                                })
    return dic_imputations
