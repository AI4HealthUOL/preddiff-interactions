import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import pickle
from datetime import datetime
from tqdm import tqdm


from . import utils_imagenet, utils_places
from ..imputers import single_shoot_imputer, general_imputers, color_sampling_imputer
from ..imputers.imputer_base import ImputerBase

from typing import Tuple, Callable


def perform_experiment(n_measurements=50, high_to_low=True, imputer_name='cv2', model_name='places', data_folder='',
                       n_imputations=5):
    """
    Calculates the faithfulness for using attributions for images stored in the data_folder
    # n_measurements = 50  # determines spacing from 0 to 1 percent occluded image
    # high_to_low = True  # start with occluding highest relevance
    """
    print(f'n_imputations = {n_imputations}')
    # load data
    # data_folder = f'data/pixel_flipping/{model_name}/'
    list_path_to_dics = glob.glob(f'{data_folder}2021*')
    assert len(list_path_to_dics) > 0, f'no attributions found: {data_folder}2021*'
    list_dic_data = [pickle.load(open(file_path, 'rb')) for file_path in list_path_to_dics]

    splitted_datafolder = data_folder.split('/')
    if model_name == 'places':
        assert splitted_datafolder[-2][:6] == 'places'
        model, features_blobs = utils_places.load_model(device=torch.device('cpu'))
        dataset_img, imgs_selection = utils_places.load_places_data()
        rng = np.random.default_rng()
        index_selection = rng.choice(dataset_img.__len__(), 1000, replace=False)  # replace=False: unique indices
        temp = [np.array(dataset_img.__getitem__(i)[0]) for i in index_selection]
        imgs_train = np.array(temp)
    elif model_name == 'cub':
        assert splitted_datafolder[-2][:3] == 'cub'
        import argparse
        args = argparse.Namespace()
        args.model_cuda_device = 1
        args.max_images = -1
        args.model_checkpoint = "../checkpoints/vgg_caltech_full_net_trained.pth"
        imgs_test, imgs_train, imgs_selection = utils_imagenet.load_vgg16_cub200_data(max_images=args.max_images)
        model = utils_imagenet.load_vgg16_cub200_model(args=args)
    else:
        assert False, 'not implemented yet'

    n_image = 5  # how many images are measured
    # PredDiff
    if imputer_name == 'TrainSet':
        imputer = general_imputers.TrainSetImputer(train_data=imgs_train)
    elif imputer_name == 'GaussianNoise':
        sigma = imgs_train[:300].std(axis=0)
        imputer = general_imputers.GaussianNoiseImputer(train_data=imgs_train, sigma=sigma)
    elif imputer_name == "Histogram":
        imputer = color_sampling_imputer.ColorHistogramImputer(train_data=imgs_train)
    elif imputer_name == 'cv2_telea':
        imputer = single_shoot_imputer.OpenCVInpainting(inpainting_algorithm='telea')
    else:
        assert False, f'incorrect imputer argument: {imputer_name}'

    dic_pixelflipping = {'n_measurements': n_measurements,
                         'imputer_occluding': imputer.imputer_name, }

    list_all_imputers = []
    for dic in tqdm(list_dic_data, desc='imputers'):  # loop all different attributions methods
        attribution_imputer = dic['args'].imputer
        list_all_imputers.append(attribution_imputer)
        for i_image in range(n_image):  # measure all images
            probability, x = faithfulness_image(predict_probability_func=model.predict, imputer=imputer,
                                                image=dic[f'image_{i_image}'],
                                                attribution=dic[f'relevance_{i_image}'],
                                                segmentation=dic[f'seg_{i_image}'],
                                                n_measurements=n_measurements,
                                                high_to_low=high_to_low,
                                                n_imputations=n_imputations)
            dic_pixelflipping[f'{attribution_imputer}_probability_{i_image}'] = probability
            dic_pixelflipping[f'{attribution_imputer}_percentage_occluded_{i_image}'] = x

    dic_pixelflipping['imputer_attributions'] = list_all_imputers

    dic_pixelflipping['n_images'] = n_image
    dic_pixelflipping['high_to_low'] = high_to_low
    dic_pixelflipping['n_imputations'] = n_imputations

    # store results by date, time and arguments
    now = datetime.now()
    dic_pixelflipping['now'] = f"{now.date()}_{now.strftime('%H%M')}"
    file_name = f"pixelflipping_{model_name}_{imputer.imputer_name}_{n_image}"
    if high_to_low is False:
        file_name += "_False"

    save_dir = data_folder
    save_file = save_dir + file_name
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(dic_pixelflipping, open(save_file, 'wb'))


def faithfulness_image(predict_probability_func: Callable, imputer: ImputerBase,
                       image: np.ndarray, attribution: np.ndarray, segmentation: np.ndarray,
                       n_measurements=10, n_imputations=5, high_to_low=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates faithfulness for a single image based on superpixels.
    high_to_low = True      # if True occlude high relevance pixels first
    ---
    RETURN
    probability: prediction for all different occlusions
    """
    if imputer.imputer_name == 'cv2_telea':
        n_imputations = 1
    # only uses a single imputation per data point
    n_segments = np.max(segmentation) + 1

    # check input
    assert n_segments == len(attribution), 'attribution does not fit to segmentation'
    assert image.shape[1:] == segmentation.shape, 'image does not fit to segmentation'

    # setup varaibles
    n_measurements = n_measurements if n_segments > n_measurements else n_segments
    if n_measurements > n_segments/4:           # dense spacing due to many measurements
        # linear spacing of measurements
        n_superpixels_occluded = np.linspace(0, n_segments, num=n_measurements, dtype=np.int)
    else:
        # log-spacing for more accurate analysis close to the full image
        n_superpixels_occluded = np.unique(np.geomspace(1, n_segments, num=n_measurements, dtype=np.int))
        n_superpixels_occluded = np.concatenate([[0], n_superpixels_occluded])

    arg_attribution_sorted = np.argsort(attribution)
    if high_to_low is True:
        arg_attribution_sorted = arg_attribution_sorted[::-1]       # start with highest attributions

    # infer baseline prediction
    with torch.no_grad():
        base_prediction = predict_probability_func(image[np.newaxis])
    predicted_class = int(base_prediction[0].argmax())
    predicted_probability = base_prediction[0, predicted_class]

    # generate imputations, n_imputations=1 per measurement
    list_imputations = []
    for n_next in n_superpixels_occluded[1:]:       # iterative include more superpixels and store mask
        if n_next == n_segments:        # never occlude the complete image, one superpixel remains
            n_next -= 1
        mask = np.array([segmentation == s for s in arg_attribution_sorted[:n_next]]).sum(axis=0)
        mask_boolean = np.array(mask, dtype=np.bool)
        mask_temp = np.tile(mask_boolean, (3, 1, 1))
        if imputer.imputer_name == "TrainSet":
            imputer.seg = segmentation
        imputations_temp, weights = imputer.impute(test_data=image[np.newaxis], mask_impute=mask_temp, n_imputations=n_imputations)
        imputations_temp = imputations_temp[:, 0]       # n_imputations, n_images=1
        image_imputed = np.tile(image.copy(), (n_imputations, 1, 1, 1))
        image_imputed[:, mask_temp] = imputations_temp[:, mask_temp]
        list_imputations.append(image_imputed)

    imputations = np.array(list_imputations)        # n_measurements, n_imputations, n_images=1,...

    # infer all predictions at once with batch_size
    with torch.no_grad():
        list_imputed_predictions = []
        for i in range(n_imputations):
            imputed_predictions = predict_probability_func(imputations[:, i])
            list_imputed_predictions.append(imputed_predictions)

    imputed_probability = np.array(list_imputed_predictions)[:, :, predicted_class].mean(axis=0)
    # imputed_probability = imputed_predictions[:, predicted_class]

    probabilities = np.concatenate((np.array(predicted_probability), imputed_probability), axis=None)
    percentage_occluded = n_superpixels_occluded/n_segments

    return probabilities, percentage_occluded
