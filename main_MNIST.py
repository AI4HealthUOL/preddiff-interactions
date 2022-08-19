import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

from datetime import datetime
import argparse
import glob

import functools

from skimage.segmentation import slic
import torch

from pred_diff import preddiff, shapley
from pred_diff.imputers import general_imputers, color_sampling_imputer, single_shoot_imputer, vae_impute
from pred_diff.imputers.imputer_base import ImputerBase
from pred_diff.tools import utils_mnist as ut_mnist
from pred_diff.tools import init_plt as init_plt




def old_main_20Jan():
    model, df_train, df_test, target_test = ut_mnist.get_model_and_data(max_epochs=20, retrain=False)
    # model.T = 1.21
    # from importlib import reload
    # reload(preddiff)
    n_imputations = 10

    print(f"{model.T = }")

    iparam = ut_mnist.ImgParams(n_pixel=28, block_size=1)

    imputer = general_imputers.TrainSetImputer(train_data=df_train.to_numpy())

    # imputer = vae_impute.VAEImputer(df_train=df_train, epochs=20)

    def df_round_clip(df):
        return df.clip(lower=0, upper=1).div(1. / 255).round().div(255.)

    # imputer = vae_impute.VAEImputer(train_data=df_train, pus=0, epochs=2, custom_postprocessing_fn=df_round_clip)
    pd_explainer = preddiff.PredDiff(model, df_train, n_imputations=n_imputations, regression=False,
                                     imputer=imputer, fast_evaluation=True, n_group=200, unified_integral=False)

    # pd_trainset = preddiff.PredDiff(model, df_train, imputer_cls=impute.TrainSetMahalanobisImputer, regression=False, n_jobs=8, n_estimators=20)

    # pd_trainset = preddiff.PredDiff(model, df_train, n_imputations=n_imputations, regression=False,
    #                                 imputer=imputer, fast_evaluation=True, n_group=200)
    # pd_vae = preddiff.PredDiff(model, df_train, n_imputations=n_imputations, regression=False,
    #                            imputer_cls=vae.VAEImputer, gpus=0, epochs=2, custom_postprocessing_fn=df_round_clip)

    df_cherry_picked = df_test.iloc[[4, 15, 84, 9]]  # one digits each: 4, 5, 8, 9

    data_random = df_test.iloc[np.random.randint(low=0, high=df_test.shape[0], size=2)]

    data = df_cherry_picked[:]
    # data = df_test.iloc[:20]
    # data = data_random

    data_np = data.to_numpy().reshape(-1, iparam.n_pixel, iparam.n_pixel)

    # select explainer depending on imputer
    # pd_explainer = pd_trainset
    # pd_explainer = pd_vae

    m_relevance, prediction_prob, m_list = ut_mnist.get_relevances(explainer=pd_explainer,
                                                                   data=data_np, img_params=iparam)

    paper_plot = True
    cbar = True
    save = False
    key = 'mean'
    if paper_plot is True:
        from pred_diff.tools import init_plt as init_plt

        init_plt.update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=False)
        show_all_digits = False
    else:
        plt.style.use('default')
        show_all_digits = True

    # visualize imputations =========================
    # squared mask
    # mask_squared = np.zeros(data_np.shape[1:], dtype=np.bool)
    # mask_squared[8:12, 8:12] = True
    # mask_squared[16:20, 12:16] = True
    # for i in range(data_np.shape[0]):
    #     ut_mnist.plot_imputations(imputer=imputer, image=data_np, mask=mask_squared, i_sample=i)

    for img_id in np.arange(data.shape[0])[:]:
        n_importance = 1
        i_reference = ut_mnist.get_reference_pixel(m_relevance=m_relevance, prediction_prob=prediction_prob,
                                                   img_id=img_id, n_importance=n_importance)

        m_interaction = ut_mnist.get_interaction(explainer=pd_explainer, data=data_np, iparam=iparam, m_list=m_list,
                                                 i_reference=i_reference, key=key)

        i_vertical, i_horizontal = divmod(i_reference, iparam.max_index)
        # scale_factor = 1.
        # color = '#77dd77'
        # rect = plt.Rectangle((i_horizontal * iparam.block_size, i_vertical * iparam.block_size),
        #                      scale_factor*iparam.block_size, scale_factor*iparam.block_size, linewidth=0, edgecolor=color,
        #                      facecolor=color)

        rect = functools.partial(ut_mnist.plot_rect, i_reference=i_reference, iparam=iparam)

        if show_all_digits is True:
            ut_mnist.plot_all_digits(m_list_collected=m_relevance, prob_classes=prediction_prob, data=data,
                                     img_params=iparam, img_id=img_id,
                                     imputer='trainset')
            ut_mnist.plot_all_digits(m_list_collected=m_interaction, prob_classes=prediction_prob, img_params=iparam,
                                     data=data,
                                     img_id=img_id, imputer='interaction', rect=rect)
        else:
            ut_mnist.plot_predicted_digit(relevance=m_relevance, interaction=m_interaction,
                                          prob_classes=prediction_prob,
                                          data_digit=data, rect=rect, img_params=iparam, image_id=img_id,
                                          imputer=f'PredDiff{key=}_',
                                          save=save, cbar=cbar)
            # plot_predicted_digit(m_list_collected=m_interaction, prob_classes=prediction_prob, data_digit=data,
            #              img_params=iparam, image_id=img_id, imputer='interaction', rect=rect, save=save, cbar=cbar)

            # ut_mnist.plot_comparison(m_list_collected=m_relevance, prob_classes=prediction_prob, data_digit=data,
            #              img_params=iparam, image_id=img_id, imputer='PredDiff', save=save, cbar=cbar)

            # plot_filtersize_dependence(explainer=pd_explainer, data_digit=data, img_param=iparam)

    # CODE for SHAP comparison
    flag_shap = False
    if flag_shap is True:
        import os

        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from torchvision import datasets as ds, transforms
        import pandas as pd

        mnist_train_ds = ds.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        target_train = pd.DataFrame(data=np.stack([y for x, y in mnist_train_ds]))
        X = df_train.to_numpy()
        y = np.squeeze(target_train.to_numpy())

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

        clf = RandomForestClassifier(n_estimators=10, max_depth=20)
        clf.fit(X_train, y_train)

        print(f'train score: {clf.score(X_train, y_train)}')  # 99.9%
        print(f'val score: {clf.score(X_val, y_val)}')  # 97.3%

        # test data
        X_test = df_test.to_numpy()
        y_test = np.squeeze(target_test.to_numpy())
        print(f'val score: {clf.score(X_test, y_test)}')  # 96.9%

        predictions = clf.predict(data)
        explainer = shap.TreeExplainer(model=clf)
        all_shap_values = explainer.shap_values(data)

        for i_image in range(data_np.shape[0]):
            shap_interaction_values = explainer.shap_interaction_values(data[i_image:i_image + 1])
            i_digit = predictions[i_image]
            shap_values = all_shap_values[i_digit][i_image]
            most_important_pixel = shap_values.argmax()
            shap_interaction_matrix = shap_interaction_values[i_digit][0]
            interaction_to_pixel = shap_interaction_matrix[most_important_pixel]
            interaction_to_pixel[most_important_pixel] = 0

            figsize = plt.rcParams['figure.figsize'].copy()
            figsize[1] = 0.7 * figsize[1]

            title = f"shap_{data.index[i_image]}"
            fig = plt.figure(title, figsize=figsize)
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title('(a) Shapley values')
            ut_mnist.imshow_digit(fig=fig, ax=ax, digit=data_np[i_image], heatmap=shap_values.reshape(28, 28))

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title('(b) SHAP interaction')
            rect = functools.partial(ut_mnist.plot_rect, i_reference=most_important_pixel, iparam=iparam)
            ut_mnist.imshow_digit(fig=fig, ax=ax, digit=data_np[i_image], heatmap=interaction_to_pixel.reshape(28, 28))
            rect(ax)
            plt.tight_layout(pad=0.1)


def initialize_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_shapley_coalitions', type=int, default=5, help='Number of coalitions S used to calculate'
                                                                               'Shapley values. Only used for Shapley.')
    parser.add_argument('--n_imputations', type=int, default=1)
    parser.add_argument('--n_images', type=int, default=4,
                        help="max number of images, for which rel. and int. are evaluated. "
                             "4: paper selection, 5: fixed random selection for PredDiff vs. shapley comparison "
                             "otherwise random selection")
    parser.add_argument("--imputer", type=str, default='TrainSet',
                        choices=['TrainSet', 'GaussianNoise', 'VAEImputer'])
    parser.add_argument('--explainer', type=str, default='Shapley', choices=['PredDiff', 'Shapley'])
    parser.add_argument('--interaction', default=True, action="store_true")
    parser.add_argument('--n_interaction_pixel', type=int, default=1,
                        help="for how many of the most relevant segments interactions will be evaluated")
    parser.add_argument('--random_interaction_pixel', action="store_true",
                        help="select random reference pixel")
    parser.add_argument('--visualize_imputations', type=bool, default=False)
    parser.add_argument("--n_segments", type=int, default=100, help="approximately number of superpixel")
    parser.add_argument('--compactness', type=float, default=0.5, help="Compactness parameter used for SLIC superpixel")
    parser.add_argument('--image_data_dir', type=str,
                        default="../../data/CUB_200_2011/CUB_200_2011/images_sorted/test/")
    parser.add_argument("--model_cuda_device", type=int, default=0, help="-1 for cpu")
    # parser.add_argument("--model", type=str, default="places", choices=["vgg16_imagenet", "vgg16_cub200", 'places'])
    # parser.add_argument("--model_checkpoint", type=str, default="../checkpoints/vgg_caltech_full_net_trained.pth",
    #                     help="location of model checkpoint if not provided by torchvision")
    # parser.add_argument('--imputer_checkpoint', type=str,
    #                     default="../../pd-impute/Diverse-Structure-Inpainting/checkpoints/imagenet_random",
    #                     help='location of the imputer checkpoint')
    # parser.add_argument('--imputer_batch_size', type=int, default=8,
    #                     help='batch size for vq-vae imputer, n_imputations must be multiple of it')
    parser.add_argument('--save_dir', default='data/', type=str, help="directory, where segments, relevances and interaction relevances\
                                                        will be stored in a folder named depending on date, time and args")
    parser.add_argument('--n_group', type=int, default=600, help='must be multiple of n_imputations')

    args = parser.parse_args()
    return args


def generate_superpixel_slic_mnist(image: np.ndarray, n_segments: int, compactness: float):    # good value: com=0.5
    """image: shape: (n_pixel**2)"""
    assert len(image.shape) == 1, 'provide only a single image'
    # increase compactness to produce more squared superpixel
    image_reshape = image.reshape((28, 28))
    seg = slic(image_reshape, n_segments=n_segments, compactness=compactness, start_label=0)

    # plt.figure()
    # plt.imshow(image_reshape, alpha=1)
    # plt.imshow(seg, alpha=0.3, cmap='prism')

    seg_flatten = seg.reshape(28**2)
    list_of_masks = [seg_flatten == s for s in np.unique(seg)]
    return list_of_masks, seg


def generate_mnist_imputations(imputer: ImputerBase, image_selection: np.ndarray, n_segments: int, compactness: float):
    n_imputations = 5
    dic_imputations = {'n_example_imputations': n_imputations,
                       'n_segments': n_segments,
                       'n_images': image_selection.shape[0]}
    for i, image in enumerate(image_selection):
        # superpixel
        list_of_masks, seg = generate_superpixel_slic_mnist(image=image, n_segments=n_segments, compactness=compactness)
        if args.imputer == "TrainSet":  # use completely factorizing imputer distribution
            imputer.seg = seg.flatten()
            pass

        i_first_superpixel = int(len(list_of_masks) / 10 * 3)
        i_second_superpixel = int(len(list_of_masks) / 10 * 7)
        i_third_superpixel = int(len(list_of_masks) / 10 * 5)
        mask_reference_superpixel = list_of_masks[i_first_superpixel] + list_of_masks[i_second_superpixel] + \
                                    list_of_masks[i_third_superpixel]

        # prepare imputations
        imputations, _ = imputer.impute(test_data=image[np.newaxis], mask_impute=mask_reference_superpixel,
                                        n_imputations=n_imputations)

        image_imputed = np.stack([image[np.newaxis].copy() for _ in range(n_imputations)])      # n_imutations, 1, *mask.shape
        image_imputed[:, :, mask_reference_superpixel] = imputations[:, :, mask_reference_superpixel]
        dic_imputations.update({f'image_original_{i}': image,
                                f'image_imputed_{i}': image_imputed,
                                'imputer': imputer.imputer_name,
                                f'segments_{i}': seg,
                                f'mask_{i}': mask_reference_superpixel
                                })
    return dic_imputations


def calculate_attributions_mnist(args):
    model, df_train, df_test, target_test = ut_mnist.get_model_and_data(max_epochs=20, retrain=False)
    imgs_train = df_train.to_numpy()
    imgs_test = df_test.to_numpy()


    # store results by date, time and arguments
    now = datetime.now()
    file_name = f"{now.date()}_{now.strftime('%H%M')}_mnist{args.n_images}_resolution{args.n_segments}_{args.imputer}_{args.n_imputations}"
    if args.explainer == 'Shapley':
        file_name += f'_shapley{args.n_shapley_coalitions}'
    args.save_file = args.save_dir + file_name
    os.makedirs(args.save_dir, exist_ok=True)

    dict_mnist = args.__dict__

    # set-up selected imputer
    # PredDiff
    if args.imputer == "TrainSet":
        imputer = general_imputers.TrainSetImputer(train_data=imgs_train)
    elif args.imputer == "GaussianNoise":
        sigma = imgs_train[:300].std(axis=0)
        imputer = general_imputers.GaussianNoiseImputer(train_data=imgs_train, sigma=sigma)
    elif args.imputer == "Histogram":
        imputer = color_sampling_imputer.ColorHistogramImputer(train_data=imgs_train)
    elif args.imputer == 'cv2_telea':
        imputer = single_shoot_imputer.OpenCVInpainting(inpainting_algorithm='telea')
    elif args.imputer == 'MeanImputer':
        imputer = single_shoot_imputer.MeanImputer(train_data=imgs_train)
    elif args.imputer == 'VAEImputer':
        def df_round_clip(df):
            return df.clip(lower=0, upper=1).div(1. / 255).round().div(255.)
        imputer = vae_impute.VAEImputer(train_data=imgs_train, gpus=1, epochs=20, custom_postprocessing_fn=df_round_clip,
                                        gibbs_iterations=10)
        # pd_vae = preddiff.PredDiff(model, df_train, n_imputations=n_imputations, regression=False,
        #                            imputer_cls=vae.VAEImputer, gpus=0, epochs=2, custom_postprocessing_fn=df_round_clip)
    else:
        assert False, f'incorrect imputer argument: {args.imputer}'

    if args.explainer == 'PredDiff':
        explainer = preddiff.PredDiff(model, train_data=imgs_train, imputer=imputer, regression=False,
                                      n_imputations=args.n_imputations, n_group=args.n_group,
                                      fast_evaluation=True)
    elif args.explainer == 'Shapley':
        # check input args for validity
        assert args.n_shapley_coalitions > 0, 'Invalid n_shapley_coalitions, not positive'
        assert args.n_shapley_coalitions > args.n_imputations, 'Use more coalitions than imputations'
        import warnings
        if args.n_imputations > 1:
            warnings.warn('Depreciated to use multiple imputations per coalition.', UserWarning)

        explainer = shapley.ShapleyExplainer(model, train_data=imgs_train, imputer=imputer, regression=False,
                                             n_coalitions=args.n_shapley_coalitions, n_imputations=args.n_imputations,
                                             n_group=args.n_group)
    else:
        assert False, f'incorrect explainer argument: {args.explainer}'

    # select images
    preselected_reference_superpixels = None
    if args.n_images == 4:          # use the prespecified paper selection
        index_selection = [4, 15, 84, 9]                # one digits each: 4, 5, 8, 9
        # preselected_reference_superpixels = [9, 17, 25, 21]
    elif args.n_images == 1:
        index_selection = [7149]
        preselected_reference_superpixels = [22]
    elif args.n_images == 5:
        index_selection = [7891, 7149, 2002, 9082, 4901]
        assert args.n_segments == 50, 'preselected reference superpixel only valid for 50 slic superpixels'
        if args.imputer == 'TrainSet':
            preselected_reference_superpixels = [15, 22, 18, 28, 22]
        elif args.imputer == 'VAEImputer':
            # if args.explainer == 'Shapley':
            preselected_reference_superpixels = [25, 22, 18, 28, 39]
    elif args.n_images == 50:
        index_selection = [9275, 2330, 8593, 3286, 1903, 9206, 4488, 5096, 4577, 2372, 8938,
                           6475, 5081, 9126, 6517, 5692, 6384, 2413, 2974, 1902, 8453,  650,
                           102, 2915, 5042,  798, 7823, 2042, 9252, 3789, 2613, 7848, 6931,
                           9890, 3923, 7456, 3248,  580, 8641, 3278, 2119, 2755, 6230, 9943,
                           1275,  687, 9520, 9757, 7601, 1074]
        assert args.n_segments == 50, 'preselected reference superpixel only valid for 50 slic superpixels'
        preselected_reference_superpixels = [33, 10, 35, 15, 32, 21, 29, 17, 25, 25, 10, 17, 23, 14,  7, 20, 14, 14, 24,
                                             15, 35, 29, 19, 30, 21, 17, 22, 31, 24, 31, 10, 30, 21, 34, 18, 11, 19, 30,
                                             23, 10, 28, 13, 21, 17, 10, 34, 25, 14, 16, 27]
    else:
        assert args.n_images > 0
        rng = np.random.default_rng(0)
        temp = rng.choice(np.arange(0, imgs_test.shape[0]), size=args.n_images, replace=False)
        index_selection = list(temp)

    # create dict to visualize imputations
    get_dict_imputation_visualization = False
    if get_dict_imputation_visualization is True:
        dict_imputations = generate_mnist_imputations(imputer=imputer, image_selection=np.stack([imgs_test[i] for i in index_selection]),
                                   n_segments=args.n_segments, compactness=args.compactness)
        args.save_file = args.save_file + '_imputations'
        dict_imputations.update(args.__dict__)
        return dict_imputations

    # Relevances
    dict_mnist['n_images'] = len(index_selection)
    dict_mnist['image_index_selection'] = index_selection
    collect_highest_relevance_reference_index = []
    for i, img_index in enumerate(index_selection):
        print(f'image {i} of {len(index_selection)} total')
        img = imgs_test[img_index]
        img_model = torch.tensor(img[np.newaxis])

        masks, seg = generate_superpixel_slic_mnist(img, n_segments=args.n_segments, compactness=args.compactness)
        if args.explainer == 'PredDiff':
            m_values = explainer.relevances(img[np.newaxis], list_masks=masks)
        elif args.explainer == 'Shapley':
            if args.imputer == "TrainSet":      # use completely factorizing imputer distribution
                explainer.imputer.seg = seg.flatten()
                pass
            m_values = explainer.shapley_values(data_test=img[np.newaxis], list_masks=masks,
                                                base_feature_mask=seg.flatten())
        else:
            assert False, 'explainer not defined'

        predicted_class = m_values[0]['pred'][0].argmax()

        dict_mnist[f'image_{img_index}'] = img
        img_predict_proba = model.predict_proba(img_model)
        dict_mnist[f'predict_proba_{img_index}'] = img_predict_proba
        # dict_mnist[f'masks_{img_index}'] = masks
        dict_mnist[f'seg_{img_index}'] = seg
        m_mean = np.array([m['mean'][0][predicted_class] for m in m_values])
        dict_mnist[f'relevance_{img_index}'] = m_mean

        if args.interaction is True:
            if args.random_interaction_pixel:
                np.random.seed(42)
                assert False, 'args.n_segments is not the actual number of segments'
                reference_segments = np.random.choice(args.n_segments, size=args.n_interaction_pixel, replace=False)
            else:
                reference_segments = m_mean.argsort()[::-1][:args.n_interaction_pixel]      # get n most relevant
                if preselected_reference_superpixels is not None:
                    reference_segments = [preselected_reference_superpixels[i]]

            collect_highest_relevance_reference_index.append(reference_segments)

            for ref_s in reference_segments:
                interaction_masks = [[masks[ref_s], masks[s]] for s in np.unique(seg) if s != ref_s]
                if args.explainer == 'PredDiff':
                    interaction_list_pd = explainer.interactions(img[np.newaxis],
                                                                 list_interaction_masks=interaction_masks)
                elif args.explainer == 'Shapley':
                    interaction_list_pd = \
                        explainer.shapley_interaction_index(data_test=img[np.newaxis], base_feature_mask=seg.flatten(),
                                                            list_interaction_masks=interaction_masks)
                    # assert False, "shapley interaction index not available"
                else:
                    assert False, 'explainer does not support interactions'
                m_int_mean = np.array(
                    [m['mean'][0][predicted_class] if i != ref_s else 0 for i, m in enumerate(interaction_list_pd)])
                dict_mnist[f'interaction_{img_index}_{ref_s}'] = m_int_mean
                dict_mnist[f'reference_{img_index}'] = reference_segments
            dict_mnist['n_interaction'] = len(reference_segments)
    print(np.array(collect_highest_relevance_reference_index).squeeze())
    # transform via string = 'printed list'
    # list_of_strings = string.split(' ')
    # index_list = [int(value) for value in list_of_strings]
    return dict_mnist


def _get_relevance_data(dict_mnist: dict, img_index: int):
    digit = dict_mnist[f'image_{img_index}'].reshape((28, 28))
    img_relevance = dict_mnist[f'relevance_{img_index}']

    img_seg = dict_mnist[f'seg_{img_index}']
    heatmap = np.zeros(digit.shape)
    for seg_index, seg_relevance in zip(np.unique(img_seg), img_relevance):
        mask = img_seg == seg_index
        heatmap[mask] = seg_relevance

    return digit, heatmap


def _get_interaction_data(dict_mnist: dict, img_index: int):
    digit = dict_mnist[f'image_{img_index}'].reshape((28, 28))
    img_relevance = dict_mnist[f'relevance_{img_index}']
    img_seg = dict_mnist[f'seg_{img_index}']


    reference_index = dict_mnist[f'reference_{img_index}'][0]
    interactions_reference_index = dict_mnist[f'interaction_{img_index}_{reference_index}']
    heatmap_interaction = np.zeros(digit.shape)
    for seg_index, seg_interaction in zip([s for s in np.unique(img_seg) if s != reference_index],
                                          interactions_reference_index):
        mask_interaction = img_seg == seg_index
        heatmap_interaction[mask_interaction] = seg_interaction
    return digit, heatmap_interaction, reference_index


def visualize_mnist_attributions(dict_mnist: dict):
    explainer = dict_mnist['explainer']

    init_plt.update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=False)

    for img_index in dict_mnist['image_index_selection']:
        digit, heatmap = _get_relevance_data(dict_mnist, img_index)

        # title = f"{imputer}{data_digit.index[image_id]}"
        figsize = plt.rcParams['figure.figsize'].copy()
        figsize[1] = 0.7 * figsize[1]
        fig = plt.figure(f'mnist_{img_index}_{explainer}', figsize=figsize)

        ax_relevance = plt.subplot(1, 2, 1)   # relevance
        ax_interaction = plt.subplot(1, 2, 2)  # interactions

        if explainer == 'PredDiff':
            ax_relevance.set_title('(a) relevance')
            cbar_label_relevance = r"  ${\bar{m}}\,$"
            ax_interaction.set_title('(b) joint effect')
            cbar_label_interaction = r"  $\,\,{\bar{m}}^{f_c^{\,Y\,Z}}$"
            sign_flipped = 1
        elif explainer == 'Shapley':
            ax_relevance.set_title('(a) relevance')
            cbar_label_relevance = r"  $\phi$"
            ax_interaction.set_title('(b) interaction')
            cbar_label_interaction = r"$\phi_{Y, Z}$"
            sign_flipped = -1       # additional sign flipped
        else:
            assert False
        ut_mnist.imshow_digit(fig=fig, ax=ax_relevance, digit=digit, heatmap=heatmap,
                              cbar_label=cbar_label_relevance)


        digit, heatmap_interaction, reference_superpixel_index = _get_interaction_data(dict_mnist, img_index)

        ut_mnist.imshow_digit(fig=fig, ax=ax_interaction, digit=digit, heatmap=sign_flipped*heatmap_interaction,
                              cbar_label=cbar_label_interaction)

        # plot reference superpixel
        masked_data = np.ones(digit.shape)
        mask = dict_mnist[f'seg_{img_index}'] == reference_superpixel_index
        masked_data = np.ma.masked_where(mask != 1, masked_data)
        ax_interaction.imshow(masked_data, interpolation='nearest', cmap='Accent')

        plt.tight_layout(pad=0.1)


def visualize_mnist_imputations(dict_imputations: dict):
    imputer = dict_imputations['imputer']
    n_cols = dict_imputations['n_example_imputations'] + 1
    n_images = min(dict_imputations['n_images'], 6)

    fig = plt.figure(f'visualize_{imputer}', figsize=(3, 2))
    # ax.set_title('original')
    for i_image in range(n_images):
        ax = plt.subplot(n_images, n_cols, i_image * n_cols + 1)
        ax.imshow(dict_imputations[f'image_original_{i_image}'].reshape(28, 28), cmap='Greys')
        mask = dict_imputations[f'mask_{i_image}'].reshape(28, 28)

        masked_data = np.ones_like(mask)
        masked_data = np.ma.masked_where(mask != 1, masked_data)
        ax.imshow(masked_data, alpha=0.8, interpolation='nearest', cmap='Set1')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.3)
        # plt.axis(False)

        image_imputed = dict_imputations[f'image_imputed_{i_image}']
        for i_imputation in range(dict_imputations['n_example_imputations']):
            ax = plt.subplot(n_images, n_cols, i_image*n_cols + 2+i_imputation)
            # ax.set_title(f'i = {i_imputation}')
            ax.imshow(image_imputed[i_imputation, 0].reshape(28, 28), cmap='Greys')
            # plt.colorbar()
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.3)
            # plt.axis(False)
            ax.set_alpha(0.5)

    fig.tight_layout(pad=0.25)


def compare_preddiff_vs_shapley(dict_preddiff: dict, dict_shapley: dict, str_attribution='relevance'):
    # n_images = len(dict_preddiff['image_index_selection'])
    image_index_selection = dict_preddiff['image_index_selection']
    assert dict_shapley['image_index_selection'] == image_index_selection, 'Mismatch in image selections between dict'

    init_plt.update_rcParams(fig_width_pt=234.88*2, half_size_image=False)
    figsize = plt.rcParams['figure.figsize'].copy()
    figsize[1] = 0.33 * figsize[1]
    for str_attribution in ['relevance', 'interaction']:
        for row, current_dict in enumerate([dict_preddiff, dict_shapley]):
            explainer = current_dict['explainer']
            fig, ax_array = plt.subplots(1, len(image_index_selection),
                                         num=f'{explainer}_{str_attribution}', figsize=figsize)
            for index, img_index in enumerate(image_index_selection):

                sign_flipped = 1
                if str_attribution == 'relevance':
                    digit, heatmap = _get_relevance_data(current_dict, img_index)
                elif str_attribution == 'interaction':
                    digit, heatmap, highest_relevance_index = _get_interaction_data(current_dict, img_index)
                    if explainer == 'Shapley':
                        sign_flipped = -1
                else:
                    assert False, f'Invalid input: {str_attribution}'
                #
                ut_mnist.imshow_digit(fig=fig, ax=ax_array[index], digit=digit, heatmap=sign_flipped*heatmap,
                                      cbar=False)

                if str_attribution == 'interaction':
                    # plot reference superpixel
                    masked_data = np.ones(digit.shape)
                    mask = current_dict[f'seg_{img_index}'] == highest_relevance_index
                    masked_data = np.ma.masked_where(mask != 1, masked_data)
                    ax_array[index].imshow(masked_data, interpolation='nearest', cmap='Accent')
                plt.tight_layout(pad=0.3)

    # ax_array[0, 2].set_title(str_attribution)
    # ax_array[0, 0].set_ylabel('PredDiff')


def compare_computational_scaling(path_to_folder='data/mnist_attributions/computational_scaling/'):
    # load all dict
    init_plt.update_rcParams(fig_width_pt=234.88 * 2, half_size_image=True)
    figsize = (6.46, 2)

    all_dict_list = glob.glob(path_to_folder + '*')

    list_dict_preddiff = []
    list_dict_shapley = []
    for path in all_dict_list:
        current_dict = pickle.load(open(path, 'rb'))
        if current_dict['explainer'] == 'PredDiff':
            list_dict_preddiff.append(current_dict)
        elif current_dict['explainer'] == 'Shapley':
            list_dict_shapley.append(current_dict)

    list_dict_preddiff.sort(key=lambda element: element['n_imputations'])
    list_dict_shapley.sort(key=lambda element: element['n_shapley_coalitions'])

    preddiff_n_imputations = np.array([current_dict['n_imputations'] for current_dict in list_dict_preddiff])
    shapley_n_imputations = np.array([current_dict['n_shapley_coalitions'] for current_dict in list_dict_shapley])
    image_index_selection = list_dict_preddiff[0]['image_index_selection']

    def extract_heatmaps(list_of_dict):
        list_heatmaps_relevance = []
        list_heatmaps_interaction = []
        for current_dict in list_of_dict:
            temp_heatmap_relevance = []
            temp_heatmap_interaction = []
            for img_index in image_index_selection:
                digit, heatmap = _get_relevance_data(current_dict, img_index)
                temp_heatmap_relevance.append(heatmap)

                digit, heatmap, highest_relevance_index = _get_interaction_data(current_dict, img_index)
                temp_heatmap_interaction.append(heatmap)

            list_heatmaps_relevance.append(np.stack(temp_heatmap_relevance))
            list_heatmaps_interaction.append(np.stack(temp_heatmap_interaction))

        heatmap_relevance = np.stack(list_heatmaps_relevance)
        # r_max = heatmap_relevance.max(axis=(2, 3))
        # heatmap_relevance = heatmap_relevance / r_max[..., np.newaxis, np.newaxis]
        heatmap_interaction = np.stack(list_heatmaps_interaction)
        # i_max = heatmap_interaction.max(axis=(2, 3))
        # heatmap_interaction = heatmap_interaction / i_max[..., np.newaxis, np.newaxis]
        return heatmap_relevance, heatmap_interaction

    preddiff_heatmap_relevance, preddiff_heatmap_interaction = extract_heatmaps(list_dict_preddiff)
    shapley_heatmap_relevance, shapley_heatmap_interaction = extract_heatmaps(list_dict_shapley)

    def get_mean_and_error(array_of_heatmaps: np.ndarray):
        distance_to_reference = np.linalg.norm(array_of_heatmaps[:-1] - array_of_heatmaps[-1], axis=(2, 3), ord='fro')
        mean_heatmaps = distance_to_reference.mean(axis=1)
        error_heatmaps = distance_to_reference.std(axis=1)/np.sqrt(distance_to_reference.shape[1])
        return mean_heatmaps, error_heatmaps

    def calculate_cosine_similarity(array_of_heatmaps: np.ndarray, reference_heatmap: np.ndarray):
        a = array_of_heatmaps.reshape(*array_of_heatmaps.shape[:2], -1)
        b = reference_heatmap.reshape(*reference_heatmap.shape[:1], -1)
        a_times_b = a * b
        cos_sim = a_times_b.sum(axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        return cos_sim.mean(axis=-1), cos_sim.std(axis=-1)/np.sqrt(cos_sim.shape[-1])


    mean_preddiff_relevance, error_preddiff_relevance = calculate_cosine_similarity(preddiff_heatmap_relevance[:-1], preddiff_heatmap_relevance[-1])
    mean_preddiff_interaction, error_preddiff_interaction = calculate_cosine_similarity(preddiff_heatmap_relevance[:-1], preddiff_heatmap_relevance[-1])
    # mean_preddiff_relevance, error_preddiff_relevance = get_mean_and_error(preddiff_heatmap_relevance)
    # mean_preddiff_interaction, error_preddiff_interaction = get_mean_and_error(preddiff_heatmap_interaction)

    mean_shapley_relevance, error_shapley_relevance = calculate_cosine_similarity(shapley_heatmap_relevance[:-1], shapley_heatmap_relevance[-1])
    mean_shapley_interaction, error_shapley_interaction = calculate_cosine_similarity(shapley_heatmap_relevance[:-1], shapley_heatmap_relevance[-1])
    # mean_shapley_relevance, error_shapley_relevance = get_mean_and_error(shapley_heatmap_relevance)
    # mean_shapley_interaction, error_shapley_interaction = get_mean_and_error(shapley_heatmap_interaction)

    color_preddiff = 'C1'
    color_shapley = 'C2'
    fig = plt.figure('Computational Comparison', figsize=figsize)

    # relevances
    error_bars = False
    ax_relevance = plt.subplot(2, 1, 1)

    ax_relevance.tick_params(left=True,
                             bottom=False,
                             labelleft=True,
                             labelbottom=False)
    if error_bars is True:

        plt.errorbar(x=preddiff_n_imputations[:-1], y=mean_preddiff_relevance, yerr=error_preddiff_relevance,
                     fmt='D-', label=r'Relevance: $PredDiff$ ', color=color_preddiff)
        plt.errorbar(x=shapley_n_imputations[:-1], y=mean_shapley_relevance, yerr=error_shapley_relevance,
                     fmt='s-', label='Relevance: Shapley', color=color_shapley)


        distance_pd_sh = np.linalg.norm(preddiff_heatmap_relevance[:-1] - shapley_heatmap_relevance[-1], axis=(2, 3), ord='fro')
        plt.errorbar(preddiff_n_imputations[:-1], distance_pd_sh.mean(axis=1), distance_pd_sh.std()/np.sqrt(distance_pd_sh.shape[1]),
                     fmt='^', label=r'$\phi \approx \bar{m}$', linestyle='solid', color='C3')
    else:
        plt.plot(preddiff_n_imputations[:-1], mean_preddiff_relevance,
                     'D-', label='Relevance: PredDiff ', color=color_preddiff)
        plt.plot(shapley_n_imputations[:-1], mean_shapley_relevance,
                     's-', label='Relevance: Shapley', color=color_shapley)

        preddiff_approximates_shapley_relevance, _ = calculate_cosine_similarity(preddiff_heatmap_relevance[:-1],
                                                        shapley_heatmap_relevance[-1])
        # plt.plot(preddiff_n_imputations[:-1], preddiff_approximates_shapley_relevance,
        #          '^', label=r'$\phi \approx \bar{m}$', linestyle='solid', color='C3')
        # distance_pd_sh = np.linalg.norm(preddiff_heatmap_relevance[:-1] - shapley_heatmap_relevance[-1], axis=(2, 3),
        #                                 ord='fro')
        # plt.plot(preddiff_n_imputations[:-1], distance_pd_sh.mean(axis=1),
        #              '^', label=r'$\phi \approx \bar{m}$', linestyle='solid', color='C3')

    # interactions
    ax_interaction = plt.subplot(2, 1, 2, sharex=ax_relevance)
    if error_bars is True:
        plt.errorbar(x=preddiff_n_imputations[:-1], y=mean_preddiff_interaction, yerr=error_preddiff_interaction,
                     fmt='D-', label='  $PredDiff$\n' + r'$||\bar{m}^{\#} - \bar{m}^{\infty}||$', color=color_preddiff)
        plt.errorbar(x=shapley_n_imputations[:-1], y=mean_shapley_interaction, yerr=error_shapley_interaction,
                     fmt='s-', label='  Shapley\n' + '$||\phi^{\#} - \phi^{\infty}||$', color=color_shapley)


        distance_pd_sh = np.linalg.norm(preddiff_heatmap_interaction[:-1] - shapley_heatmap_interaction[-1], axis=(2, 3), ord='fro')
        plt.errorbar(preddiff_n_imputations[:-1], distance_pd_sh.mean(axis=1), distance_pd_sh.std()/np.sqrt(distance_pd_sh.shape[1]),
                     label='Approximation\n' + r'  $||\bar{m}^{\#} - \phi^{\infty}||$', fmt='^', linestyle='solid', color='C3')
    else:
        plt.plot(preddiff_n_imputations[:-1], mean_preddiff_interaction,
                     'D-', label='$PredDiff$', color=color_preddiff)        #  + r'$||\bar{m}^{\#} - \bar{m}^{\infty}||$'
        plt.plot(shapley_n_imputations[:-1], mean_shapley_interaction,
                     's-', label='Shapley', color=color_shapley)     # + '$||\phi^{\#} - \phi^{\infty}||$'

        preddiff_approximates_shapley_interactions, _ = calculate_cosine_similarity(preddiff_heatmap_interaction[:-1], -shapley_heatmap_interaction[-1])
        # plt.plot(preddiff_n_imputations[:-1], preddiff_approximates_shapley_interactions, '^',
        #              label='Approximation\n' + r'$||\bar{m}^{\#} - \phi^{\infty}||$',  linestyle='solid',
        #              color='C3')
        # distance_pd_sh = np.linalg.norm(preddiff_heatmap_interaction[:-1] - shapley_heatmap_interaction[-1],
        #                                 axis=(2, 3), ord='fro')
        # plt.plot(preddiff_n_imputations[:-1], distance_pd_sh.mean(axis=1), '^',
        #              label='Approximation\n' + r'$||\bar{m}^{\#} - \phi^{\infty}||$',  linestyle='solid',
        #              color='C3')

    # ax_interaction.tick_params(left=True,
    #                bottom=,
    #                labelleft=True,
    #                labelbottom=True)

    from matplotlib.offsetbox import AnchoredText

    text_str = f'(a) Relevances'
    at = AnchoredText(text_str, prop=dict(size=8), frameon=False, loc='lower right', pad=0.2, borderpad=0.2)
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.15")
    ax_relevance.add_artist(at)

    text_str = f'(b) Interactions'
    at = AnchoredText(text_str, prop=dict(size=8), frameon=False, loc='lower right', pad=0.2, borderpad=0.2)
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.15")
    ax_interaction.add_artist(at)


    # ax_relevance.semilogy()
    # ax_interaction.semilogy()
    ax_relevance.set_xlim(0, 630)
    ax_relevance.set_ylim(-0.1, 1.1)
    ax_interaction.set_ylim(-0.1, 1.1)

    # distance_preddiff_vs_shapley = np.linalg.norm(preddiff_heatmap_relevance[-1] - shapley_heatmap_relevance[-1],
    #                                               axis=(1, 2), ord='fro')
    # plt.axhline(distance_preddiff_vs_shapley.mean(), label=r'$|PredDiff - Shapley|_F$ rel', color=f'C5', linestyle='dashdot')
    #
    # distance_preddiff_vs_shapley_interaction = np.linalg.norm(preddiff_heatmap_interaction[-1] - shapley_heatmap_interaction[-1],
    #                                               axis=(1, 2), ord='fro')
    # plt.axhline(distance_preddiff_vs_shapley_interaction.mean(), label=r'$|PredDiff - Shapley|_F$ int', color=f'C6',
    #             linestyle='dotted')
    # error = distance_preddiff_vs_shapley_interaction.std()/50
    # print(error)


    ax_interaction.set_ylabel('                Cosine similarity', loc='bottom')
    plt.xlabel('# model calls')
    # plt.legend(ncol=3, bbox_to_anchor=(0., 1.07, 1., .102), loc='upper center', framealpha=1)
    # plt.legend(ncol=2, loc=(0.3, 0.18), framealpha=1)
    plt.legend(ncol=2, loc='lower center', framealpha=1, title='Speed of Convergence', title_fontsize=9)
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0)  # change padding between all subplots, call after tight_layout()


if __name__ == '__main__':
    # old_main_20Jan()
    args = initialize_parser()
    dict_mnist_new = calculate_attributions_mnist(args=args)
    # pickle.dump(dict_mnist_new, open(args.save_file, 'wb'))
    # visualize_mnist_attributions(dict_mnist=dict_mnist_new)

    file_path = 'data/2022-05-13_1537_mnist4_resolution150_TrainSet_600'
    # file_path = 'data/mnist_attributions/vae_imputer/2022-02-11_1657_mnist4_resolution50_VAEImputer_600'
    # file_path = 'data/2022-02-07_1228_mnist4_resolution50_TrainSet_5_imputations'

    dict_mnist = pickle.load(open(file_path, 'rb'))
    visualize_mnist_attributions(dict_mnist=dict_mnist)


    # file_path = 'data/mnist_attributions/vae_imputer/'
    file_path = 'data/mnist_attributions/visualize_imputations/2022-02-07_1354_mnist4_resolution50_VAEImputer_5_imputations'
    # file_path = 'data/mnist_attributions/visualize_imputations/2022-02-07_1335_mnist4_resolution50_TrainSet_5_imputations'

    dict_mnist = pickle.load(open(file_path, 'rb'))
    # visualize_mnist_imputations(dict_imputations=dict_mnist)

    #
    # file_path = 'data/mnist_attributions/2022-01-27_1628_mnist5_resolution50_TrainSet_1200'
    # # file_path = 'data/mnist_attributions/2022-01-26_1526_mnist4_resolution50_TrainSet_600'
    file_path = 'data/mnist_attributions/vae_imputer/2022-02-04_1742_mnist5_resolution50_VAEImputer_600_update'
    file_path = 'data/2022-02-14_1620_mnist5_resolution50_VAEImputer_600'
    dict_preddiff = pickle.load(open(file_path, 'rb'))
    # visualize_mnist_attributions(dict_mnist=dict_preddiff)


    # file_path = 'data/mnist_attributions/2022-01-26_1809_mnist4_resolution50_TrainSet_1_shapley600'
    # file_path = 'data/mnist_attributions/2022-01-27_1758_mnist5_resolution50_TrainSet_1_shapley1200'
    file_path = 'data/mnist_attributions/vae_imputer/2022-02-15_1448_mnist5_resolution50_VAEImputer_1_shapley600_update'
    dict_shapley = pickle.load(open(file_path, 'rb'))
    # visualize_mnist_attributions(dict_mnist=dict_shapley)
    # compare_preddiff_vs_shapley(dict_preddiff=dict_preddiff, dict_shapley=dict_shapley)

    compare_computational_scaling()


    file_path = 'data/2022-02-21_1006_mnist1_resolution50_VAEImputer_1_shapley600'
    dict_single_6 = pickle.load(open(file_path, 'rb'))
    # visualize_mnist_attributions(dict_mnist=dict_single_6)



    file_path = 'data/mnist_attributions/vae_imputer/2022-02-08_1222_mnist5_resolution50_VAEImputer_1_shapley600'
    dict_5digits = pickle.load(open(file_path, 'rb'))
    # visualize_mnist_attributions(dict_mnist=dict_5digits)

    dict_new = dict_shapley.copy()
    # keys = ['image_7891', 'predict_proba_7891', 'seg_7891', 'relevance_7891', 'interaction_7891_25', 'reference_7891']
    keys = [ 'image_7149', 'predict_proba_7149', 'seg_7149', 'relevance_7149', 'interaction_7149_22', 'reference_7149']
    # for key in keys:
    #     dict_new[key] = dict_single_6[key]
    #
    # pickle.dump(dict_new, open('data/mnist_attributions/vae_imputer/2022-02-15_1448_mnist5_resolution50_VAEImputer_1_shapley600_update', 'wb'))