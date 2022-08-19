import numpy as np
import pickle
import os

from datetime import datetime
import argparse

import torch

from pred_diff import preddiff, shapley
from pred_diff.imputers import general_imputers, color_sampling_imputer, single_shoot_imputer
from pred_diff.tools_imagenet import utils_imagenet, utils_places
try:
    from pred_diff.imputers.vqvae_imputer.vqvae_impute import vqvae_imputer
    VQAE_IMPORT_SUCCESSFUL = True
except ImportError:
    VQAE_IMPORT_SUCCESSFUL = False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imputer", type=str, default='MeanImputer',
                        choices=['TrainSet', 'GaussianNoise', 'Histogram', 'vq-vae', 'cv2_telea', 'MeanImputer'])
    parser.add_argument("--n_segments", type=int, default=100, help="approximately number of superpixel")
    parser.add_argument('--image_data_dir', type=str,
                        default="../../data/CUB_200_2011/CUB_200_2011/images_sorted/test/")
    parser.add_argument("--model_cuda_device", type=int, default=0, help="-1 for cpu")
    parser.add_argument("--model", type=str, default="places", choices=["vgg16_imagenet", "vgg16_cub200", 'places'])
    parser.add_argument("--model_checkpoint", type=str, default="../checkpoints/vgg_caltech_full_net_trained.pth",
                        help="location of model checkpoint if not provided by torchvision")
    parser.add_argument('--imputer_checkpoint', type=str,
                        default="../../pd-impute/Diverse-Structure-Inpainting/checkpoints/imagenet_random",
                        help='location of the imputer checkpoint')
    parser.add_argument('--imputer_batch_size', type=int, default=8,
                        help='batch size for vq-vae imputer, n_imputations must be multiple of it')
    parser.add_argument('--save_dir', default='data/', type=str, help="directory, where segments, relevances and interaction relevances\
                                                     will be stored in a folder named depending on date, time and args")
    parser.add_argument('--n_imputations', type=int, default=1)
    parser.add_argument('--n_group', type=int, default=64, help='must be multiple of n_imputations')
    parser.add_argument('--n_interaction_pixel', type=int, default=3,
                         help="for how many of the most relevant segments interactions will be evaluated")
    parser.add_argument('--random_interaction_pixel', action="store_true",
                         help="select random reference pixel")
    parser.add_argument('--n_images', type=int, default=50,
                        help="max number of images, for which rel. and int. are evaluated. loads a prespecified selection",
                        choices=[5, 25, 50])
    parser.add_argument('--visualize_imputations', type=bool, default=False)
    parser.add_argument('--shapley_coalitions', type=int, default=-1, help='-1 for PredDiff else positive integer')
    parser.add_argument('--interaction', action="store_true")

    args = parser.parse_args()

    if args.shapley_coalitions != -1:
        print('Calculate Shapley values.')
        flag_shapley = True
        n_coalitions = args.shapley_coalitions
    else:
        flag_shapley = False

    # store results by date, time and arguments
    now = datetime.now()
    file_name = f"{now.date()}_{now.strftime('%H%M')}_{args.model}_samples{args.n_images}_{args.imputer}_{args.n_imputations}"
    if flag_shapley is True:
        file_name += '_shapley'
    args.save_file = args.save_dir + file_name
    os.makedirs(args.save_dir, exist_ok=True)

    dic_save = {'args': args}

    if args.model_cuda_device != -1:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    # Load images
    print(f"Loading images {args.model}...")
    if args.model == 'vgg16_cub200':
        #imgs_test, imgs_train, imgs_selection, targs_test, targs_train, targs_selection  = utils_imagenet.load_vgg16_cub200_data(n_selection=args.n_images, return_targs=True)       # Stefan
        imgs_test, imgs_train, imgs_selection, targs_test, targs_train, targs_selection = utils_imagenet.load_vgg16_cub200_data(n_selection=args.n_images,
                                                                                       data_dir=args.image_data_dir, download=False, organize_data=False, return_targs=True)   # Johanna
        model = utils_imagenet.load_vgg16_cub200_model(args=args, temperature=1.9299)
        preds = model.predict(imgs_selection)[np.arange(args.n_images),targs_selection]
        dic_save['preds'] = preds
    elif args.model == 'places':
        dataset_img, imgs_selection = utils_places.load_places_data(n_selection=args.n_images)
        model, features_blobs = utils_places.load_model(device=device)
        rng = np.random.default_rng()
        index_selection = rng.choice(dataset_img.__len__(), 1000, replace=False)      # replace=False: unique indices
        temp = [np.array(dataset_img.__getitem__(i)[0]) for i in index_selection]
        imgs_train = np.array(temp)
    else:
        assert False, f'Incorrect keyword: model = {args.model}'


    # PredDiff
    if args.imputer == "TrainSet":
        imputer = general_imputers.TrainSetImputer(train_data=imgs_train)
    elif args.imputer == "GaussianNoise":
        sigma = imgs_train[:300].std(axis=0)
        imputer = general_imputers.GaussianNoiseImputer(train_data=imgs_train, sigma=sigma)
    elif args.imputer == "Histogram":
        imputer = color_sampling_imputer.ColorHistogramImputer(train_data=imgs_train)
    elif args.imputer == "vq-vae":
        if VQAE_IMPORT_SUCCESSFUL is False:
            assert False, 'Use conda env with TensorFlow and PyTorch'
        imputer = vqvae_imputer(full_model_dir=args.imputer_checkpoint, batch_size=args.n_imputations)
    elif args.imputer == 'cv2_telea':
        imputer = single_shoot_imputer.OpenCVInpainting(inpainting_algorithm='telea')
    elif args.imputer == 'MeanImputer':
        imputer = single_shoot_imputer.MeanImputer(train_data=imgs_train)
    else:
        assert False, f'incorrect imputer argument: {args.imputer}'

    # store visualizations of imputer
    if args.visualize_imputations is True:
        print('visualize imputations')
        dic_imputations = utils_imagenet.visualize_imputations(imputer=imputer, image_selection=imgs_selection)
        pickle.dump(dic_imputations, open(args.save_dir + f'imputer_{args.imputer}_examples', 'wb'))

    if flag_shapley is False:
        explainer = preddiff.PredDiff(model, train_data=imgs_train, imputer=imputer, regression=False,
                                  classifier_fn_call="predict", n_imputations=args.n_imputations, n_group=args.n_group,
                                  fast_evaluation=True)
    else:
        explainer = shapley.ShapleyExplainer(model, train_data=imgs_train, imputer=imputer, regression=False,
                                             n_coalitions=n_coalitions,
                                             classifier_fn_call="predict", n_imputations=args.n_imputations, n_group=args.n_group)

    print(f'Using {imputer.imputer_name}')

    # Relevances
    for i, img in enumerate(imgs_selection):
        print(f'image {i} of {imgs_selection.shape[0]} total')

        masks, seg = utils_imagenet.generate_superpixel_slic(img, n_segments=args.n_segments)
        if flag_shapley is False:
            m_values = explainer.relevances(img[np.newaxis], list_masks=masks)
        else:
            if args.imputer == "TrainSet":
                explainer.imputer.seg = seg
            m_values = explainer.shapley_values(data_test=img[np.newaxis], list_masks=masks,
                                                base_feature_mask=np.broadcast_to(seg[np.newaxis], masks[0].shape))
        predicted_class = m_values[0]['pred'][0].argmax()

        dic_save[f'image_{i}'] = img
        # dic_save[f'masks_{i}'] = masks
        dic_save[f'seg_{i}'] = seg
        m_mean = np.array([m['mean'][0][predicted_class] for m in m_values])
        dic_save[f'relevance_{i}'] = m_mean
        
        if(args.interaction):
            assert flag_shapley is False, "shapley interaction index not available"
            if args.random_interaction_pixel:
                np.random.seed(42)
                reference_segments = np.random.choice(args.n_segments, size=args.n_interaction_pixel, replace=False)
            else:
                reference_segments = m_mean.argsort()[::-1][:args.n_interaction_pixel]
            for ref_s in reference_segments:
                interaction_masks = [[masks[ref_s], masks[s]] for s in range(seg.max()) if s!=ref_s]
                m_int_values = explainer.interactions(img[np.newaxis], list_interaction_masks=interaction_masks)
                m_int_mean = np.array([m['mean'][0][predicted_class] if i!=ref_s else 0 for i,m in enumerate(m_int_values)])
                dic_save[f'interaction_{i}_{ref_s}'] = m_int_mean
                dic_save[f'reference_{i}'] = reference_segments
            dic_save['n_interaction'] = args.n_interaction_pixel

    dic_save['n_images'] = imgs_selection.shape[0]
    pickle.dump(dic_save, open(args.save_file, 'wb'))


if __name__ == '__main__':
    main()
