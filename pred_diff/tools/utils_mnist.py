import copy
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torchvision import datasets as ds, transforms

# from main_MNIST import data
from . import mnist_lightning as mnist_lightning, temperature_scaling as t_scaling
from ..imputers.imputer_base import ImputerBase

class ImgParams:
    def __init__(self, n_pixel=28, block_size=2):
        self.n_pixel = n_pixel
        self.block_size = block_size  # 1, 2, 4, 7, 14
        self.max_index = int(np.floor(n_pixel / block_size))


def get_model_and_data(max_epochs=1, retrain=True):
    # data
    mnist_train_ds = ds.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_train_dl = torch.utils.data.DataLoader(mnist_train_ds, batch_size=32, num_workers=4)

    mnist_val_ds = ds.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_val_dl = torch.utils.data.DataLoader(mnist_val_ds, batch_size=100, num_workers=4, drop_last=False)

    # Train
    path_save = "MNIST/Jan29_epoch=10dummy"
    if retrain is True:
        print('retrain model')
        # model
        model = mnist_lightning.FlatMNISTModel(conv_encoder=False)
        # model = mnist_lightning.ConvMNISTModel()

        # most basic trainer, uses good defaults
        trainer = pl.Trainer(gpus=0, progress_bar_refresh_rate=20, max_epochs=max_epochs, checkpoint_callback=False)
        trainer.fit(model, mnist_train_dl, mnist_val_dl)

        model_tsc = t_scaling.TemperatureScalingCalibration(model)
        trainer_tsc = pl.Trainer(gpus=0, progress_bar_refresh_rate=20, max_epochs=1)  # ,checkpoint_callback=False)
        mnist_val_dl_large = torch.utils.data.DataLoader(mnist_val_ds, batch_size=10000, num_workers=4, drop_last=False)
        trainer_tsc.fit(model_tsc, mnist_val_dl_large)
        model.T = model_tsc.temperature.detach().numpy()

        trainer.save_checkpoint(path_save + '.ckpt')

        # trainer_tsc.save_checkpoint(path_save + '_tsc.ckpt')
    else:
        print('load model')
        # model = mnist_lightning.FlatMNISTModel.load_from_checkpoint(checkpoint_path="pred_diff/datasets/model_mnist.ckpt")
        model = mnist_lightning.ConvMNISTModel.load_from_checkpoint(checkpoint_path="pred_diff/datasets/model_mnist_cnn.ckpt")
        # model_tsc = mnist_lightning.FlatMNISTModel.load_from_checkpoint(checkpoint_path="MNIST/Jan20_epoch=20_tsc.ckpt")
        model.T = 1.9898
        # model_tsc = t_scaling.TemperatureScalingCalibration(model)
        # trainer_tsc = pl.Trainer(gpus=0, progress_bar_refresh_rate=20, max_epochs=2)  # ,checkpoint_callback=False)
        # mnist_val_dl_large = torch.utils.data.DataLoader(mnist_val_ds, batch_size=1000, num_workers=4, drop_last=False)
        # trainer_tsc.fit(model_tsc, mnist_val_dl_large)
        # model.T = model_tsc.temperature.detach().numpy()

    df_train = pd.DataFrame(data=np.stack([x.numpy().flatten() for x, y in mnist_train_ds]))
    df_test = pd.DataFrame(data=np.stack([x.numpy().flatten() for x, y in mnist_val_ds]))
    target_test = pd.DataFrame(data=np.stack([y for x, y in mnist_val_ds]))
    return model, df_train, df_test, target_test


def generate_list_superpixel(iparam):
    """
    generates a list indexes of a squared super-pixel covering the image
    :return:
    List of superpixel
    """
    def get_blocks(ix, iy, max_index):
        assert (ix < max_index)
        assert (iy < max_index)
        lst = [iparam.n_pixel * i + j for i in range(ix * iparam.block_size, ix * iparam.block_size + iparam.block_size)
               for j in range(iy * iparam.block_size, iy * iparam.block_size + iparam.block_size)]
        return lst
    max = iparam.max_index
    impute_cols = [get_blocks(ix=ix, iy=iy, max_index=max) for ix in range(max) for iy in range(max)]
    return impute_cols


def generate_list_superpixel_mask(iparam):
    n_pixel = iparam.n_pixel
    window_size = iparam.block_size
    list_mask = []
    for i in range(n_pixel)[::window_size]:
        for j in range(n_pixel)[::window_size]:
            mask = np.zeros((n_pixel, n_pixel), dtype=np.bool)
            mask[i:i+window_size, j:j+window_size] = True
            list_mask.append(mask)

    return list_mask


def generate_list_interaction(list_superpixel, i_reference: int):
    """
    get list of index between reference pixel and all other superpixels.
    :param list_superpixel: list of individual superpixels
    :param i_reference: index to reference pixel
    :return:
    list of interactions with reference pixel
    """
    reference_pixel = list_superpixel[i_reference]
    list_interaction = [[reference_pixel, superpixel] for superpixel in list_superpixel]
    del list_interaction[i_reference]        # remove self-interaction
    return list_interaction


def collect_relevances(m_list, max_index, key='mean'):
    # mean01
    m_list2 = np.stack([np.stack(m[key], axis=0) for m in m_list], axis=0)  # segment, example, classes
    m_list2 = np.transpose(m_list2, (1, 2, 0))  # example, class, segment
    return m_list2.reshape((m_list2.shape[0], m_list2.shape[1], max_index, max_index))


def get_relevances(explainer, data: np.ndarray, img_params):
    # calculate everything
    # list_superpixels = generate_list_superpixel(iparam=img_params)
    # m_list = explainer.relevances(data_test=data, list_masks=list_superpixels)

    list_superpixels = generate_list_superpixel_mask(iparam=img_params)
    m_list = explainer.relevances(data_test=data, list_masks=list_superpixels)

    # convert data format
    m_list_collected = collect_relevances(m_list=m_list, max_index=img_params.max_index)
    prediction_prob_classes = m_list[0]["pred"]
    return m_list_collected, prediction_prob_classes, m_list


def get_reference_pixel(m_relevance: np.ndarray, prediction_prob, img_id, n_importance=1) -> int:
    """returns index to reference pixel, n_importance allows to choose first (1), second (2), etc important pixel"""
    class_predicted = np.argmax(prediction_prob[img_id])
    m = m_relevance[img_id, class_predicted].flatten()
    arg_sort = np.argsort(np.abs(m))
    i_reference = int(arg_sort[-n_importance])
    return i_reference


def get_interaction(explainer, data: np.ndarray, iparam, m_list, i_reference, key='mean'):
    # list_superpixels = generate_list_superpixel(iparam=iparam)
    list_superpixels = generate_list_superpixel_mask(iparam=iparam)

    list_interaction = generate_list_interaction(list_superpixels, i_reference=i_reference)


    individual_contributions = True
    m_interaction = explainer.interactions(data_test=data, list_interaction_masks=list_interaction[:],
                                           individual_contributions=individual_contributions)

    m_interaction.insert(i_reference, m_interaction[0])
    m_interaction_collected = collect_relevances(m_list=m_interaction, max_index=iparam.max_index, key=key)

    temp_mean = np.array([tmp for tmp in m_list[i_reference]['mean']])

    # calculate self-interaction
    i_vertical, i_horizontal = divmod(i_reference, iparam.max_index)
    temp = m_interaction_collected[:, :, i_vertical, i_horizontal].copy()
    # assert np.alltrue(temp == temp_mean), 'selected wrong reference pixel'
    m_interaction_collected[:, :, i_vertical, i_horizontal] = 0     # temp - m_interaction_collected.sum(axis=(-1, -2))
    return m_interaction_collected


def imshow_digit(fig: plt.Figure, ax: plt.Axes, digit: np.ndarray, heatmap: np.ndarray, cbar=True, cbar_label=''):
    """plot digit with superimposed heatmap"""
    ax.imshow(digit, cmap=plt.cm.binary)
    vmx = np.amax(np.abs(heatmap))
    im = ax.imshow(heatmap, cmap=plt.cm.coolwarm, vmax=vmx, vmin=-vmx, alpha=0.75)
    if cbar is True:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_xlabel(cbar_label)
        cbar.ax.xaxis.set_label_position('top')

    # ax.set_axis_off()         remove frame, ticks and labels

    # leaves frame and removes ticks and labels
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def imshow_m_values(fig: plt.Figure, ax: plt.Axes, m_list_collected, data_digits: pd.DataFrame, img_params,
                    image_id: int, label_id: int, cbar=True, cbar_label=''):
    """plot plane single digit using custom data format generated by get_relevance/get_interaction"""
    digit = np.array(data_digits.iloc[image_id]).reshape(img_params.n_pixel, img_params.n_pixel)
    if isinstance(m_list_collected[image_id], pd.DataFrame):
        heatmap = np.kron(m_list_collected[image_id].iloc[label_id], np.ones((img_params.block_size, img_params.block_size)))
    else:
        heatmap = np.kron(m_list_collected[image_id, label_id], np.ones((img_params.block_size, img_params.block_size)))

    imshow_digit(fig=fig, ax=ax, digit=digit, heatmap=heatmap, cbar=cbar, cbar_label=cbar_label)


def plot_predicted_digit(relevance, interaction, prob_classes, img_params, data_digit: pd.DataFrame, image_id=17, imputer='',
                         rect=None, save=False, cbar=True):
    """plot digit with highest 'prob_classes' score"""
    prob = prob_classes.iloc[image_id]
    class_predicted = int(np.argmax(prob))

    title = f"{imputer}{data_digit.index[image_id]}"
    figsize = plt.rcParams['figure.figsize'].copy()
    figsize[1] = 0.7*figsize[1]
    fig = plt.figure(title, figsize=figsize)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('(a) relevance')
    imshow_m_values(fig=fig, ax=ax, m_list_collected=relevance, data_digits=data_digit, img_params=img_params,
                    image_id=image_id, label_id=class_predicted,
                    cbar=cbar, cbar_label=r"  ${\bar{m}}\,$")

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('(b) joint effect')
    imshow_m_values(fig=fig, ax=ax, m_list_collected=interaction, data_digits=data_digit, img_params=img_params,
                    image_id=image_id, label_id=class_predicted,
                    cbar=cbar, cbar_label=r"  $\,\,{\bar{m}}^{f_c^{\,Y\,Z}}$")
    if rect is not None:
        # ax.add_patch(copy.copy(rect))
        rect(ax)

    fig.tight_layout(pad=0.1)
    plt.tight_layout(pad=0.1)
    if save is True:
        fig.savefig(f'draft/figures/mnist/{title}', bbox_inches='tight')

    # plt.savefig(f"mnist_{title}.pdf")


def plot_comparison(m_list_collected, prob_classes, img_params, data_digit: pd.DataFrame, image_id=17, imputer='',
                    save=False, cbar=True):
    """plot predicted class and compare to three less likely class predictions"""
    title = f"Compare_{imputer}{data_digit.index[image_id]}"
    fig = plt.figure(title, figsize=[5.525058807250588, 1.7073370663614087])

    prob = prob_classes.iloc[image_id]
    mask_class_sorted = np.argsort(prob)
    for i, label_class in enumerate(mask_class_sorted[::-1]):     # start with most probable class
        if i == 4:
            break
        ax = fig.add_subplot(1, 4, i+1)
        imshow_m_values(fig=fig, ax=ax, m_list_collected=m_list_collected, data_digits=data_digit, img_params=img_params,
                        image_id=image_id, label_id=label_class, cbar=cbar)
        # prob = prob_classes.iloc[img_id][lbl_id]
        # title = f"label: {lbl_id}, {imputer}\n" \
        #         f"prob: {prob:.4f}"
        subplot_title = f'label = {label_class}\n' \
                        f'prob = {prob[label_class]:.1E}'
        ax.set_title(subplot_title)
    fig.tight_layout(pad=0.1)

    if save is True:
        fig.savefig(f'draft/figures/mnist/{title}', bbox_inches='tight')


def plot_filtersize_dependence(explainer, data_digit, img_param, cbar=False, imputer=''):
    list_block_size = [2, 4, 7, 14]
    iparam = copy.copy(img_param)

    fig_list = [plt.figure(f'Filtersize_{imputer}{image_id}') for image_id in data_digit.index]
    for i_ax, block_size in enumerate(list_block_size):
        iparam.block_size = block_size
        iparam.max_index = int(np.floor(iparam.n_pixel / block_size))
        m_list_collected, prediction_prob, m_list = get_relevances(explainer=explainer,
                                                                        data=data_digit, img_params=iparam)

        for image_id, fig in enumerate(fig_list):
            ax = fig.add_subplot(1, 4, i_ax + 1)
            prob = prediction_prob.iloc[image_id]
            class_predicted = int(np.argmax(prob))
            imshow_m_values(fig=fig, ax=ax, m_list_collected=m_list_collected, data_digits=data_digit,
                            img_params=iparam,
                            image_id=int(image_id), label_id=class_predicted, cbar=cbar)
            subplot_title = f'{(block_size, block_size)}'
            ax.set_title(subplot_title)
            fig.tight_layout(pad=0.1)


def plot_all_digits(m_list_collected, prob_classes, img_params, data, img_id=17, imputer='', rect=None):
    """plot all digits in one figure and add some information to subtitles"""
    fig, axs = plt.subplots(5, 2, figsize=(4.4, 9))
    for lbl_id in range(10):
        ax = axs[lbl_id % 5, lbl_id // 5]

        imshow_m_values(fig=fig, ax=ax, m_list_collected=m_list_collected, data_digits=data, img_params=img_params,
                        image_id=img_id, label_id=lbl_id)

        prob = prob_classes.iloc[img_id][lbl_id]
        title = f"label: {lbl_id}, {imputer}\n" \
                f"prob: {prob:.4f}"
        ax.set_title(title)
        if rect is not None:
            # ax.add_patch(copy.copy(rect))
            rect(ax)
    fig.tight_layout(pad=1.)


def plot_rect(ax: plt.Axes, i_reference, iparam):
    i_vertical, i_horizontal = divmod(i_reference, iparam.max_index)
    masked_data = np.ones((28, 28))
    mask = np.zeros((28, 28))
    mask[i_vertical * iparam.block_size:(i_vertical + 1) * iparam.block_size,
    i_horizontal * iparam.block_size:(i_horizontal + 1) * iparam.block_size
    ] = 1
    masked_data = np.ma.masked_where(mask != 1, masked_data)
    ax.imshow(masked_data, interpolation='nearest', cmap='Accent')


def plot_imputations(imputer: ImputerBase, image: np.ndarray, mask: np.ndarray, i_sample=0):
    # prepare imputations
    n_imputations = 4
    imputations, _ = imputer.impute(test_data=image, mask_impute=mask, n_imputations=n_imputations)

    image_imputed = np.array([image.copy() for _ in range(n_imputations)])
    image_imputed[:, :, mask] = imputations[:, :, mask]

    fig = plt.figure(f'{imputer.imputer_name}_{i_sample}', figsize=(7, 1.4))
    n_cols = n_imputations + 1
    ax = plt.subplot(1, n_cols, 1)
    # ax.set_title('original')
    cbar = ax.imshow(image[i_sample], cmap=plt.cm.binary)
    # plt.colorbar(cbar)

    mask_one_channel = mask.copy()
    # assert np.alltrue(np.unique(mask_one_channel) == [0, 1]), 'mask needs to cover uniformly cover all channels'

    masked_data = np.ones_like(mask_one_channel)
    masked_data = np.ma.masked_where(mask_one_channel != 1, masked_data)
    ax.imshow(masked_data, alpha=0.8, interpolation='nearest', cmap='Set1')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for i_imputation in range(n_imputations):
        ax = plt.subplot(1, n_cols, 2+i_imputation)
        # ax.set_title(f'i = {i_imputation}')
        cbar = ax.imshow(image_imputed[i_imputation, i_sample], cmap=plt.cm.binary)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # plt.colorbar(cbar)

    fig.tight_layout(pad=0.1)

