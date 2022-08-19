import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import skimage.measure
from skimage.segmentation import slic

from ..imputers.imputer_base import ImputerBase
from . import utils_imagenet
from captum.attr import visualization as viz


def plot_imputations(dic):
    try:
        imputer = dic['imputer']
        n_cols = dic['n_example_imputations'] + 1
    except:
        imputer = dic["args"].imputer
        n_cols = dic['n_imputations'] + 1
    fig = plt.figure(f'visualize_{imputer}', figsize=(10, 8))
    # ax.set_title('original')
    for i_image in range(5):
        ax = plt.subplot(5, n_cols, i_image * n_cols + 1)
        ax.imshow(dic[f'image_original_{i_image}'].transpose(1, 2, 0))
        mask_one_channel = dic[f'mask_{i_image}'].sum(axis=0) / 3
        assert np.alltrue(np.unique(mask_one_channel) == [0, 1]), 'mask needs to cover uniformly cover all channels'

        masked_data = np.ones_like(mask_one_channel)
        masked_data = np.ma.masked_where(mask_one_channel != 1, masked_data)
        ax.imshow(masked_data, alpha=0.8, interpolation='nearest', cmap='Set1')
        plt.axis(False)

        image_imputed = dic[f'image_imputed_{i_image}']
        for i_imputation in range(dic['n_example_imputations']):
            ax = plt.subplot(5, n_cols, i_image*n_cols + 2+i_imputation)
            # ax.set_title(f'i = {i_imputation}')
            ax.imshow(image_imputed[i_imputation, 0].transpose(1, 2, 0))
            plt.axis(False)

    fig.tight_layout(pad=0.1)


def get_imputation(image, imputer, percentage: float, n_segments: int):
    segmentation = slic(image.transpose(1, 2, 0), n_segments=n_segments, compactness=20, start_label=1)
    seg_unique = np.unique(segmentation)
    n_seg = len(seg_unique)
    mask_flat = np.zeros_like(segmentation)
    if percentage >= 1:
        n_current = percentage
    else:
        n_current = int(percentage * n_seg)
    seg_impute = np.random.choice(seg_unique, n_current, replace=False)
    print(f'For {percentage:.2%} there are {len(seg_impute)} imputed segments.')
    for i_seg in seg_impute:
        mask_temp = segmentation == i_seg
        mask_flat[mask_temp] = 1
    mask = np.broadcast_to(mask_flat[None], shape=(3, *mask_flat.shape))
    imputations, _ = imputer.impute(test_data=np.array(image[None]), mask_impute=mask, n_imputations=1)
    return np.array(np.squeeze(imputations), dtype=np.int), len(seg_unique)


def visualize_imputer_on_the_fly(imputer, image: np.ndarray, n_segments: int):
    """This function creates imputations and visualizes them for small to large image occupancy."""
    assert image.shape[0] == 3

    title = f'Imputations {imputer.imputer_name}_superpixel{n_segments}'
    list_nseg_or_percentage = [1, 3, 0.25, 0.5, 0.75]
    n_col = len(list_nseg_or_percentage) + 1
    # plt.figure('imputation')
    size = 1
    fig, ax_np = plt.subplots(1, n_col, figsize=(n_col*size, 1.5*size), num=title)
    for i, ax in enumerate(ax_np):
        if i > 0:
            percentage = list_nseg_or_percentage[i-1]
            imputations, n_actual_segments = get_imputation(image, imputer, percentage=percentage, n_segments=n_segments)
            ax.imshow(imputations.transpose(1, 2, 0))
            if percentage >= 1:
                ax.set_title(r'$\frac{' + str(percentage) + '}{' + str(n_actual_segments) + '} = $' + f'{percentage/n_actual_segments:.1%}')
            else:
                ax.set_title(f'{percentage:.0%}')
        else:
            ax.imshow(image.transpose(1, 2, 0))
            ax.set_title(f'Original')

        ax.tick_params(left=False,
                       bottom=False,
                       labelleft=False,
                       labelbottom=False)

    plt.tight_layout(pad=0.2)
    # plt.figure('mask')
    # plt.imshow(mask_flat, alpha=0.1)




def segment_centroids(segments):
    props = skimage.measure.regionprops(segments+1)
    label = np.array([p.label for p in props]) - 1 #regions with label 0 are ignored
    centroids = np.array([p.centroid for p in props])
    centroids = centroids[label]
    return centroids

def hanging_line(point1, point2):
    # from https://stackoverflow.com/questions/63560005/draw-curved-lines-to-connect-points-in-matplotlib
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b
    return (x,y)

def bezier_curve(p1,p2, color, alpha, indent=0.2, lw=1.0):
    x = np.array([p1[0],p2[0]])
    y = np.array([p1[1],p2[1]])
    dist = ((x-y)**2).sum()**0.5
    
    Path = mpath.Path
    indent = indent*dist
    verts = [(xi + d, yi) for xi, yi in zip(x,y) for d in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', lw=lw, edgecolor=color, alpha=alpha)
    return patch

def m_values(dic_data, interactions=False, int_line="bezier", n_int=2, int_lw=1.0, int_ms=2.0, only_int_segments=False, norm=False):
    n = dic_data['n_images']
    args = dic_data['args']
    #preds = dic_data['preds'] if 'preds' in dic_data else None
    fig = plt.figure(args.imputer, figsize=(10, 2))
    for i in range(n):
        image = dic_data[f'image_{i}'].transpose(1, 2, 0)
        seg = dic_data[f'seg_{i}']
        relevance = dic_data[f'relevance_{i}']

        rel = np.zeros_like(seg).astype(float)
        for s, mvalue in enumerate(relevance):
            rel[seg == s] = mvalue

        ax = plt.subplot(1, n, i+1)

        if (interactions or norm):
            vmax = 1
        else:
            vmax = np.abs(rel).max()

        if(interactions):
            segment_cache = []
            centroids = segment_centroids(seg)
            root_markers = ["P","s","^","<", ">", "o","D"]
            for j,ref_s in enumerate(dic_data[f'reference_{i}']):
                segment_cache.append(ref_s)
                interaction = dic_data[f"interaction_{i}_{ref_s}"]
                interaction_ranking = interaction.argsort()[::-1]
                for s, color in zip(np.concatenate([interaction_ranking[-n_int:], interaction_ranking[:n_int]]), (["blue"]*n_int+["red"]*n_int)):
                    segment_cache.append(s)
                    point_root = [centroids[ref_s,1], centroids[ref_s,0]]
                    point_partner = [centroids[s,1], centroids[s,0]]
                    rel_intensity = np.clip(np.sqrt(np.abs(interaction[s])/relevance.max()), 0 ,1)
                    if int_line=="straightline":
                        ax.plot(x, y, color="red", lw=int_lw, alpha=rel_intensity)
                    elif int_line=="caternary":
                        x,y = hanging_line(point_root, point_partner)
                        ax.plot(x, y, color=color, lw=int_lw,  alpha=rel_intensity)
                    elif int_line=="bezier":
                        patch = bezier_curve(point_root,point_partner, color=color, alpha=rel_intensity, lw=int_lw)
                        ax.add_patch(patch)
                    ax.plot(point_root[0], point_root[1], marker=root_markers[j], ms=int_ms, color=color)#,  alpha=rel_intensity)
                    ax.plot(point_partner[0], point_partner[1], marker='.', ms=int_ms, color=color) #,  alpha=rel_intensity)
            ax.set_xlim(0,image.shape[0])
            #if preds is not None:
            #    ax.set_title(str(preds[i]))
            if only_int_segments:
                mask = np.isin(seg, segment_cache)
                rel[~mask] = 0

        cbar = ax.imshow(rel, alpha=1, cmap="seismic", vmax=vmax, vmin=-vmax)
        if not (interactions or norm):
            plt.colorbar(cbar, fraction=0.046, pad=0.04)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(image, alpha=0.3)

    plt.tight_layout(pad=0.1)


def standard_captum(dic):
    """This is just a wrapper for the standard captum visualizations"""
    attributions = dic['attributions']
    images_np = dic['images']
    for i, img in enumerate(images_np):
        attributions_image = attributions[i]
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(img, (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap='binary',
                                              show_colorbar=True)


def convert_to_superpixel_relevance(image: np.ndarray, pixelwise_attributions, n_superpixel: int,
                                    attributions_normed=True) -> np.ndarray:
    masks, seg = utils_imagenet.generate_superpixel_slic(image, n_segments=n_superpixel)

    rel = np.zeros_like(seg).astype(float)
    for s in range(seg.max()):
        mask_superpixel = seg == s
        attrr_superpixel = pixelwise_attributions[:, mask_superpixel]
        # WARNING: this is not unique
        if attributions_normed is True:
            rel_superpixel = np.sum(np.abs(attrr_superpixel))  # convert to a single relevance
        else:
            rel_superpixel = np.sum(attrr_superpixel)  # convert to a single relevance

        rel[mask_superpixel] = rel_superpixel

    # scale to one
    relevance_superpixel = rel / rel.max()
    return relevance_superpixel


def gradientbased_attributions(dic, attributions_normed=True):
    """Aggregate the pixelwise attributions to superpixel and display in PredDiff style"""
    attributions = dic['attributions']
    images_np = dic['images']

    # plot attributions with superpixel
    fig = plt.figure(f'gradientbased_{dic["method"]}', figsize=(10, 6))
    for i_image, image in enumerate(images_np):
        print(f'{i_image}')
        if isinstance(attributions, np.ndarray) is False:
            attributions_image = np.array(attributions[i_image].detach())
        else:
            attributions_image = attributions[i_image]
        for i_superpixel, n_superpixel in enumerate([128, 1000, 10000]):
            # aggregate attributions according to superpixels
            rel = convert_to_superpixel_relevance(image=image, pixelwise_attributions=attributions_image,
                                                  n_superpixel=n_superpixel, attributions_normed=attributions_normed)

            # plot
            ax = plt.subplot(3, 5, i_image + i_superpixel * 5 + 1)

            vmax = np.abs(rel).max()
            if rel.min() >= 0:
                cbar = ax.imshow(rel, alpha=1, cmap='Reds', vmax=vmax, vmin=0)
            else:
                cbar = ax.imshow(rel, alpha=1, cmap='seismic', vmax=vmax, vmin=-vmax)
            plt.colorbar(cbar, fraction=0.046, pad=0.04)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.imshow(np.transpose(image, (1, 2, 0)), alpha=0.3)
    plt.tight_layout(pad=0.5)


def superpixel_resolution_dependece(dic_data):
    n = dic_data['n_images']
    args = dic_data['args']

    # convert image and superpixel-wise attributions to a single pixelwise array
    images_np = np.stack([dic_data[f'image_{i}'] for i in range(n)])
    segmentation_np = np.stack([dic_data[f'seg_{i}'] for i in range(n)])

    attributions_list = []
    for i in range(n):
        relevance_superpixel = dic_data[f'relevance_{i}']
        seg = segmentation_np[i]
        pixelwise_attributions = np.zeros_like(images_np[0], dtype=np.float)
        for i_seg in range(seg.max()):
            mask = np.stack([seg == i_seg for _ in range(3)])
            pixelwise_attributions[mask] = relevance_superpixel[i_seg]
        attributions_list.append(pixelwise_attributions)

    attributions_np = np.stack(attributions_list)
    dic_data['attributions'] = attributions_np
    dic_data['images'] = images_np
    dic_data['method'] = args.imputer
    # plt.figure()
    # # plt.imshow(np.transpose(attributions_np[0], (1, 2, 0)), alpha=1)
    # rel = np.transpose(attributions_np[0], (1, 2, 0))
    # vmax = np.abs(rel).max()
    # cbar = plt.imshow(rel[:,:,0], alpha=1, cmap="seismic", vmax=vmax, vmin=-vmax)
    # plt.colorbar()
    # plt.imshow(np.transpose(images_np[0], (1, 2, 0)), alpha=0.3)

    gradientbased_attributions(dic=dic_data, attributions_normed=False)


def occlude_single_image(dic_pixelflipping, i_image: int):
    plt.figure()
    plt.title(f'Occlude with {dic_pixelflipping["imputer_occluding"]}, n_meas = {dic_pixelflipping["n_measurements"]}')
    list_imputers = dic_pixelflipping['imputer_attributions']
    for imputer_attributions in list_imputers:
        probability = dic_pixelflipping[f'{imputer_attributions}_probability_{i_image}']
        x = dic_pixelflipping[f'{imputer_attributions}_percentage_occluded_{i_image}']
        plt.plot(x, probability, label=imputer_attributions)
    plt.semilogy()
    plt.legend()


def average_faithfulness(dic_pixelflipping):
    n_images = dic_pixelflipping['n_images']
    plt.figure(f'{dic_pixelflipping["imputer_occluding"]}_measurements_{dic_pixelflipping["n_measurements"]}_{dic_pixelflipping["high_to_low"]}')
    plt.title(f'Occlude with {dic_pixelflipping["imputer_occluding"]}, n_meas: {dic_pixelflipping["n_measurements"]}, n_images: {n_images}')
    plt.title(f'{dic_pixelflipping["imputer_occluding"]}, highest relevance first: {dic_pixelflipping["high_to_low"]}')
    list_imputers = dic_pixelflipping['imputer_attributions']
    for imputer_attributions in list_imputers:
        all_probabilities = np.array([dic_pixelflipping[f'{imputer_attributions}_probability_{i}']
                                      for i in range(n_images)])
        # all_probabilities_stack = np.stack([dic_pixelflipping[f'{imputer_attributions}_probability_{i}']
        #                               for i in range(n_images)])
        all_x = np.array([dic_pixelflipping[f'{imputer_attributions}_percentage_occluded_{i}']
                          for i in range(n_images)])
        mean_probability = all_probabilities.mean(axis=0)
        error_probability = all_probabilities.std(axis=0)/np.sqrt(n_images)
        mean_x = all_x.mean(axis=0)
        # plt.errorbar(mean_x, mean_probability, error_probability, label=imputer_attributions)
        # solid stripe to indicate errors
        [line_plot] = plt.plot(mean_x, mean_probability, '.', label=imputer_attributions)
        plt.fill_between(mean_x, mean_probability - error_probability, mean_probability + error_probability,
                         color=line_plot.get_color(), alpha=0.2)
    # plt.semilogy()
    plt.loglog()
    plt.xlabel('Percentage occluded')
    plt.ylabel('Probability prediction')
    plt.legend()
    plt.tight_layout(pad=0.3)

