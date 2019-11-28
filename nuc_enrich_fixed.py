from __future__ import division

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import sh
import warnings

from pims import Bioformats
from PIL import Image
import skimage.filters as sf
import skimage.io as si
import skimage.measure as sme
import skimage.morphology as smo
import scipy.ndimage as sni
import cPickle as pickle


def load_TIF_series(filename):
    raw_czyx = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]

    return name, raw_czyx


def save(img, filename):
    """
    Save image using PIL
    """
    if len(img.shape) == 2:
        Image.fromarray(img).save(filename)
    else:
        assert len(img.shape) == 3

        # https://github.com/python-pillow/Pillow/pull/2406
        imlist = map(Image.fromarray, img)
        imlist[0].save(filename, save_all=True, append_images=imlist[1:])


def compute_maxproj_mask(lim_zyx):
    """
    Construct a cell mask based on the maximum intensity projection of the lim
    stack. This mask will be used to pick the nuclei in cells that overexpress
    the lim protein.

    lim_zyx = lim stack

    return = binarized maximum projection of lim stack
    """
    lim_mip_yx = np.sum(lim_zyx, axis=0)
    lim_thresh = sf.threshold_li(lim_mip_yx)
    lim_mask_yx = smo.binary_opening(
        sni.morphology.binary_fill_holes(lim_mip_yx > lim_thresh))
    return lim_mask_yx


def select_z_using_lim_mask(nuc_mask_zyx, lim_mask_yx, area_thresh):
    """
    Find the slice with the maximum nuclear area in bright cells.

    nuc_mask_zyx = binarized nuc stack
    lim_mask_yx = binarized maximum projection of lim stack
    area_thresh = area threshold below which to ignore

    return = index of the slice and nuclear mask for ratio computation
    """
    clean_mask_zyx = []
    for z, nuc_mask_yx in enumerate(nuc_mask_zyx):
        nuc_labeled_yx = sme.label(
            smo.binary_opening(sni.morphology.binary_fill_holes(nuc_mask_yx)))
        label_r = [props.label for props in sme.regionprops(nuc_labeled_yx)
                   if props.area >= area_thresh]

        if len(label_r) == 0:
            clean_mask_zyx.append(np.zeros_like(nuc_mask_yx))
            continue

        # For pixels masked out by lim_mask_yx, set label to zero (background).
        labeled_yx = nuc_labeled_yx * lim_mask_yx
        label_max = max(label_r, key=lambda label: (labeled_yx == label).sum())
        clean_mask_zyx.append(labeled_yx == label_max)

    score_z = map(np.sum, clean_mask_zyx)

    z_max_score = np.argmax(score_z)
    return z_max_score


def compute_nuc_mask(nuc_sm_zyx, z_max_score, lim_mask_yx, area_thresh, sigma):
    nuc_sm_yx = nuc_sm_zyx[z_max_score]
    nuc_mask_yx = nuc_sm_yx > sf.threshold_isodata(nuc_sm_yx[lim_mask_yx])
    nuc_labeled_yx = sme.label(
        smo.binary_opening(sni.morphology.binary_fill_holes(nuc_mask_yx)))
    label_r = [props.label for props in sme.regionprops(nuc_labeled_yx)
               if props.area >= area_thresh]
    labeled_yx = nuc_labeled_yx * lim_mask_yx
    label_max = max(label_r, key=lambda label: (labeled_yx == label).sum())
    nuc_mask_yx = (labeled_yx == label_max)

    return nuc_mask_yx


def dilate_nuc_mask(nuc_mask_yx, lim_mask_yx, dilat_factor):
    """
    Dilate nuc_mask until the area is twice the original area

    nuc_mask_yx = nuclear mask for ratio computation
    lim_mask_yx = binarized maximum projection of lim stack
    dilat_factor = dilation factor

    return = dilated nuclear mask
    """
    area_threshold = dilat_factor * np.sum(nuc_mask_yx)

    dilated_yx = nuc_mask_yx
    area_prev = dilated_yx.sum()
    while True:
        dilated_yx = smo.binary_dilation(dilated_yx) & lim_mask_yx
        area_dilated = dilated_yx.sum()

        if area_dilated >= area_threshold:
            break

        # Area doesn't grow any more.
        if area_dilated == area_prev:
            break
        area_prev = area_dilated

    return dilated_yx


def compute_local_ratio(raw_czyx, dilat_factor, area_thresh, sigma):
    """
    Compute the nuclear enrichment of LIM proteins.
    Nuclear enrichment = average intensity within nucleus / average intensity
    in local cytosol

    raw_czyx = multichannel stack, with nuc in [0] and lim in [1]

    return = intensity ratio, nucleus frame with the largest nuclear area, lim
             frame used for computation, index of the slice used for
             computation
    """
    assert isinstance(raw_czyx, np.ndarray)
    assert len(raw_czyx.shape) == 4

    # Compute lim mask by thresholding lim mip
    lim_zyx = raw_czyx[1]
    lim_mask_yx = compute_maxproj_mask(lim_zyx)

    # Compute nuc mask by thresholding the region covered by lim mask
    nuc_zyx = raw_czyx[0]
    nuc_sm_zyx = np.asarray([sf.gaussian(nuc_yx, sigma) for nuc_yx in nuc_zyx])
    nuc_roi_p = np.concatenate([nuc_sm_yx[lim_mask_yx] for nuc_sm_yx in nuc_sm_zyx])
    nuc_mask_zyx = nuc_sm_zyx > sf.threshold_isodata(nuc_roi_p)

    # Use the slice with the maximum nuc area to compute lim ratio.
    z_max_nuc = select_z_using_lim_mask(nuc_mask_zyx, lim_mask_yx, area_thresh)
    lim_yx = raw_czyx[1, z_max_nuc]

    # Exclude images with more than 500 saturated pixels
    if np.sum(lim_yx == 4095) > 500:
        return

    actin_yx = raw_czyx[2, z_max_nuc]
    nuc_mask_yx = compute_nuc_mask(nuc_sm_zyx, z_max_nuc, lim_mask_yx,
                                   area_thresh, sigma)

    # Dilate the nuc mask to produce the cyto mask.
    dilated_yx = dilate_nuc_mask(nuc_mask_yx, lim_mask_yx, dilat_factor)
    cyto_mask_yx = dilated_yx & (nuc_mask_yx == 0)

    # Compute actin mask by thresholding the region covered by lim_mask_yx
    actin_thresh = sf.threshold_li(actin_yx[lim_mask_yx])
    actin_mask_yx = actin_yx > actin_thresh

    # Compute clean nuc mask by excluding actin region
    nuc_clean_mask_yx = nuc_mask_yx & (~ actin_mask_yx)

    # Compute clean cyto mask by excluding actin region
    cyto_clean_mask_yx = cyto_mask_yx & (~ actin_mask_yx)

    # Compute the ratio of average intensity in the lim channel inside to
    # outside of the nucleus.
    nuc_area = np.sum(nuc_clean_mask_yx)
    cyto_area = np.sum(cyto_clean_mask_yx)
    lim_nuc_avg = np.sum(nuc_clean_mask_yx * lim_yx) / nuc_area
    lim_cyto_avg = np.sum(cyto_clean_mask_yx * lim_yx) / cyto_area
    nuc_ratio_local = lim_nuc_avg / lim_cyto_avg

    return (nuc_ratio_local, z_max_nuc, lim_mask_yx, actin_mask_yx,
            nuc_clean_mask_yx, cyto_clean_mask_yx)


def compute_local_ratio_for_TIF(folder_tif, dilat_factor=2,
                                area_thresh=5000, sigma=3):
    folder_computed = op.join('output/fixed/computed-slice/local',
                              op.basename(folder_tif))
    sh.mkdir('-p', folder_computed)

    ratio_s = []
    name_s = []
    print op.basename(folder_tif)

    for tif_filename in sorted(glob(op.join(folder_tif, 'Series*.tif'))):
        name, raw_czyx = load_TIF_series(tif_filename)
        if compute_local_ratio(raw_czyx, dilat_factor, area_thresh, sigma) is None:
            continue
        else:
            (nuc_ratio_local, z_max_nuc, lim_mask_yx, actin_mask_yx, nuc_clean_mask_yx,
             cyto_clean_mask_yx) = compute_local_ratio(raw_czyx, dilat_factor,
                                                       area_thresh, sigma)
            ratio_s.append(nuc_ratio_local)
            name_s.append(name)
            print name, 'nuc_ratio_local:', nuc_ratio_local, 'z_max_nuc:', z_max_nuc

            if name.startswith('Series'):
                suffix = name[len('Series'):]
            lim_mask_filename = op.join(folder_computed, 'lim-mask-' + suffix +
                                        '-z%d' % z_max_nuc + '.tiff')
            actin_mask_filename = op.join(folder_computed, 'actin-mask-' + suffix +
                                          '-z%d' % z_max_nuc + '.tiff')
            nuc_clean_mask_filename = op.join(folder_computed, 'nuc-clean-mask-' +
                                              suffix + '-z%d' % z_max_nuc +
                                              '.tiff')
            cyto_clean_mask_filename = op.join(
                folder_computed, 'cyto-clean-mask-' + suffix + '-z%d' % z_max_nuc +
                '.tiff')

            save(lim_mask_yx.astype(np.uint16), lim_mask_filename)
            save(actin_mask_yx.astype(np.uint16), actin_mask_filename)
            save(nuc_clean_mask_yx.astype(np.uint16), nuc_clean_mask_filename)
            save(cyto_clean_mask_yx.astype(np.uint16), cyto_clean_mask_filename)

    folder_pkl = 'output/fixed/ratio-pkl/local'
    sh.mkdir('-p', folder_pkl)
    output_pkl = op.join(folder_pkl, op.basename(folder_tif) + '.pkl')

    output = dict(
        name_s=name_s,
        ratio_s=ratio_s,
    )

    with open(output_pkl, 'w') as f:
        pickle.dump(output, f)

    folder_txt = 'output/fixed/ratio-txt/local'
    sh.mkdir('-p', folder_txt)
    output_txt = op.join(folder_txt, op.basename(folder_tif) + '.txt')

    with open(output_txt, 'w') as f:
        for s, name in enumerate(name_s):
            f.write('%-9s  %.4f\n' % (name, ratio_s[s]))


def compute_global_ratio(raw_czyx, area_thresh, sigma):
    assert isinstance(raw_czyx, np.ndarray)
    assert len(raw_czyx.shape) == 4

    # Compute lim mask by thresholding lim mip
    lim_zyx = raw_czyx[1]
    lim_mask_yx = compute_maxproj_mask(lim_zyx)

    # Compute nuc mask by thresholding the region covered by lim mask
    nuc_zyx = raw_czyx[0]
    nuc_sm_zyx = np.asarray([sf.gaussian(nuc_yx, sigma) for nuc_yx in nuc_zyx])
    nuc_roi_p = np.concatenate([nuc_sm_yx[lim_mask_yx] for nuc_sm_yx in nuc_sm_zyx])
    nuc_mask_zyx = nuc_sm_zyx > sf.threshold_isodata(nuc_roi_p)

    # Use the slice with the maximum nuc area to compute lim ratio.
    z_max_nuc = select_z_using_lim_mask(nuc_mask_zyx, lim_mask_yx, area_thresh)
    lim_yx = raw_czyx[1, z_max_nuc]
    if np.sum(lim_yx == 4095) > 500:
        return

    actin_yx = raw_czyx[2, z_max_nuc]
    nuc_mask_yx = compute_nuc_mask(nuc_sm_zyx, z_max_nuc, lim_mask_yx,
                                   area_thresh, sigma)

    # Compute actin mask by thresholding the region covered by lim_mask_yx
    actin_thresh = sf.threshold_li(actin_yx[lim_mask_yx])
    actin_mask_yx = actin_yx > actin_thresh

    # Compute cell mask
    cell_thresh = sf.threshold_li(lim_yx)
    cell_mask_yx = smo.binary_opening(
        sni.morphology.binary_fill_holes(lim_yx > cell_thresh))

    # Compute clean masks
    nuc_clean_mask_yx = nuc_mask_yx & (~ actin_mask_yx)
    actin_clean_mask_yx = actin_mask_yx & (~ nuc_mask_yx)
    cyto_clean_mask_yx = cell_mask_yx & (~ nuc_mask_yx) & (~ actin_mask_yx)

    # Compute the ratio of average intensity in the lim channel inside to
    # outside of the nucleus.
    nuc_area = np.sum(nuc_clean_mask_yx)
    actin_area = np.sum(actin_clean_mask_yx)
    cyto_area = np.sum(cyto_clean_mask_yx)

    lim_nuc_avg = np.sum(nuc_clean_mask_yx * lim_yx) / nuc_area
    lim_actin_avg = np.sum(actin_clean_mask_yx * lim_yx) / actin_area
    lim_cyto_avg = np.sum(cyto_clean_mask_yx * lim_yx) / cyto_area

    nuc_ratio_global = lim_nuc_avg / lim_cyto_avg
    actin_ratio_global = lim_actin_avg / lim_cyto_avg

    return (nuc_ratio_global, actin_ratio_global, z_max_nuc,
            actin_clean_mask_yx, nuc_clean_mask_yx, cyto_clean_mask_yx)


def compute_global_ratio_for_TIF(folder_tif, area_thresh=5000, sigma=3):
    folder_computed = op.join('output/fixed/computed-slice/global',
                              op.basename(folder_tif))
    sh.mkdir('-p', folder_computed)

    nuc_ratio_s = []
    actin_ratio_s = []
    name_s = []
    print op.basename(folder_tif)

    for tif_filename in sorted(glob(op.join(folder_tif, 'Series*.tif'))):
        name, raw_czyx = load_TIF_series(tif_filename)
        if compute_global_ratio(raw_czyx, area_thresh, sigma) is None:
            continue
        else:
            (nuc_ratio_global, actin_ratio_global, z_max_nuc, actin_clean_mask_yx,
             nuc_clean_mask_yx, cyto_clean_mask_yx) = compute_global_ratio(
                 raw_czyx, area_thresh, sigma)

            nuc_ratio_s.append(nuc_ratio_global)
            actin_ratio_s.append(actin_ratio_global)
            name_s.append(name)
            print name, 'nuc_ratio:', nuc_ratio_global, 'actin_ratio:',
            print actin_ratio_global, 'z:', z_max_nuc

            if name.startswith('Series'):
                suffix = name[len('Series'):]
            actin_clean_mask_filename = op.join(folder_computed,
                                                'actin-clean-mask-' + suffix +
                                                '-z%d' % z_max_nuc + '.tiff')
            nuc_clean_mask_filename = op.join(folder_computed, 'nuc-clean-mask-' +
                                              suffix + '-z%d' % z_max_nuc +
                                              '.tiff')
            cyto_clean_mask_filename = op.join(folder_computed,
                                               'cyto-clean-mask-' + suffix + '-z%d'
                                               % z_max_nuc + '.tiff')

            save(actin_clean_mask_yx.astype(np.uint16), actin_clean_mask_filename)
            save(nuc_clean_mask_yx.astype(np.uint16), nuc_clean_mask_filename)
            save(cyto_clean_mask_yx.astype(np.uint16), cyto_clean_mask_filename)

    folder_pkl = 'output/fixed/ratio-pkl/global'
    sh.mkdir('-p', folder_pkl)
    output_pkl = op.join(folder_pkl, op.basename(folder_tif) + '.pkl')

    output = dict(
        name_s=name_s,
        nuc_ratio_s=nuc_ratio_s,
        actin_ratio_s=actin_ratio_s,
    )

    with open(output_pkl, 'w') as f:
        pickle.dump(output, f)

    folder_txt = 'output/fixed/ratio-txt/global'
    sh.mkdir('-p', folder_txt)
    output_txt = op.join(folder_txt, op.basename(folder_tif) + '.txt')

    with open(output_txt, 'w') as f:
        for s, name in enumerate(name_s):
            f.write('%-9s  %.4f  %.4f\n' % (name, nuc_ratio_s[s],
                                            actin_ratio_s[s]))


if __name__ == '__main__':
    for folder_tif in sorted(glob('data/nuc-enrich-fixed/*/*')):
        compute_local_ratio_for_TIF(folder_tif)
