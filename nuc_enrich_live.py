from __future__ import division

import cPickle as pickle
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


def load_image(fn_lim, fn_actin, fn_nuc):
    lim_tyx = si.imread(fn_lim)
    print 'Loading', fn_lim
    actin_tyx = si.imread(fn_actin)
    print 'Loading', fn_actin
    nuc_tyx = si.imread(fn_nuc)
    print 'Loading', fn_nuc
    print 'Loading done'
    return lim_tyx, actin_tyx, nuc_tyx


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


def dilate_nuc_mask(nuc_mask_yx, lim_mask_yx, dilat_factor):
    area_threshold = dilat_factor * np.sum(nuc_mask_yx)
    nuc_dilated_yx = nuc_mask_yx
    area_prev = nuc_dilated_yx.sum()

    while True:
        nuc_dilated_yx = smo.binary_dilation(nuc_dilated_yx) & lim_mask_yx
        area_dilated = nuc_dilated_yx.sum()

        if area_dilated >= area_threshold:
            break

        # Area doesn't grow any more.
        if area_dilated == area_prev:
            break
        area_prev = area_dilated

    return nuc_dilated_yx


def compute_lim_ratio(fn_lim, fn_actin, fn_nuc, area_thresh=1000,
                      dilat_factor=2):
    lim_tyx, actin_tyx, nuc_tyx = load_image(fn_lim, fn_actin, fn_nuc)

    folder_computed = op.join('output/live/mask',
                              op.split(op.split(op.abspath(fn_lim))[0])[1])
    sh.mkdir('-p', folder_computed)

    # Compute global nuclear enrichment
    cell_tyx = lim_tyx
    cell_mask_tyx = []
    print 'Computing cell masks'
    for t, cell_yx in enumerate(cell_tyx):
        cell_sm_yx = sf.gaussian(cell_yx, sigma=3)
        cell_mask_yx = smo.binary_opening(sni.morphology.binary_fill_holes(
            (cell_sm_yx > sf.threshold_triangle(cell_sm_yx))))
        cell_mask_tyx.append(cell_mask_yx)

    nuc_roi_mask_tyx = []
    print 'Computing nuclear roi masks'
    for t, (nuc_yx, cell_mask_yx) in enumerate(zip(nuc_tyx, cell_mask_tyx)):
        nuc_sm_yx = sf.gaussian(nuc_yx, sigma=3)
        nuc_thresh = sf.threshold_isodata(nuc_sm_yx[cell_mask_yx])
        nuc_mask_yx = smo.binary_opening(sni.morphology.binary_fill_holes(
            nuc_sm_yx > nuc_thresh))
        nuc_labeled_yx = sme.label(nuc_mask_yx)

        label_r = []
        for props in sme.regionprops(nuc_labeled_yx,
                                     intensity_image=cell_mask_yx):
            if (props.area >= area_thresh) and props.min_intensity:
                label_r.append(props.label)

        assert label_r

        labeled_yx = nuc_labeled_yx * cell_mask_yx
        label_sum_r = [(labeled_yx == label).sum() for label in label_r]
        label_max = label_r[np.argmax(label_sum_r)]
        nuc_roi_mask_yx = (labeled_yx == label_max)
        nuc_roi_mask_tyx.append(nuc_roi_mask_yx)

    cell_roi_mask_tyx = []
    print 'Computing cell roi masks'
    for t, (cell_mask_yx, nuc_roi_mask_yx) in enumerate(
        zip(cell_mask_tyx, nuc_roi_mask_tyx)
    ):
        cell_labeled_yx = sme.label(cell_mask_yx)
        label_cell_r = list(set(cell_labeled_yx[nuc_roi_mask_yx]))
        assert len(label_cell_r) == 1
        cell_roi_mask_yx = (cell_labeled_yx == label_cell_r[0])
        cell_roi_mask_tyx.append(cell_roi_mask_yx)

        cell_roi_mask_yx_fn = op.join(folder_computed,
                                      'cell-roi-mask-%03d.tiff' % t)
        save((cell_roi_mask_yx * (-1)).astype(np.uint16), cell_roi_mask_yx_fn)

    actin_tyx = list(actin_tyx)
    actin_roi_mask_tyx = []
    actin_clean_mask_tyx = []
    nuc_clean_mask_tyx = []
    cyto_clean_mask_tyx = []
    print 'Computing clean actin, nuc, and cyto masks'
    for t, (actin_yx, nuc_roi_mask_yx, cell_roi_mask_yx) in enumerate(
        zip(actin_tyx, nuc_roi_mask_tyx, cell_roi_mask_tyx)
    ):
        actin_thresh = sf.threshold_li(actin_yx[cell_roi_mask_yx])
        actin_roi_mask_yx = actin_yx > actin_thresh
        actin_roi_mask_tyx.append(actin_roi_mask_yx)
        actin_clean_mask_yx = actin_roi_mask_yx & (~ nuc_roi_mask_yx)
        actin_clean_mask_tyx.append(actin_clean_mask_yx)

        nuc_clean_mask_yx = nuc_roi_mask_yx & (~ actin_roi_mask_yx)
        nuc_clean_mask_tyx.append(nuc_clean_mask_yx)

        cyto_clean_mask_yx = cell_roi_mask_yx & (
            ~ actin_roi_mask_yx) & (~ nuc_roi_mask_yx)
        cyto_clean_mask_tyx.append(cyto_clean_mask_yx)

        actin_clean_mask_yx_fn = op.join(folder_computed,
                                         'actin-clean-mask-%03d.tiff' % t)
        nuc_clean_mask_yx_fn = op.join(folder_computed,
                                       'nuc-clean-mask-%03d.tiff' % t)
        cyto_clean_mask_yx_fn = op.join(folder_computed,
                                        'cyto-clean-mask-%03d.tiff' % t)
        save((actin_clean_mask_yx * (-1)).astype(np.uint16),
             actin_clean_mask_yx_fn)
        save((nuc_clean_mask_yx * (-1)).astype(np.uint16),
             nuc_clean_mask_yx_fn)
        save((cyto_clean_mask_yx * (-1)).astype(np.uint16),
             cyto_clean_mask_yx_fn)

    lim_tyx = list(lim_tyx)
    nuc_ratio_t = []
    actin_ratio_t = []
    print 'Computing ratios'
    for t, (lim_yx, actin_clean_mask_yx, nuc_clean_mask_yx, cyto_clean_mask_yx) \
        in enumerate(zip(lim_tyx, actin_clean_mask_tyx, nuc_clean_mask_tyx,
                         cyto_clean_mask_tyx)):
        nuc_area = np.sum(nuc_clean_mask_yx)
        actin_area = np.sum(actin_clean_mask_yx)
        cyto_area = np.sum(cyto_clean_mask_yx)

        lim_nuc_avg = np.sum(nuc_clean_mask_yx * lim_yx) / nuc_area
        lim_actin_avg = np.sum(actin_clean_mask_yx * lim_yx) / actin_area
        lim_cyto_avg = np.sum(cyto_clean_mask_yx * lim_yx) / cyto_area

        nuc_ratio = lim_nuc_avg / lim_cyto_avg
        actin_ratio = lim_actin_avg / lim_cyto_avg
        nuc_ratio_t.append(nuc_ratio)
        actin_ratio_t.append(actin_ratio)
        print 't =', t, 'nuc ratio =', nuc_ratio, 'actin ratio =', actin_ratio

    result = {'nuc_ratio_t': nuc_ratio_t, 'actin_ratio_t': actin_ratio_t}

    folder_pkl = 'output/live/ratio-pkl'
    sh.mkdir('-p', folder_pkl)
    output_pkl = op.join(folder_pkl,
                         op.split(op.split(op.abspath(fn_lim))[0])[1] + '.pkl')

    with open(output_pkl, 'w') as f:
        pickle.dump(result, f)

    folder_txt = 'output/live/ratio-txt'
    sh.mkdir('-p', folder_txt)
    output_txt = op.join(folder_txt,
                         op.split(op.split(op.abspath(fn_lim))[0])[1] + '.txt')

    with open(output_txt, 'w') as f:
        for nuc_ratio, actin_ratio in zip(nuc_ratio_t, actin_ratio_t):
            f.write('%.4f  %.4f\n' % (nuc_ratio, actin_ratio))

    return nuc_ratio_t, actin_ratio_t


if __name__ == '__main__':
    compute_lim_ratio('data/nuc-enrich-live/eb080218_001/fhl2.tif',
                      'data/nuc-enrich-live/eb080218_001/actin.tif',
                      'data/nuc-enrich-live/eb080218_001/nuc.tif')
