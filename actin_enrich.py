from __future__ import division

import cPickle as pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import sh
import warnings

from pims import ND2_Reader
import skimage.filters as sf
import skimage.io as si
import skimage.measure as sme
import skimage.morphology as smo
import skimage.util as su
import scipy.ndimage as sni


class NoFocusError(Exception):
    pass


def find_focus(raw_zyxc, cell_area_thresh, erosion_factor=None):
    lim_zyx = raw_zyxc[:, :, :, 0]
    actin_zyx = raw_zyxc[:, :, :, 1]
    nuc_zyx = raw_zyxc[:, :, :, 2]

    lim_mip_yx = np.max(lim_zyx, axis=0)

    lim_mip_sm_yx = sf.gaussian(lim_mip_yx)
    cell_mask_yx = smo.binary_opening(sni.binary_fill_holes(
        (lim_mip_sm_yx > sf.threshold_li(lim_mip_sm_yx))))

    cell_labeled_yx = sme.label(cell_mask_yx, connectivity=2)
    region_r = [region
                for region in sme.regionprops(cell_labeled_yx)
                if region.area >= cell_area_thresh]
    if not region_r:
        raise NoFocusError('region_r is empty')

    region = max(region_r, key=lambda region: region.area)
    cell_clean_mask_yx = (cell_labeled_yx == region.label)

    # Erode the clean cell mask to a fraction of its original size
    if erosion_factor:
        area_threshold = erosion_factor * np.sum(cell_clean_mask_yx)
        eroded_yx = cell_clean_mask_yx
        area_prev = eroded_yx.sum()
        print 'Eroding cell mask'
        while True:
            eroded_yx = smo.binary_erosion(eroded_yx)
            area_eroded = eroded_yx.sum()
            if area_eroded <= area_threshold:
                break
        cell_clean_mask_yx = eroded_yx

    actin_score_z = []
    # Normalize the sobel image by intensity
    for actin_yx in actin_zyx:
        actin_sb_yx = sf.sobel(actin_yx, mask=cell_clean_mask_yx)
        actin_score = (np.sum(actin_sb_yx[cell_clean_mask_yx]) /
                       np.sum(actin_yx[cell_clean_mask_yx]))
        actin_score_z.append(actin_score)

    z_focus = np.argmax(actin_score_z)
    nuc_focus_yx = nuc_zyx[z_focus]
    lim_focus_yx = lim_zyx[z_focus]
    actin_focus_yx = actin_zyx[z_focus]

    return (cell_clean_mask_yx, lim_focus_yx, actin_focus_yx, nuc_focus_yx,
            z_focus)


def dilate_nuc_mask(nuc_mask_yx, lim_mask_yx, dilat_factor):
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


def compute_ratio(cell_clean_mask_yx, lim_yx, actin_yx, nuc_yx, dilat_factor):
    # Use mean for actin thresholding
    actin_thresh = sf.threshold_mean(actin_yx[cell_clean_mask_yx])
    actin_mask_yx = (actin_yx > actin_thresh) * cell_clean_mask_yx

    # Use li for nuclear thresholding
    nuc_thresh = sf.threshold_li(sf.gaussian(nuc_yx)[cell_clean_mask_yx])
    nuc_mask_yx = (smo.binary_opening(sni.binary_fill_holes((
        sf.gaussian(nuc_yx) > nuc_thresh)))) * cell_clean_mask_yx

    nuc_labeled_yx = sme.label(nuc_mask_yx, connectivity=2)
    region_r = [region for region in sme.regionprops(nuc_labeled_yx)]

    if not region_r:
        raise NoFocusError('region_r is empty')

    region = max(region_r, key=lambda region: region.area)
    nuc_mask_yx = (nuc_labeled_yx == region.label)

    # Exclude nuclear region in actin mask, and actin region in nuclear mask
    actin_clean_mask_yx = actin_mask_yx & (~ nuc_mask_yx)
    nuc_clean_mask_yx = nuc_mask_yx & (~ actin_mask_yx)

    cyto_mask_yx = cell_clean_mask_yx & (~ nuc_mask_yx) & (~ actin_mask_yx)

    actin_area = np.sum(actin_clean_mask_yx)
    nuc_area = np.sum(nuc_clean_mask_yx)
    cyto_area = np.sum(cyto_mask_yx)
    cell_area = np.sum(cell_clean_mask_yx)

    lim_actin_avg = np.sum(lim_yx * actin_clean_mask_yx) / actin_area
    lim_nuc_avg = np.sum(lim_yx * nuc_clean_mask_yx) / nuc_area
    lim_cyto_avg = np.sum(lim_yx * cyto_mask_yx) / cyto_area
    lim_cell_avg = np.sum(lim_yx * cell_clean_mask_yx) / cell_area

    actin_ratio = lim_actin_avg / lim_cyto_avg
    nuc_ratio = lim_nuc_avg / lim_cyto_avg

    dilated_yx = dilate_nuc_mask(nuc_mask_yx, cell_clean_mask_yx, dilat_factor)
    cyto_local_mask_yx = dilated_yx & (~ nuc_mask_yx) & (actin_mask_yx)
    cyto_local_area = np.sum(cyto_local_mask_yx)
    lim_cyto_local_avg = np.sum(lim_yx * cyto_local_mask_yx) / cyto_local_area
    nuc_local_ratio = lim_nuc_avg / lim_cyto_local_avg

    return (actin_ratio, nuc_ratio, nuc_local_ratio, actin_mask_yx,
            nuc_mask_yx, lim_cell_avg)


def compute_ratio_for_all(cell_area_thresh=1000, erosion_factor=None,
                          dilat_factor=2):

    folder_txt = 'output/screen-txt'
    sh.mkdir('-p', folder_txt)

    folder_mask = 'output/mask'
    sh.mkdir('-p', folder_mask)

    for path in sorted(glob('data/done-tiff/*/*/*')):
        subfolder_mask = op.join(folder_mask, op.relpath(path, 'data/done-tiff'))
        sh.mkdir('-p', subfolder_mask)

        subfolder_txt = op.join(folder_txt, op.relpath(
            path, 'data/done-tiff').split('/')[0])
        sh.mkdir('-p', subfolder_txt)

        basename = '%s-%s' % (path.split('/')[-2], path.split('/')[-1])
        finish_fn = op.join(subfolder_txt, '.finish.' + basename)
        if op.isfile(finish_fn):
            return

        prefix_n = []
        actin_ratio_n = []
        nuc_ratio_n = []
        nuc_local_ratio_n = []
        lim_cell_avg_n = []
        for filename in sorted(glob(op.join(path, '*.tif*'))):
            print 'Loading', filename

            prefix = op.splitext(op.basename(filename))[0]
            img = si.imread(filename)
            if len(img.shape) == 3:
                img_yxc = img
                lim_yx = img_yxc[:, :, 0]
                actin_yx = img_yxc[:, :, 1]
                nuc_yx = img_yxc[:, :, 2]

                lim_sm_yx = sf.gaussian(lim_yx)
                cell_mask_yx = smo.binary_opening(sni.binary_fill_holes(
                    (lim_sm_yx > sf.threshold_li(lim_sm_yx))))

                cell_labeled_yx = sme.label(cell_mask_yx, connectivity=2)
                region_r = [region
                            for region in sme.regionprops(cell_labeled_yx)
                            if region.area >= cell_area_thresh]
                region = max(region_r, key=lambda region: region.area)
                cell_clean_mask_yx = (cell_labeled_yx == region.label)

                # Erode the clean cell mask to a fraction of its original size
                if erosion_factor:
                    area_threshold = erosion_factor * np.sum(cell_clean_mask_yx)
                    eroded_yx = cell_clean_mask_yx
                    area_prev = eroded_yx.sum()
                    print 'Eroding cell mask'
                    while True:
                        eroded_yx = smo.binary_erosion(eroded_yx)
                        area_eroded = eroded_yx.sum()
                        if area_eroded <= area_threshold:
                            break
                    cell_clean_mask_yx = eroded_yx

                if np.sum(lim_yx[cell_clean_mask_yx] == 4095) >= 500:
                    continue
                else:
                    output = compute_ratio(
                        cell_clean_mask_yx, lim_yx, actin_yx, nuc_yx,
                        dilat_factor)
                    (actin_ratio, nuc_ratio, nuc_local_ratio, actin_mask_yx,
                     nuc_mask_yx, lim_cell_avg) = output

                    cell_mask_fn = op.join(subfolder_mask, prefix + '-cell.tiff')
                    actin_mask_fn = op.join(subfolder_mask, prefix + '-actin.tiff')
                    nuc_mask_fn = op.join(subfolder_mask, prefix + '-nuc.tiff')

            elif len(img.shape) == 4:
                img_zyxc = img
                (cell_clean_mask_yx, lim_yx, actin_yx,
                 nuc_yx, z_focus) = find_focus(
                     img_zyxc, cell_area_thresh=cell_area_thresh,
                     erosion_factor=erosion_factor)

                if np.sum(lim_yx[cell_clean_mask_yx] == 4095) >= 500:
                    continue
                else:
                    output = compute_ratio(
                        cell_clean_mask_yx, lim_yx, actin_yx, nuc_yx,
                        dilat_factor)
                    (actin_ratio, nuc_ratio, nuc_local_ratio, actin_mask_yx,
                     nuc_mask_yx, lim_cell_avg) = output

                    cell_mask_fn = op.join(subfolder_mask, prefix +
                                           '-z%d-cell.tiff' % z_focus)
                    actin_mask_fn = op.join(subfolder_mask, prefix +
                                            '-z%d-actin.tiff' % z_focus)
                    nuc_mask_fn = op.join(subfolder_mask, prefix +
                                          '-z%d-nuc.tiff' % z_focus)

            else:
                raise Exception('Unexpected image shape')

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*is a low contrast image')
                si.imsave(cell_mask_fn, su.img_as_uint(cell_clean_mask_yx))
                si.imsave(actin_mask_fn, su.img_as_uint(actin_mask_yx))
                si.imsave(nuc_mask_fn, su.img_as_uint(nuc_mask_yx))

            print prefix,
            print 'actin ratio =', actin_ratio, 'nuc ratio =', nuc_ratio

            prefix_n.append(prefix)
            actin_ratio_n.append(actin_ratio)
            nuc_ratio_n.append(nuc_ratio)
            nuc_local_ratio_n.append(nuc_local_ratio)
            lim_cell_avg_n.append(lim_cell_avg)

        ratios = {
            'prefix_n': prefix_n,
            'actin_ratio_n': actin_ratio_n,
            'nuc_ratio_n': nuc_ratio_n,
            'nuc_local_ratio_n': nuc_local_ratio_n,
            'lim_cell_avg_n': lim_cell_avg_n,
        }

        ratios_txt_fn = op.join(subfolder_txt, '%s.txt' % basename)
        with open(ratios_txt_fn, 'w') as f:
            for n, prefix in enumerate(prefix_n):
                f.write('%-9s  %.4f  %.4f  %.4f  %.4f\n' % (
                    prefix, actin_ratio_n[n], nuc_ratio_n[n],
                    nuc_local_ratio_n[n], lim_cell_avg_n[n]))

        sh.touch(finish_fn)


if __name__ == '__main__':
    compute_ratio_for_all()
