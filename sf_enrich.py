from __future__ import division
import cPickle as pickle
from glob import glob
import numpy as np
import os.path as op
import sh
import skimage.filters as sf
import skimage.io as si
import skimage.morphology as smo
import skimage.util as su


def compute_ratio(fn_mip, fn_roi, depth=16):
    mip = si.imread(fn_mip)
    assert len(mip.shape) == 3 and mip.shape[0] == 2
    lim = mip[1]

    # Calculate the number of saturated pixels
    if depth == 16:
        sat_num = np.sum(lim == 65535)
    elif depth == 12:
        sat_num = np.sum(lim == 4095)
    actin = mip[0]
    roi = si.imread(fn_roi)

    roi_mask = (roi == 255)
    actin_thresh = sf.threshold_yen(actin[roi_mask])
    actin_mask = (actin > actin_thresh) & roi_mask
    cyto_mask = roi_mask & (~ actin_mask)

    actin_area = np.sum(actin_mask)
    cyto_area = np.sum(cyto_mask)
    lim_actin_avg = np.sum(lim * actin_mask) / actin_area
    lim_cyto_avg = np.sum(lim * cyto_mask) / cyto_area
    ratio = lim_actin_avg / lim_cyto_avg

    return ratio, actin_mask, cyto_mask, sat_num


def compute_ratio_for_all():
    for path in sorted(glob('data/sf-enrich/*/*')):
        protein = path.split('/')[-3]
        date = path.split('/')[-2]
        condition = path.split('/')[-1]
        output_mask = op.join('output/sf-enrich/mask', protein, date, condition)
        sh.mkdir('-p', output_mask)
        output_ratio = 'output/sf-enrich/ratio'
        sh.mkdir('-p', output_ratio)

        suffix_n = []
        ratio_n = []
        for fn_mip in sorted(glob(op.join(path, 'mip-bgs/*.tif'))):
            suffix = op.splitext(op.basename(fn_mip))[0]
            suffix_n.append(suffix)
            fn_roi = op.join(path, 'roi/%s.png' % suffix)
            assert op.isfile(fn_roi)

            ratio, actin_mask, cyto_mask, sat_num = compute_ratio(fn_mip, fn_roi)

            # Exclude images with more than 5 saturated pixels
            if sat_num > 5:
                continue
            else:
                print fn_mip, ratio
                ratio_n.append(ratio)

                fn_actin_mask = op.join(output_mask, 'actin-mask-%s.tif' % suffix)
                si.imsave(fn_actin_mask, su.img_as_uint(actin_mask))
                fn_cyto_mask = op.join(output_mask, 'cyto-mask-%s.tif' % suffix)
                si.imsave(fn_cyto_mask, su.img_as_uint(cyto_mask))

        fn_ratio = op.join(output_ratio, '%s-%s-%s.txt' % (protein, date, condition))
        with open(fn_ratio, 'w') as f:
            for ratio in ratio_n:
                f.write('%.4f\n' % ratio)


if __name__ == '__main__':
    compute_ratio_for_all()
