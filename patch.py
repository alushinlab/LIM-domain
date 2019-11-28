from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import sh
import warnings

from pims import ND2_Reader
from PIL import Image
import skimage.filters as sf
import skimage.io as si
import skimage.measure as sme
import skimage.morphology as smo
import skimage.util as su
import scipy.ndimage as sni
import cPickle as pickle


def find_roi(raw_ct, fn_actin_bgs, t_start=6, window_size=1):
    """
    Find regions of interest
    """
    nc, nt, ny, nx = raw_ct.shape

    # Threshold background-subtracted actin
    actin_bgs_t = si.imread(fn_actin_bgs)
    actin_mask_t = []

    # Generate soft actin masks to filter detected dots later on
    print 'Computing actin masks'
    for t, actin_bgs in enumerate(actin_bgs_t):
        if t <= 150:
            thresh = sf.threshold_mean(actin_bgs_t[150])
        else:
            thresh = sf.threshold_mean(actin_bgs)
        mask = smo.binary_opening(actin_bgs > thresh)
        clean = smo.remove_small_objects(mask, min_size=16, connectivity=1)
        actin_mask_t.append(clean)

    # Average over window_size and sobel
    sobel_t = np.zeros((nt, ny, nx))
    for t in xrange(nt):
        if t < t_start:
            continue

        # Average over a window centered at t.
        t_b = t - window_size // 2
        t_e = t_b + window_size
        t_b = max(t_start, t_b)
        t_e = min(nt, t_e)
        img = raw_ct[1][t_b:t_e].mean(axis=0)
        sobel_t[t] = sf.sobel(img)

    # Find bright spots on each frame
    bright_t = np.zeros((nt, ny, nx), dtype=bool)
    for t, sobel in enumerate(sobel_t):
        if t < t_start:
            continue

        thresh = sf.threshold_triangle(sobel)
        thresholded = sobel > thresh

        disk = smo.disk(1)
        clean = smo.binary_opening(thresholded, selem=disk)
        bright_t[t] = sni.binary_fill_holes(clean)

    # Keep only "persistent" bright spots
    persistent_t = np.zeros((nt, ny, nx), dtype=bool)
    adjacent_t = np.zeros((nt, ny, nx), dtype=bool)
    for t, bright in enumerate(bright_t):
        adjacent = bright_t[t - 1] if t > 0 else None
        if t + 1 < len(bright_t):
            adjacent = (adjacent | bright_t[t + 1]) \
                if adjacent is not None else bright_t[t + 1]
        adjacent_t[t] = adjacent

        labeled = sme.label(bright, connectivity=2)
        labels = [reg.label
                  for reg in sme.regionprops(labeled,
                                             intensity_image=adjacent)
                  if reg.max_intensity != 0]
        persistent_t[t] = np.isin(labeled, labels)

    return persistent_t, actin_mask_t


def compute_props(raw_ct, fn_actin_bgs, output_pkl, output_txt, t_start=6,
                  window_size=1, max_jump=20):
    final_t, actin_mask_t = find_roi(raw_ct, fn_actin_bgs, t_start=t_start,
                                     window_size=window_size)

    labeled_t = [sme.label(final, connectivity=2)
                 for t, final in enumerate(final_t)]

    trace_d, regions = identify_traces(labeled_t, actin_mask_t, max_jump)

    # Compute lifetime of each dot.
    lifetime_d = [tr[-1][0] - tr[0][0] + 1
                  for tr in trace_d]

    # Extract coordinates for each node
    coords_d = []
    for trace in trace_d:
        coords_n = [regions[node].coords for node in trace]
        coords_d.append(coords_n)

    # Extract centroid for each node
    centroid_d = []
    for trace in trace_d:
        centroid_n = [regions[node].centroid for node in trace]
        centroid_d.append(centroid_n)

    # Compute the local intensity ratio of each dot
    nc, nt, ny, nx = raw_ct.shape
    intensity_ratio_max_d = []
    for trace, lifetime in zip(trace_d, lifetime_d):
        intensity_ratio_n = []
        for node in trace:
            reg = regions[node]
            y_i, x_i = reg.coords.T  # split the two columns
            radius = max(1, int(2 * np.sqrt(reg.area)))

            inner = np.zeros((ny, nx), dtype=bool)
            inner[y_i, x_i] = 1
            outer = smo.binary_dilation(inner, smo.disk(radius)) & ~inner

            image = raw_ct[1][node[0]]
            inner_mean = image[inner].mean()
            outer_mean = image[outer].mean()
            intensity_ratio_per_node = inner_mean / outer_mean
            intensity_ratio_n.append(intensity_ratio_per_node)

        intensity_ratio_max_d.append(np.max(intensity_ratio_n))

        print '%-3d  %.4f' % (lifetime, intensity_ratio_max_d[-1])

    # Find lifetime, area, major axis length, and eccentricity of each dot
    area_max_d = []
    major_axis_length_max_d = []
    aspect_ratio_max_d = []
    eccentricity_max_d = []
    for trace in trace_d:
        if trace[-1][0] == nt - 1:  # ends on the last frame
            continue

        else:
            area_n = [regions[node].area for node in trace]
            major_axis_length_n = [regions[node].major_axis_length for node in
                                   trace]
            aspect_ratio_n = [(regions[node].major_axis_length /
                               regions[node].minor_axis_length) for node in
                              trace]
            eccentricity_n = [regions[node].eccentricity for node in trace]

            area_max_d.append(np.max(area_n))
            major_axis_length_max_d.append(np.max(major_axis_length_n))
            aspect_ratio_max_d.append(np.max(aspect_ratio_n))
            eccentricity_max_d.append(np.max(eccentricity_n))

    # Calculate actin intensity before and after unbinding
    actin_before_d = []
    actin_after_d = []
    for trace in trace_d:
        mask = np.zeros((ny, nx), dtype=bool)
        for node in trace:
            y_i, x_i = regions[node].coords.T
            mask[y_i, x_i] = 1

        t_b = trace[0][0]
        t_e = trace[-1][0] + 1
        actin_before = ((raw_ct[0, t_b:t_e] * mask)
                        .sum(axis=(-1, -2)).mean(axis=-1))
        actin_after  = ((raw_ct[0, t_e:t_e + 20] * mask)
                        .sum(axis=(-1, -2)).mean(axis=-1))
        actin_before_d.append(actin_before)
        actin_after_d.append(actin_after)

    output = dict(
        trace_d=trace_d,
        lifetime_d=lifetime_d,
        intensity_ratio_max_d=intensity_ratio_max_d,
        area_max_d=area_max_d,
        major_axis_length_max_d=major_axis_length_max_d,
        aspect_ratio_max_d=aspect_ratio_max_d,
        eccentricity_max_d=eccentricity_max_d,
        actin_before_d=actin_before_d,
        actin_after_d=actin_after_d,
        coords_d=coords_d,
        centroid_d=centroid_d,
    )

    sh.mkdir('-p', op.dirname(op.abspath(output_pkl)))
    with open(output_pkl, 'w') as f:
        pickle.dump(output, f)

    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for d, lifetime in enumerate(lifetime_d):
            f.write('%-3d  %.4f  %.4f  %.4f  %.4f  %.4f  %4f  %4f\n' % (
                lifetime, intensity_ratio_max_d[d], area_max_d[d],
                major_axis_length_max_d[d], aspect_ratio_max_d[d],
                eccentricity_max_d[d], actin_before_d[d], actin_after_d[d]))


def filter_props(pkl_filenames, lifetime_cutoff=5, intensity_ratio_cutoff=1.2):
    lifetime_d = []
    intensity_ratio_max_d = []
    area_max_d = []
    actin_before_d = []
    actin_after_d = []

    for filename in pkl_filenames:
        with open(filename) as f:
            pkl = pickle.load(f)

        lifetime_d += pkl['lifetime_d']
        intensity_ratio_max_d += pkl['intensity_ratio_max_d']
        area_max_d += pkl['area_max_d']
        actin_before_d += pkl['actin_before_d']
        actin_after_d += pkl['actin_after_d']

    lifetime_D = []
    intensity_ratio_max_D = []
    area_max_D = []
    actin_ratio_D = []
    for (lifetime, intensity_ratio_max, area_max,
         actin_before, actin_after) in zip(
             lifetime_d, intensity_ratio_max_d, area_max_d, actin_before_d,
             actin_after_d):
        if (lifetime > lifetime_cutoff) and (intensity_ratio_max >
                                             intensity_ratio_cutoff):
            lifetime_D.append(lifetime)
            intensity_ratio_max_D.append(intensity_ratio_max)
            area_max_D.append(area_max)
            actin_ratio = actin_before / actin_after
            actin_ratio_D.append(actin_ratio)

    # Convert number of frames to actual lifetime.
    # Imaging interval: 2 s
    lifetime_D = [lifetime * 2 for lifetime in lifetime_D]

    # Convert number of pixels to micron2. Pixel size: 0.267 um.
    area_max_D = [area_max * 0.071 for area_max in area_max_D]

    sh.mkdir('output/filtered', '-p')
    output_txt = 'output/filtered/filtered-props.txt'
    with open(output_txt, 'w') as f:
        for D, lifetime in enumerate(lifetime_D):
            f.write('%-3d  %.4f  %.4f  %.4f\n' % (
                lifetime, intensity_ratio_max_D[D], area_max_D[D],
                actin_ratio_D[D]))


def identify_traces(labeled_t, actin_mask_t, max_jump):
    """
    Identify dot traces from overlapping labels
    """
    # Every label l in frame t defines a node (t, l). "graph" is a dict such
    # that graph[(t, l)] is a list of nodes that are "connected" to (t, l).
    graph = {}
    regions = {}
    for t, labeled in enumerate(labeled_t):
        for reg in sme.regionprops(labeled):
            l = reg.label
            node = (t, l)

            regions[node] = reg
            graph.setdefault(node, []).append(node)

            y_i, x_i = reg.coords.T  # split the two columns
            for t_prev in xrange(max(0, t - max_jump), t):
                for l_prev in set(labeled_t[t_prev][y_i, x_i]) - {0}:
                    node_prev = (t_prev, l_prev)
                    graph.setdefault(node, []).append(node_prev)
                    graph.setdefault(node_prev, []).append(node)

    # Perform a depth-first traversal of the graph, recording each connected
    # component as a trace (i.e., a dot). Index traces by _d.
    all_trace_d = []
    remaining = set(graph.keys())
    while remaining:
        trace = []
        to_visit = [remaining.pop()]
        while to_visit:
            node = to_visit.pop()
            trace.append(node)
            for neighbor in graph[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    to_visit.append(neighbor)
        all_trace_d.append(sorted(trace))

    # Remove static dots, i.e., traces that persist to the last frame.
    dynamic_trace_d = [
        tr for tr in all_trace_d if tr[-1][0] != len(labeled_t) - 1]

    # Remove the traces that do not overlap with the actin masks
    trace_d = []
    for trace in dynamic_trace_d:
        for node in trace:
            overlap = (labeled_t[node[0]] == node[1]) & actin_mask_t[node[0]]
            if overlap.any() and node != trace[-1]:
                continue
            elif overlap.any() and node == trace[-1]:
                trace_d.append(trace)
            else:
                break

    return trace_d, regions


def load_ND2(filename):
    """Load ND2 file as numpy arrays"""
    print 'Loading %s' % filename
    with quiet_ND2_Reader(filename) as f_tc:
        f_tc.iter_axes = 't'
        f_tc.bundle_axes = 'cyx'

        assert ''.join(f_tc.iter_axes) == 't'
        assert ''.join(f_tc.bundle_axes) == 'cyx'
        nt, nc, nx, ny = [f_tc.sizes[a] for a in 'tcxy']

        raw_ct = np.zeros((nc, nt, ny, nx), f_tc[1].dtype)
        for t, f_c in enumerate(f_tc):
            for c, f in enumerate(f_c):
                raw_ct[c, t] = f

        return raw_ct


def quiet_ND2_Reader(filename, series=0, channel=0):
    """Silenced ND2_Reader()"""
    message = 'Please call FramesSequenceND.__init__()'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message)
        return ND2_Reader(filename, series=series, channel=channel)


if __name__ == '__main__':
    raw_ct = load_ND2('data/dots/20190126-0.nd2')
    compute_props(raw_ct, 'data/dots/actin-bgs/20190126-0.tif',
                  'output/dots/20190126-0.pkl',
                  'output/dots/20190126-0.txt')
    filter_props(['output/dots/20190126-0.pkl'])
