import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from Basics.sensorParams import D, PPMM
from compose.superposition import SuperPosition

VERTICES_START = 10  # vertices start at line 11 in ply file
Z_THRESHOLD = 0.2  # lower bound for masking objects


def press_object(object_path, dome_map, press_depth):
    """
    Calculates the contact mask and the raw deformations after pressing the object on the gelpad

    @param object_path: .ply file describing the object
    @param dome_map: (D, D) array representing the gelpad model, in pixels; the height is zero at the center and
    increases as the radius increases; there might be zeros at the four corners; for example:
        0 3 2 3 0
        3 2 1 2 3
        2 1 0 1 2
        3 2 1 2 3
        0 3 2 3 0
    @param press_depth: the extent of indentation measured in millimeters; note that the starting configuration is when
    the most protruding point of the object is z aligned with the gelpad's center
    @return:
        contact_mask: (D, D) array indicating points at which the object is in contact with the gelpad
        gel_map: (D, D) array representing the raw z deformations, in pixels
    """
    # parse the vertices (k, 3) of the object
    lines = open(object_path).readlines()
    vertices = np.array([
        list(map(float, line.strip().split(" ")))
        for line in lines[VERTICES_START:]
    ])

    # obtain the mask (k) for points in range after converting to pixels
    x_mean = np.mean(vertices[:, 0])
    y_mean = np.mean(vertices[:, 1])
    x_scaled = ((vertices[:, 0] - x_mean) * PPMM + D // 2).astype(int)
    y_scaled = ((vertices[:, 1] - y_mean) * PPMM + D // 2).astype(int)
    mask = (0 < x_scaled) & (x_scaled < D) & (0 < y_scaled) & (y_scaled < D) & (vertices[:, 2] > Z_THRESHOLD)

    # construct and update the height map (D, D) representing extent of indentation at each point
    height_map = np.zeros((D, D))
    height_map[y_scaled[mask], x_scaled[mask]] = vertices[mask, 2]
    height_map -= np.max(height_map)
    height_map += press_depth
    height_map = height_map * PPMM

    # obtain the contact mask (D, D) and the gel map (D, D)
    contact_mask = height_map > dome_map
    gel_map = np.zeros((D, D))
    gel_map[contact_mask] = height_map[contact_mask] - dome_map[contact_mask]

    return contact_mask, gel_map


def fill_zeros(image):
    """
    Use linear interpolation to fill zeros

    @param image: (H, W) array representing one channel of the image
    @return: filled: (H, W) array with zeros filled
    """
    points = np.nonzero(image)
    values = image[points].ravel()
    xi = np.meshgrid(
        np.arange(0, image.shape[0]),
        np.arange(0, image.shape[1])
    )

    # for some reason we need the transpose here
    filled = interpolate.griddata(points, values, tuple(xi), method="linear", fill_value=0).T

    return filled


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-obj", default='square', help="Object to be tested; supported: square, cylinder6")
    parser.add_argument('-dx', default=0.0, type=float, help='Shear load on the x axis')
    parser.add_argument('-dy', default=0.0, type=float, help='Shear load on the y axis')
    parser.add_argument('-dz', default=1.0, type=float, help='Shear load on the z axis')
    args = parser.parse_args()

    # obtain contact mask and gel map
    object_path = os.path.join("..", "data", "objects", "%s.ply" % args.obj)
    dome_map = np.load(os.path.join("..", "calibs", "dome_gel.npy"))
    raw_deform = np.array([args.dx, args.dy, args.dz])
    press_depth = raw_deform[2]
    contact_mask, gel_map = press_object(object_path, dome_map, press_depth)

    # obtain result map
    fem_path = os.path.join("..", "calibs", "femCalib.npz")
    sp = SuperPosition(fem_path)
    result_map = sp.propagate_deform(raw_deform, contact_mask, gel_map)

    # visualize
    plt.figure(0)

    plt.subplot(3, 1, 1)
    fig = plt.imshow(fill_zeros(result_map[:, :, 0]), cmap='seismic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 1, 2)
    fig = plt.imshow(fill_zeros(result_map[:, :, 1]), cmap='seismic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 1, 3)
    fig = plt.imshow(fill_zeros(result_map[:, :, 2]), cmap='seismic')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    # plt.show()
    output_path = os.path.join("..", "results", "%s_compose.jpg" % args.obj)
    plt.savefig(output_path)
