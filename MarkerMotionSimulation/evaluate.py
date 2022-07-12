import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from Basics.sensorParams import H, W, D
from ground_truth import GroundTruth
from simulation import Simulation


def fill(image):
    points = np.nonzero(image)
    values = image[points].ravel()
    xi = np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
    filled = interpolate.griddata(
        points, values, tuple(xi), method="linear", fill_value=0
    ).T

    return filled


def crop(image):
    h = H // 2
    w = W // 2
    d = D // 2
    cropped = image[d - h:d + h, d - w:d + w]

    return cropped


def visualize(image, image2=None):
    plt.figure(0)

    xy_width = max(
        abs(np.min(image[:, :, 0])), abs(np.max(image[:, :, 0])),
        abs(np.min(image[:, :, 1])), abs(np.max(image[:, :, 1]))
    )
    xy_norm = plt.Normalize(-xy_width, xy_width)
    z_width = max(abs(np.min(image[:, :, 2])), abs(np.max(image[:, :, 2])))
    z_norm = plt.Normalize(-z_width, z_width)

    if image2 is None:
        plt.subplot(3, 1, 1)
        plt.imshow(crop(fill(image[:, :, 0].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 1, 2)
        plt.imshow(crop(fill(image[:, :, 1].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 1, 3)
        plt.imshow(crop(fill(image[:, :, 2].T)), cmap="RdBu", norm=z_norm)
        plt.axis("off")
    else:
        plt.subplot(3, 2, 1)
        plt.imshow(crop(fill(image[:, :, 0].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 2, 3)
        plt.imshow(crop(fill(image[:, :, 1].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 2, 5)
        plt.imshow(crop(fill(image[:, :, 2].T)), cmap="RdBu", norm=z_norm)
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.imshow(crop(fill(image2[:, :, 0].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 2, 4)
        plt.imshow(crop(fill(image2[:, :, 1].T)), cmap="RdBu", norm=xy_norm)
        plt.axis("off")
        plt.subplot(3, 2, 6)
        plt.imshow(crop(fill(image2[:, :, 2].T)), cmap="RdBu", norm=z_norm)
        plt.axis("off")

    plt.show()


def main_compare_sim_gt():
    gt_data = "0630_dome_triangle_0.0_0.0_0.5"
    sim_obj = "triangle"
    dx = 0.0
    dy = 0.0
    dz = 0.5

    print("PARAMETERS:")
    print("\tGround truth:\t%s" % gt_data)
    print("\tSimulation:  \t%s %.2f %.2f %.2f" % (sim_obj, dx, dy, dz))
    print("")

    print("Beginning ground truth digging...")
    gt_map = GroundTruth.dig(gt_data)

    print("Beginning simulation...")
    sim_map = Simulation(sim_obj).run(dx, dy, dz)

    print("Beginning visualization...")
    visualize(gt_map, sim_map)


def main_extract_gt():
    gt_path = os.path.join("..", "data", "GT")
    for gt_data in os.listdir(os.path.join("..", "data", "FEM")):
        print(gt_data)
        if input("Extract? (y/N): ") not in {"Y", 'y'}:
            continue
        print("Extraction started ...")
        gt_map = GroundTruth.dig(gt_data)
        np.save(gt_path + "/" + gt_data, gt_map)
        print("Extraction successful.")


if __name__ == "__main__":
    main_compare_sim_gt()
