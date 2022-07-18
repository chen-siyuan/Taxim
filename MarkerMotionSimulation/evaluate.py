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


def visualize(image, image2=None, text=None, path=None):
    plt.figure()

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

    if text is not None:
        plt.figtext(0.01, 0.01, text,
                    fontfamily="monospace")

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def parse_experiment(experiment):
    data = experiment.split("_")
    gt_data = experiment
    sim_obj = "_".join(data[2:-3])
    dx, dy, dz = map(float, data[-3:])
    return gt_data, sim_obj, dx, dy, dz


def perform_experiment(experiment):
    print("Running experiment: %s" % experiment)
    gt_data, sim_obj, dx, dy, dz = parse_experiment(experiment)

    print("Beginning ground truth digging...")
    gt_map = GroundTruth.dig(gt_data)

    print("Beginning simulation...")
    sim_map, log = Simulation(sim_obj).run(dx, dy, dz)
    text = "%s - (%.2f %.2f %.2f %.2f)" % (experiment,
                                           log[0],
                                           log[1] - log[0],
                                           log[2] - log[1],
                                           log[3] - log[2])
    path = os.path.join("..", "data", "plots", "%s.png" % experiment)

    print("Beginning visualization...")
    visualize(gt_map, sim_map, text=text, path=path)

    print("")


def main_perform_experiments():
    experiments = [
        # "0630_dome_9_pattern_0.0_0.5_0.5", (seems to be mislabeled)
        # "0630_dome_9_pattern_0.2_0.2_0.6", (not pronounced enough)
        # "0630_dome_9_pattern_0.2_0.2_1.0", (not pronounced enough)
        "0630_dome_cylinder6_0.0_0.0_0.5",
        "0630_dome_dome_0.3_0.4_0.8",
        "0630_dome_edge_0.0_-0.5_0.5",
        "0630_dome_grid_0.0_0.0_0.3",
        "0630_dome_indent_0.0_0.0_0.5",
        "0630_dome_pyramid_-0.2_0.0_0.4",
        "0630_dome_side_cylinder2_0.3_0.3_0.6",
        # "0630_dome_side_cylinder3_0.3_0.0_0.7", (seems to be mislabeled)
        "0630_dome_side_cylinder5_0.0_0.0_0.8",
        "0630_dome_side_cylinder5_0.0_0.5_0.8",
        "0630_dome_square_0.5_0.3_0.6",
        "0630_dome_star_0.0_0.0_0.5",
        "0630_dome_star_0.2_0.4_0.4",
        "0630_dome_stride_0.0_0.0_0.6",
        "0630_dome_triangle_0.0_0.0_0.5"
    ]
    for experiment in experiments:
        perform_experiment(experiment)


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
    main_perform_experiments()
