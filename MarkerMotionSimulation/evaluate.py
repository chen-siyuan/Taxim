import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from Basics.sensorParams import H, W, D

from simulation import Simulation
from ground_truth import GroundTruth


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

    if image2 is None:
        plt.subplot(3, 1, 1)
        plt.imshow(crop(fill(image[:, :, 0])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 1, 2)
        plt.imshow(crop(fill(image[:, :, 1])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 1, 3)
        plt.imshow(crop(fill(image[:, :, 2])), cmap="RdBu")
        plt.axis("off")
    else:
        plt.subplot(3, 2, 1)
        plt.imshow(crop(fill(image[:, :, 0])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 2, 3)
        plt.imshow(crop(fill(image[:, :, 1])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 2, 5)
        plt.imshow(crop(fill(image[:, :, 2])), cmap="RdBu")
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.imshow(crop(fill(image2[:, :, 0])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 2, 4)
        plt.imshow(crop(fill(image2[:, :, 1])), cmap="RdBu")
        plt.axis("off")
        plt.subplot(3, 2, 6)
        plt.imshow(crop(fill(image2[:, :, 2])), cmap="RdBu")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    GT_DATA = "0630_dome_square_0.5_0.3_0.6"
    SIM_OBJ = "square"
    DX = 0.5
    DY = 0.3
    DZ = 0.6

    print("PARAMETERS:")
    print("\tGround truth:\t%s" % GT_DATA)
    print("\tSimulation:  \t%s %.2f %.2f %.2f" % (SIM_OBJ, DX, DY, DZ))
    print("")

    print("Beginning ground truth digging...")
    gt_map = GroundTruth.dig(GT_DATA)

    print("Beginning simulation...")
    sim_map = Simulation(SIM_OBJ).run(DX, DY, DZ)

    print("Beginning visualization...")
    visualize(gt_map, sim_map)
