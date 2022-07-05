import numpy as np
from matplotlib import pyplot as plt

from Basics.sensorParams import PPMM, H, W, D
from scipy import interpolate

def fill_zeros(image):
    """
    Use linear interpolation to fill zeros

    @param image: (H, W) array representing one channel of the image
    @return: filled: (H, W) array with zeros filled
    """
    points = np.nonzero(image)
    values = image[points].ravel()
    xi = np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))

    # for some reason we need the transpose here
    filled = interpolate.griddata(
        points, values, tuple(xi), method="linear", fill_value=0
    ).T

    return filled

class DataLoader:
    def __init__(self, x_path, y_path, z_path):
        self.xs = []
        self.ys = []
        self.zs = []
        self.dxs = []
        for line in open(x_path).readlines()[1:]:
            line_split = line.split()
            self.xs.append(float(line_split[1]) * 1000.0)
            self.ys.append(float(line_split[2]) * 1000.0)
            self.zs.append(float(line_split[3]) * 1000.0)
            self.dxs.append(float(line_split[4]) * 1000.0)
        self.xs = np.array(self.xs, dtype=float)
        self.ys = np.array(self.ys, dtype=float)
        self.zs = np.array(self.zs, dtype=float)
        self.dxs = np.array(self.dxs, dtype=float)
        self.dys = np.array(list(
            float(line.split()[4]) * 1000.0
            for line in open(y_path).readlines()[1:]), dtype=float)
        self.dzs = np.array(list(
            float(line.split()[4]) * 1000.0
            for line in open(z_path).readlines()[1:]
        ), dtype=float)

        self.xs -= np.mean(self.xs)
        self.ys -= np.mean(self.ys)
        self.zs -= np.min(self.zs)

    def generate_gt(self):
        x_scaled = (self.xs * PPMM + D // 2).astype(int)
        y_scaled = (self.ys * PPMM + D // 2).astype(int)
        mask = np.logical_and.reduce((
            0 <= x_scaled, x_scaled < D,
            0 <= y_scaled, y_scaled < D
        ))

        z_map = np.zeros((D, D))
        z_map[x_scaled[mask], y_scaled[mask]] = self.zs[mask]

        dx_map = np.zeros((D, D))
        dx_map[x_scaled[mask], y_scaled[mask]] = self.dxs[mask]

        dy_map = np.zeros((D, D))
        dy_map[x_scaled[mask], y_scaled[mask]] = self.dys[mask]

        dz_map = np.zeros((D, D))
        dz_map[x_scaled[mask], y_scaled[mask]] = self.dzs[mask]

        return z_map, dx_map, dy_map, dz_map


if __name__ == "__main__":
    obj = "0630_dome_square_0.5_0.3_0.6"
    path = "../../data/FEM/" + obj + "/" + obj + "_"
    x_path = path + "x.txt"
    y_path = path + "y.txt"
    z_path = path + "z.txt"
    dl = DataLoader(x_path, y_path, z_path)
    z_map, dx_map, dy_map, dz_map = dl.generate_gt()

    h = H // 2
    w = W // 2
    d = D // 2
    crop = lambda M: M[d - h:d + h, d - w:d + w]

    # visualize
    plt.figure(0)

    plt.subplot(4, 1, 1)
    fig = plt.imshow(fill_zeros(crop(z_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 2)
    fig = plt.imshow(fill_zeros(crop(dx_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    fig = plt.imshow(fill_zeros(crop(dy_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    fig = plt.imshow(fill_zeros(crop(dz_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()
