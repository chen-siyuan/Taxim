from matplotlib import pyplot as plt
import numpy as np

from evaluate import crop, fill


experiments = [
        "0630_dome_cylinder6_0.0_0.0_0.5",
        "0630_dome_dome_0.3_0.4_0.8",
        "0630_dome_edge_0.0_0.5_0.5",
        "0630_dome_grid_0.0_0.0_0.3",
        "0630_dome_indent_0.0_0.0_0.5",
        "0630_dome_pyramid_-0.2_0.0_0.4",
        "0630_dome_side_cylinder2_0.3_0.3_0.6",
        "0630_dome_side_cylinder5_0.0_0.0_0.8",
        "0630_dome_side_cylinder5_0.0_0.5_0.8",
        "0630_dome_square_0.5_0.3_0.6",
        "0630_dome_star_0.0_0.0_0.5",
        "0630_dome_star_0.2_0.4_0.4",
        "0630_dome_stride_0.0_0.0_0.6",
        "0630_dome_triangle_0.0_0.0_0.5",
        ]
path_old = "../data/output/0816/output_old/"
path_new = "../data/output/0816/output_new/"


def main_diff():
    for experiment in experiments:
        npz_old = np.load(path_old + experiment + ".npz")
        npz_new = np.load(path_new + experiment + ".npz")

        gt = npz_old["gt_map"]
        assert((gt == npz_new["gt_map"]).all())
        old = npz_old["sim_map"]
        new = npz_new["sim_map"]

        xy_width = max(
                abs(np.min(gt[:, :, 0])), abs(np.max(gt[:, :, 0])),
                abs(np.min(gt[:, :, 1])), abs(np.max(gt[:, :, 1]))
                )
        xy_norm = plt.Normalize(-xy_width, xy_width)
        z_width = max(abs(np.min(gt[:, :, 2])), abs(np.max(gt[:, :, 2])))
        z_norm = plt.Normalize(-z_width, z_width)
        norms = [xy_norm, xy_norm, z_norm]

        plt.figure()

        for i in range(3):
            plt.subplot(3, 5, i * 5 + 1)
            plt.imshow(crop(fill(gt[:, :, i].T)), cmap="RdBu", norm=norms[i])
            plt.axis("off")
            plt.subplot(3, 5, i * 5 + 2)
            plt.imshow(crop(fill(old[:, :, i].T)), cmap="RdBu", norm=norms[i])
            plt.axis("off")
            plt.subplot(3, 5, i * 5 + 3)
            plt.imshow(crop(fill(new[:, :, i].T)), cmap="RdBu", norm=norms[i])
            plt.axis("off")
            plt.subplot(3, 5, i * 5 + 4)
            plt.imshow((
                crop(fill(new[:, :, i].T))
                - crop(fill(old[:, :, i].T))
                ) * 10, cmap="RdBu", norm=norms[i])
            plt.axis("off")
            plt.subplot(3, 5, i * 5 + 5)
            plt.imshow((
                crop(fill(new[:, :, i].T))
                - crop(fill(old[:, :, i].T))
                ) * 100, cmap="RdBu", norm=norms[i])
            plt.axis("off")

        log_old = npz_old["log"]
        log_new = npz_new["log"]

        text = "%s - (%.2f %.2f %.2f %.2f) - (%.2f %.2f %.2f %.2f)" % (
                experiment[10:],
                log_old[0], log_old[1] - log_old[0],
                log_old[2] - log_old[1], log_old[3] - log_old[2],
                log_new[0], log_new[1] - log_new[0],
                log_new[2] - log_new[1], log_new[3] - log_new[2]
                )
        header = ("         Ground      Old         New          New - Old   New - Old\n"
                + "         truth       simulation  simulation         10x        100x")

        plt.figtext(0.01, 0.01, text, fontfamily="monospace")
        plt.figtext(0.01, 0.85, header, fontfamily="monospace")

        plt.savefig("../data/%s.png" % experiment)


def main_time():
    pass


if __name__ == "__main__":
    main_time()
