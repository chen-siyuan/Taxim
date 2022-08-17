from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from evaluate import crop, fill

import pandas as pd


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
    df_old = {"old sim": [], "precompute": [], "set up NNLS": [], "run NNLS": [], "propagate": []}
    df_new = {"new sim": [], "precompute": [], "set up NNLS": [], "run NNLS": [], "propagate": []}

    for experiment in experiments:
        npz_old = np.load(path_old + experiment + ".npz")
        npz_new = np.load(path_new + experiment + ".npz")

        old = npz_old["log"]
        new = npz_new["log"]

        df_old["old sim"].append(experiment[10:-12])
        df_old["precompute"].append(old[0])
        df_old["set up NNLS"].append(old[1] - old[0])
        df_old["run NNLS"].append(old[2] - old[1])
        df_old["propagate"].append(old[3] - old[2])

        df_new["new sim"].append(experiment[10:-12])
        df_new["precompute"].append(new[0])
        df_new["set up NNLS"].append(new[1] - new[0])
        df_new["run NNLS"].append(new[2] - new[1])
        df_new["propagate"].append(new[3] - new[2])

    df_old = pd.DataFrame.from_dict(df_old)
    df_new = pd.DataFrame.from_dict(df_new)

    _, axes = plt.subplots(2, 1)

    df_old.set_index("old sim").plot(kind="barh", stacked=True, ax=axes[0], xlim=(0, 40), colormap="crest")
    df_new.set_index("new sim").plot(kind="barh", stacked=True, ax=axes[1], xlim=(0, 40), colormap="crest")

    plt.show()


def main_prop():
    df_old = {"precompute": 0., "set up NNLS": 0., "run NNLS": 0., "propagate": 0.}
    df_new = {"precompute": 0., "set up NNLS": 0., "run NNLS": 0., "propagate": 0.}

    for experiment in experiments:
        npz_old = np.load(path_old + experiment + ".npz")
        npz_new = np.load(path_new + experiment + ".npz")

        old = npz_old["log"]
        new = npz_new["log"]

        df_old["precompute"] += old[0]
        df_old["set up NNLS"] += old[1] - old[0]
        df_old["run NNLS"] += old[2] - old[1]
        df_old["propagate"] += old[3] - old[2]

        df_new["precompute"] += new[0]
        df_new["set up NNLS"] += new[1] - new[0]
        df_new["run NNLS"] += new[2] - new[1]
        df_new["propagate"] += new[3] - new[2]

    df = pd.DataFrame.from_dict([df_old]).append(pd.DataFrame.from_dict([df_new]))
    df["type"] = ["old sim", "new sim"]
    df = df.set_index("type")

    print(df)

    df.loc["old sim"] /= sum(df.loc["old sim"])
    df.loc["new sim"] /= sum(df.loc["new sim"])

    print(df)

    df.plot(kind="bar", stacked=True, colormap="crest")

    plt.figtext(0.23, 0.12, "69.83s\n44.37%", fontfamily="monospace", color="white")
    plt.figtext(0.23, 0.45, "49.40s\n31.39%", fontfamily="monospace", color="white")
    plt.figtext(0.23, 0.68, "37.93s\n24.10%", fontfamily="monospace", color="white")

    plt.figtext(0.62, 0.12, "34.16s\n28.63%", fontfamily="monospace", color="white")
    plt.figtext(0.62, 0.34, "47.06s\n39.43%", fontfamily="monospace", color="white")
    plt.figtext(0.62, 0.63, "37.91s\n31.77%", fontfamily="monospace", color="white")

    plt.show()


"""
69.832879  49.403128  37.933687
34.163148  47.060397  37.910251
0.443737  0.313921   0.241041
0.001720     0.286270  0.394342   0.317668
"""


if __name__ == "__main__":
    #  main_time()
    main_prop()
