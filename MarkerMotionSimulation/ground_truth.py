from os import path

import numpy as np

from Basics.sensorParams import PPMM, D


class GroundTruth:
    @staticmethod
    def dig(data):
        x_path = path.join("..", "data", "FEM", data, "%s_x.txt" % data)
        y_path = path.join("..", "data", "FEM", data, "%s_y.txt" % data)
        z_path = path.join("..", "data", "FEM", data, "%s_z.txt" % data)

        xs = []
        ys = []
        zs = []
        dxs = []
        for line in open(x_path).readlines()[1:]:
            line_split = line.split()
            xs.append(float(line_split[1]) * 1000.0)
            ys.append(float(line_split[2]) * 1000.0)
            zs.append(float(line_split[3]) * 1000.0)
            dxs.append(float(line_split[4]) * 1000.0)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)
        dxs = np.array(dxs, dtype=float)
        dys = np.array(list(
            float(line.split()[4]) * 1000.0
            for line in open(y_path).readlines()[1:]), dtype=float)
        dzs = np.array(list(
            float(line.split()[4]) * 1000.0
            for line in open(z_path).readlines()[1:]
        ), dtype=float)

        xs -= np.mean(xs)
        ys -= np.mean(ys)
        zs -= np.min(zs)
        x_scaled = (xs * PPMM + D // 2).astype(int)
        y_scaled = (ys * PPMM + D // 2).astype(int)
        mask = np.logical_and.reduce((
            0 <= x_scaled, x_scaled < D,
            0 <= y_scaled, y_scaled < D
        ))

        result_map = np.zeros((D, D, 3))
        result_map[x_scaled[mask], y_scaled[mask], 0] = dxs[mask]
        result_map[x_scaled[mask], y_scaled[mask], 1] = dys[mask]
        result_map[x_scaled[mask], y_scaled[mask], 2] = dzs[mask]

        return result_map
