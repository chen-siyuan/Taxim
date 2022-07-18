from os import path

import numpy as np

from Basics.sensorParams import PPMM, D
from superposition import Superposition


class Simulation:
    def __init__(self, obj):
        lines = open(
            path.join("..", "data", "objects", "%s.ply" % obj)
        ).readlines()
        vertices = np.array([
            list(map(float, line.strip().split(" ")))
            for line in lines[10:]
        ])
        x_mean = np.mean(vertices[:, 0])
        y_mean = np.mean(vertices[:, 1])
        x_scaled = ((vertices[:, 0] - x_mean) * PPMM + D // 2).astype(int)
        # to account for discrepancy between the two xy axis systems
        y_scaled = ((-vertices[:, 1] + y_mean) * PPMM + D // 2).astype(int)
        mask = np.logical_and.reduce((
            0 <= x_scaled, x_scaled < D,
            0 <= y_scaled, y_scaled < D,
            vertices[:, 2] > 0.2
        ))
        self.height_map = np.zeros((D, D))
        self.height_map[x_scaled[mask], y_scaled[mask]] = vertices[mask, 2]
        self.height_map -= np.max(self.height_map)
        self.height_map *= PPMM
        self.dome_map = np.load(path.join("..", "calibs", "dome_gel.npy"))

    def run(self, dx, dy, dz):
        press_depth = dz * PPMM
        contact_mask = self.height_map + press_depth > self.dome_map
        gel_map = np.zeros((D, D))
        gel_map[contact_mask] = (self.height_map[contact_mask]
                                 + press_depth
                                 - self.dome_map[contact_mask])

        sp = Superposition()
        xy_deform = np.array([dx * PPMM, dy * PPMM])
        result_map, log = sp.propagate_deform(xy_deform, contact_mask, gel_map)

        return result_map, log
