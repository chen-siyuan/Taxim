import time
from os import path

import numpy as np
from scipy.optimize import nnls

from Basics.params import shear_friction, normal_friction
from Basics.sensorParams import D


class Superposition:
    def __init__(self):
        fem_data = np.load(path.join("..", "calibs", "femCalib.npz"),
                           allow_pickle=True)
        self.tensor_map = fem_data["tensorMap"].transpose(1, 0, 2, 3)
        self.sparse_mask = fem_data["nodeMask"]

    def propagate_deform(self, xy_deform, contact_mask, gel_map):
        time_start = time.time()
        log = []

        all_xs, all_ys = np.nonzero(self.sparse_mask)
        act_xs, act_ys = np.nonzero(np.logical_and(
            self.sparse_mask, contact_mask
        ))

        deform_map = np.zeros((D, D, 3))
        deform_map[:, :, 0:2] = xy_deform
        deform_map[:, :, 2] = gel_map

        # lower = D // 4
        # upper = D // 4 * 3
        lower = 0
        upper = D

        log.append(time.time() - time_start)

        matrices = np.zeros((act_xs.size, act_ys.size, 3))
        for i, (x1, y1) in enumerate(zip(act_xs, act_ys)):
            for j, (x2, y2) in enumerate(zip(act_xs, act_ys)):
                # if j > i:
                #     break
                dx = x2 - x1 + D // 2
                dy = y2 - y1 + D // 2
                tensor = np.zeros((3, 3))
                if lower <= dx < upper and lower <= dy < upper:
                    tensor = self.tensor_map[dx, dy]
                for k in range(3):
                    matrices[i, j, k] = tensor[k, k]
                    matrices[j, i, k] = tensor[k, k]

        log.append(time.time() - time_start)

        for i in range(3):
            deform_map[act_xs, act_ys, i] = nnls(
                matrices[:, :, i], deform_map[act_xs, act_ys, i]
            )[0]

        log.append(time.time() - time_start)

        result_map = np.zeros((D, D, 3))
        for x, y in zip(all_xs, all_ys):
            rel_xs = act_xs - x + D // 2
            rel_ys = act_ys - y + D // 2
            mask = (0 <= rel_xs) & (rel_xs < D) & (0 <= rel_ys) & (rel_ys < D)

            tensors = self.tensor_map[rel_xs[mask], rel_ys[mask], :, :]
            tensors[:, 0:2, 0:2] *= shear_friction
            tensors[:, 0:2, 2] *= normal_friction
            deforms = deform_map[act_xs[mask], act_ys[mask], :, np.newaxis]

            indiv_displacements = np.matmul(tensors, deforms)
            total_displacements = np.sum(indiv_displacements, axis=0)
            result_map[x, y, :] = total_displacements.squeeze()

        log.append(time.time() - time_start)

        return result_map, log
