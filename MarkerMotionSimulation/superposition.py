import time
from os import path

import numpy as np
from scipy.optimize import nnls

from Basics.params import shear_friction, normal_friction
from Basics.sensorParams import PPMM, D


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
        deform_map[act_xs, act_ys, 0:2] = xy_deform
        deform_map[:, :, 2] = gel_map

        log.append(time.time() - time_start)

        act_num = act_xs.size
        matrix = np.zeros((act_num * 3, act_num * 3))
        for i, (x1, y1) in enumerate(zip(act_xs, act_ys)):
            for j, (x2, y2) in enumerate(zip(act_xs, act_ys)):
                dx = x2 - x1 + D // 2
                dy = y2 - y1 + D // 2
                tensor = np.zeros((3, 3))
                if 0 <= dx < 900 and 0 <= dy < 900:
                    tensor = self.tensor_map[dx, dy, :, :]
                for k in range(3):
                    for l in range(3):
                        matrix[i * 3 + k, j * 3 + l] = tensor[k, l]
        result = np.zeros(act_num * 3)
        for i, (x, y) in enumerate(zip(act_xs, act_ys)):
            for j in range(3):
                result[i * 3 + j] = deform_map[x, y, j]

        log.append(time.time() - time_start)

        solution = nnls(matrix, result)[0]
        for i, (x, y) in enumerate(zip(act_xs, act_ys)):
            for j in range(3):
                deform_map[x, y, j] = solution[i * 3 + j]

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
