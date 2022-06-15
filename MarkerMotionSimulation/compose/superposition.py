import numpy as np

from Basics.params import shear_friction, normal_friction
from Basics.sensorParams import H, W, D, PPMM


def crop_map(deform_map):
    d = D // 2
    h = H // 2
    w = W // 2
    return deform_map[d - h : d + h, d - w : d + w, :]


class SuperPosition:
    def __init__(self, fem_path):
        fem_data = np.load(fem_path, allow_pickle=True)
        self.tensor_map = fem_data["tensorMap"]
        self.sparse_mask = fem_data["nodeMask"]

    def propagate_deform(self, raw_deform, contact_mask, gel_map):
        all_points = np.nonzero(self.sparse_mask)
        contact_points = np.nonzero(np.logical_and(self.sparse_mask, contact_mask))

        deform_map = np.zeros((D, D, 3))
        deform_map[contact_mask, 0:2] = raw_deform[0:2] * PPMM
        deform_map[:, :, 2] = gel_map

        result_map = np.zeros((D, D, 3))

        for x, y in zip(*contact_points):
            relative = np.array([x, y])[:, np.newaxis] - contact_points + D // 2

            mask_x = (0 <= relative[0]) & (relative[0] < D)
            mask_y = (0 <= relative[1]) & (relative[1] < D)
            mask = mask_x & mask_y

            tensors = self.tensor_map[relative[0, mask], relative[1, mask], :, :]
            tensors[:, 0:2, 2] *= normal_friction
            tensors[:, 0:2, 0:2] *= shear_friction

            deforms = deform_map[
                contact_points[0][mask], contact_points[1][mask], :, np.newaxis
            ]
            result_map[x, y, :] = np.sum(
                np.matmul(tensors, deforms, dtype=float), axis=0
            ).squeeze()

        for x, y in zip(*all_points):
            relative = np.array([x, y])[:, np.newaxis] - contact_points + D // 2

            mask_x = (0 <= relative[0]) & (relative[0] < D)
            mask_y = (0 <= relative[1]) & (relative[1] < D)
            mask = mask_x & mask_y

            tensors = self.tensor_map[relative[0, mask], relative[1, mask], :, :]
            tensors[:, 0:2, 2] *= normal_friction
            tensors[:, 0:2, 0:2] *= shear_friction

            deforms = deform_map[
                contact_points[0][mask], contact_points[1][mask], :, np.newaxis
            ]
            result_map[x, y, :] = np.sum(
                np.matmul(tensors, deforms, dtype=float), axis=0
            ).squeeze()

        return crop_map(result_map)
