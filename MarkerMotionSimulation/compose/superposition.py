import numpy as np
from scipy import interpolate

from Basics.params import shear_friction, normal_friction
from Basics.sensorParams import pixmm, H, W, D


def crop_map(deform_map):
    """
    Crops deform_map to match the output dimensions by taking a centered subarray

    @param deform_map: (3, d, d)
    @return: (3, h, w)
    """
    d = D // 2
    h = H // 2
    w = W // 2
    return deform_map[:, d - h:d + h, d - w:d + w]


class SuperPosition:

    def __init__(self, fem_path):
        fem_data = np.load(fem_path, allow_pickle=True)
        self.tensor_map = fem_data['tensorMap']
        self.sparse_mask = fem_data['nodeMask']

    def propagate_deform(self, local_deform, gel_map, contact_mask):
        local_kx = D // 2
        local_ky = D // 2

        all_points = np.where(self.sparse_mask == 1)
        contact_points = np.where((contact_mask == 1) & (self.sparse_mask == 1))
        num_points = contact_points[0].shape[0]

        uz = -1 * gel_map[(contact_mask == 1) & (self.sparse_mask == 1)]
        min_z = -1 * np.min(uz)

        # correct the motion within contact area
        activeMapZ = self.correct_KeyZ(contact_points, uz)
        activeMap = activeMapZ
        if local_deform[0] != 0.0:
            ux = np.zeros((num_points))
            factor = 1
            if local_deform[0] > 0:
                factor = -1
            ux[:] = factor * local_deform[0] / pixmm
            activeMapX = self.correct_KeyX(contact_points, ux, uz)
            activeMapX *= -1 * factor
            activeMap += activeMapX

        if local_deform[1] != 0.0:
            uy = np.zeros((num_points))
            factor = 1
            if local_deform[1] > 0:
                factor = -1
            uy[:] = factor * local_deform[1] / pixmm
            activeMapY = self.correct_KeyY(contact_points, uy, uz)
            activeMapY *= -1 * factor
            activeMap += activeMapY

        resultMap = np.zeros((3, D, D))

        for i in range(all_points[0].shape[0]):
            qx = all_points[1][i]
            qy = all_points[0][i]

            # vectorize
            kx_list = contact_points[1]
            ky_list = contact_points[0]

            # get relative position
            cur_x_list = qx - kx_list + local_kx
            cur_y_list = qy - ky_list + local_ky

            left = D * 0.
            right = D * 1.

            mask_x = (cur_x_list >= left) & (cur_x_list < right)
            mask_y = (cur_y_list >= left) & (cur_y_list < right)
            mask_valid = mask_x & mask_y

            # retrieve the mutual tensors
            T_list = self.tensor_map[cur_y_list[mask_valid], cur_x_list[mask_valid], :, :]

            # fraction factors
            T_list[:, 0:2, 2] *= normal_friction
            T_list[:, 0:2, 0:2] *= shear_friction
            # why is friction asymmetric here?

            # get the corrected motions for active nodes
            corrected_deform = activeMap[:, ky_list[mask_valid], kx_list[mask_valid]]
            cur_deform = np.matmul(T_list, corrected_deform.T[..., np.newaxis])
            total_deform = np.sum(cur_deform, axis=0)
            resultMap[:, qy, qx] = total_deform.squeeze()

        saveMap = crop_map(resultMap)
        return saveMap
