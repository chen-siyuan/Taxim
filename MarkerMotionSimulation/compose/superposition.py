import numpy as np

from Basics.sensorParams import D, PPMM


class Superposition:

    def __init__(self, fem_path):
        """
        Prepares tensor map and sparse mask for deformation propagation

        tensor_map: (D, D, 3, 3) array recording mutual deformation tensors; we
        make the assumption that the effect of the linear displacement
        relationship only depends on relative positions, which allows us to use
        a center-calibrated (values are from the perspective of the center) DxD
        tensor map rather than needing two locations inputs (DxDxDxD)

        sparse_mask: (D, D) array indicating points at which calibration data
        is collected and available (representative points, roughly 1%); the mask
        is dense towards the center and sparse towards the boundary

        @param fem_path: path to the FEM calibration data file
        """
        fem_data = np.load(fem_path, allow_pickle=True)
        self.tensor_map = fem_data["tensorMap"]
        self.sparse_mask = fem_data["nodeMask"]

    def propagate_deform(self, raw_deform, contact_mask, gel_map):
        """
        Propagates deformations through all points after correction of active
        points' deformations

        Initial deformations for active points are composed of x and y values in
        raw_deform and the z values in gel_map; then "virtual loads" are
        computed using non-negative least squares (directly applying the
        superposition principle results in excessive deformation); the virtual
        loads are then used to propagate deformations to all points--individual
        tensors are obtained through relative positions and then multiplied with
        respective deformations; the individual displacements are then summed to
        produce the final deformation

        @param raw_deform: (3,) array recording the raw x, y, z deformations, in
        millimeters
        @param contact_mask: (D, D) array indicating points at which the object
        is in contact with the gelpad
        @param gel_map: (D, D) array representing the raw z deformations, in
        pixels
        @return: result_map: (D, D, 3) array recording the updated, corrected
        deformations for representative points (as described by sparse_mask)
        """
        # obtain coordinates for all points and for contact points, both of
        # which are two-tuples of (k1,) arrays / (k2,) arrays
        all_xs, all_ys = np.nonzero(self.sparse_mask)
        act_xs, act_ys = np.nonzero(np.logical_and(
            self.sparse_mask, contact_mask
        ))

        # initialize deform_map (D, D, 3) with x, y values from raw_deform and
        # z values from gel_map
        deform_map = np.zeros((D, D, 3))
        deform_map[act_xs, act_ys, 0:2] = raw_deform[0:2] * PPMM
        deform_map[:, :, 2] = gel_map

        # first step: correction
        """
        TODO: obtain virtual loads for contact points through non-negative least
        squares instead of directly using their vanilla displacements for
        propagation; basically, need to calculate virtual displacements which,
        after propagation between active nodes, result in the current, raw
        displacements
        """

        # second step: propagation
        result_map = np.zeros((D, D, 3))
        for x, y in zip(all_xs, all_ys):
            # obtain the mask (D, D) for contact points that are within the
            # "range of effect"; this is essentially placing a DxD square
            # centered at the current point and only considering those that are
            # within its boundaries
            rel_xs = x - act_xs + D // 2
            rel_ys = y - act_ys + D // 2
            mask = (0 <= rel_xs) & (rel_xs < D) & (0 <= rel_ys) & (rel_ys < D)

            # retrieve the tensors (k3, 3, 3) and the deformations (k3, 3, 1);
            # the new axis is needed for matrix multiplication; note that the
            # tensors are retrieved through relative positions whereas the
            # deformations are retrieved through absolute positions; k3 refers
            # to the number of active points in range
            tensors = self.tensor_map[rel_xs[mask], rel_ys[mask], :, :]
            deforms = deform_map[act_xs[mask], act_ys[mask], :, np.newaxis]

            # calculate individual (k3, 3, 1) and total (3, 1) displacements
            # resulting from the effect of the active points; squeeze is needed
            # to fit within (3,) shape
            indiv_displacements = np.matmul(tensors, deforms)
            total_displacements = np.sum(indiv_displacements, axis=0)
            result_map[x, y, :] = total_displacements.squeeze()

        return result_map
