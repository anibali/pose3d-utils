import json
import unittest

import numpy as np
import torch
from importlib_resources import open_text

from t3d.geom.camera import CameraIntrinsics
from t3d.skeleton_normaliser import SkeletonNormaliser
from t3d.skeleton import MPI3D_SKELETON_DESC
from t3d.geom import ensure_homogeneous


class TestNormaliser(unittest.TestCase):
    def setUp(self):
        with open_text('t3d.res', 'example01_camera.json') as f:
            camera_params = json.load(f)
            self.camera = CameraIntrinsics(torch.tensor(camera_params['intrinsic'])[:3])

        with open_text('t3d.res', 'example01_univ_annot3.txt') as f:
            self.points = torch.as_tensor(np.loadtxt(f))

        self.z_ref = 3992.29

    def test_normalise_skeleton(self):
        denorm_skel = ensure_homogeneous(self.points.clone(), d=3)
        denorm_skel[:, :2] -= denorm_skel[MPI3D_SKELETON_DESC.root_joint_id, :2]
        normaliser = SkeletonNormaliser()
        norm_skel = normaliser.normalise_skeleton(denorm_skel, self.z_ref, self.camera, 2048, 2048)
        self.assertAlmostEqual(torch.dist(norm_skel[1], torch.DoubleTensor([ 0.0215, -0.1514, -0.0127,  1.0000])).item(), 0, delta=1e-4)

    def test_denormalise_skeleton(self):
        denorm_skel = ensure_homogeneous(self.points.clone(), d=3)
        denorm_skel[:, :2] -= denorm_skel[MPI3D_SKELETON_DESC.root_joint_id, :2]
        normaliser = SkeletonNormaliser()
        norm_skel = normaliser.normalise_skeleton(denorm_skel, self.z_ref, self.camera, 2048, 2048)
        recons_skel = normaliser.denormalise_skeleton(norm_skel, self.z_ref, self.camera, 2048, 2048)
        self.assertAlmostEqual(torch.dist(recons_skel, denorm_skel).item(), 0, delta=1e-4)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    unittest.main()
