import json
import unittest

import PIL.Image
import numpy as np
import torch
from importlib_resources import open_binary, open_text

import pose3d_utils.transforms as transforms
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.skeleton import CANONICAL_SKELETON_DESC
from pose3d_utils.transformers import TransformerContext


class TestTransforms(unittest.TestCase):
    def setUp(self):
        with open_binary('tests.data', 'example02_image.jpg') as f:
            self.image: PIL.Image.Image = PIL.Image.open(f).copy()

        with open_text('tests.data', 'example02_camera.json') as f:
            camera_params = json.load(f)
            self.camera = CameraIntrinsics(torch.tensor(camera_params['intrinsic'])[:3])

        with open_text('tests.data', 'example02_univ_annot3.txt') as f:
            self.points = torch.as_tensor(np.loadtxt(f))

    def assert_synced(self, camera, image, points):
        """Assert that the results of a transformation project into image space correctly.

        The 3D points are projected into 2D space, and the corresponding pixels are sampled from
        the image. These are then compared to values obtained the same way from before the
        transformation to determine whether they agree.
        """
        points2d = self.camera.project_cartesian(self.points)
        points2d_t = camera.project_cartesian(points)

        n_points = len(points2d)
        loss = 0
        for i in range(n_points):
            expected = torch.as_tensor(self.image.getpixel(tuple(points2d[i].tolist())), dtype=torch.float32)
            actual = torch.as_tensor(image.getpixel(tuple(points2d_t[i].tolist())), dtype=torch.float32)
            loss += torch.dist(expected, actual).item()
        loss /= n_points

        self.assertAlmostEqual(loss, 0, delta=5.0)

    def assert_transform_equality(self, ctx):
        camera = self.camera.clone()
        camera.x_0, camera.y_0 = 0, 0

        lhs = ctx.image_transformer.matrix.mm(camera.matrix.double())
        rhs = ctx.camera_transformer.matrix.mm(camera.matrix.double()).mm(ctx.point_transformer.matrix)
        self.assertLessEqual((lhs - rhs).abs().max(), 1e-6)

    def test_pan(self):
        ctx = TransformerContext(self.camera, self.image.width, self.image.height, msaa=1)
        ctx.add(transforms.PanImage(50, -20))

        transformed = ctx.transform(self.camera, self.image, self.points)
        self.assert_synced(*transformed)
        self.assert_transform_equality(ctx)

    def test_zoom(self):
        ctx = TransformerContext(self.camera, self.image.width, self.image.height, msaa=1)
        ctx.add(transforms.ZoomImage(1.4))

        transformed = ctx.transform(self.camera, self.image, self.points)
        self.assert_synced(*transformed)
        self.assert_transform_equality(ctx)

    def test_hflip(self):
        ctx = TransformerContext(self.camera, self.image.width, self.image.height, msaa=1)
        ctx.add(transforms.HorizontalFlip(CANONICAL_SKELETON_DESC.hflip_indices, True))

        tcamera, timage, tpoints = ctx.transform(self.camera, self.image, self.points)
        self.assert_synced(tcamera, timage, tpoints[CANONICAL_SKELETON_DESC.hflip_indices])
        self.assert_transform_equality(ctx)

    def test_rotate(self):
        ctx = TransformerContext(self.camera, self.image.width, self.image.height, msaa=1)
        ctx.add(transforms.RotateImage(30))

        transformed = ctx.transform(self.camera, self.image, self.points)
        self.assert_synced(*transformed)
        self.assert_transform_equality(ctx)

    def test_square_crop(self):
        ctx = TransformerContext(self.camera, self.image.width, self.image.height, msaa=1)
        ctx.add(transforms.SquareCrop())
        ctx.add(transforms.ChangeResolution(256, 256))

        transformed = ctx.transform(self.camera, self.image, self.points)
        self.assert_synced(*transformed)
        self.assert_transform_equality(ctx)
        self.assertEqual(transformed[1].size, (256, 256))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    unittest.main()
