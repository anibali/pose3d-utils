import torch
from torchvision.transforms.functional import adjust_hue
from PIL import Image, ImageEnhance
from abc import ABC, abstractmethod
import math

from . import ensure_homogeneous
from .camera import CameraIntrinsics


class Transformer(ABC):
    @abstractmethod
    def transform(self, obj):
        pass

    @abstractmethod
    def untransform(self, obj):
        pass


class CameraTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.matrix = torch.eye(3).double()
        self.sx = 1
        self.sy = 1

    def _mul_affine(self, matrix, A=None, t=None):
        aff = torch.eye(3).double()
        if A is not None:
            aff[0:2, 0:2].copy_(torch.as_tensor(A, dtype=torch.float64))
        if t is not None:
            aff[0:2, 2].copy_(torch.as_tensor(t, dtype=torch.float64))
        return torch.mm(aff, matrix)

    def affine(self, A=None, t=None):
        self.matrix = self._mul_affine(self.matrix, A, t)

    def scale_image(self, sx, sy):
        self.affine(
            A=[[sx,  0],
               [ 0, sy]]
        )
        self.sx *= sx
        self.sy *= sy

    def transform(self, camera: CameraIntrinsics):
        camera = camera.clone()
        x_0, y_0 = camera.x_0, camera.y_0
        camera.x_0, camera.y_0 = 0, 0
        camera.matrix = torch.mm(self.matrix.type_as(camera.matrix), camera.matrix)
        camera.x_0, camera.y_0 = x_0 * self.sx, y_0 * self.sy
        return camera

    def untransform(self, camera: CameraIntrinsics):
        camera = camera.clone()
        x_0, y_0 = camera.x_0, camera.y_0
        camera.x_0, camera.y_0 = 0, 0
        camera.matrix = torch.mm(self.matrix.inverse().type_as(camera.matrix), camera.matrix)
        camera.x_0, camera.y_0 = x_0 / self.sx, y_0 / self.sy
        return camera


def affine_warp(image: Image.Image, matrix, output_size):
    """Apply an affine transformation to the image.

    Args:
        image (Image.Image): The input image to transform
        matrix (torch.DoubleTensor): The affine transformation matrix
        output_size (tuple of int): The size of the output image

    Returns:
        The transformed image
    """
    inv_matrix = matrix.inverse().contiguous()
    image = image.transform(
        output_size,
        Image.AFFINE,
        tuple(inv_matrix[0:2].view(6)),
        Image.BILINEAR
    )
    return image


class ImageTransformer(Transformer):
    """Image transformer.

    Args:
        width: Input image width
        height: Input image height
        msaa: Multisample anti-aliasing scale factor
    """

    def __init__(self, width, height, x0, y0, msaa=1):
        super().__init__()
        self.msaa = msaa
        self.matrix = torch.eye(3).double()
        self.dest_size = torch.DoubleTensor([width, height])
        self.orig_width = width
        self.orig_height = height
        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        self.x0 = x0
        self.y0 = y0

    @property
    def output_size(self):
        """Dimensions of the transformed image in pixels (width, height)."""
        return tuple(self.dest_size.round().int().tolist())

    def adjust_colour(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def hflip(self):
        self.affine(
            A=[[-1, 0],
               [ 0, 1]]
        )

    def _mul_affine(self, matrix, A=None, t=None):
        aff = torch.eye(3).double()
        if A is not None:
            aff[0:2, 0:2].copy_(torch.as_tensor(A, dtype=torch.float64))
        if t is not None:
            aff[0:2, 2].copy_(torch.as_tensor(t, dtype=torch.float64))
        return torch.mm(aff, matrix)

    def affine(self, A=None, t=None):
        self.matrix = self._mul_affine(self.matrix, A, t)

    def translate(self, x, y):
        self.affine(t=[x, y])

    def rotate(self, radians):
        """Rotate the image counter clockwise."""
        self.affine(
            A=[[ math.cos(radians), math.sin(radians)],
               [-math.sin(radians), math.cos(radians)]],
        )

    def zoom(self, sx, sy=None):
        sy = sx if sy is None else sy
        self.affine(
            A=[[sx,  0],
               [ 0, sy]]
        )

    def resize(self, width, height):
        new_dest_size = torch.DoubleTensor([width, height])
        scale = new_dest_size / self.dest_size
        self.dest_size = new_dest_size
        self.zoom(scale[0], scale[1])

    def _transform_colour(self, image):
        enhancers = [
            (ImageEnhance.Brightness, self.brightness),
            (ImageEnhance.Contrast, self.contrast),
            (ImageEnhance.Color, self.saturation),
        ]
        for Enhancer, factor in enhancers:
            if abs(factor - 1.0) > 1e-9:
                image = Enhancer(image).enhance(factor)
        if abs(self.hue) > 1e-9:
            image = adjust_hue(image, self.hue)
        return image

    def _transform_image(self, image, inverse=False):
        # Move principle point to origin
        matrix = torch.DoubleTensor(
            [[1, 0, -self.x0],
             [0, 1, -self.y0],
             [0, 0, 1]]
        )

        # Apply transformations
        matrix = self.matrix.mm(matrix)

        # Restore principle point
        ow, oh = self.dest_size.tolist()
        matrix = self._mul_affine(matrix, t=[self.x0 * ow / self.orig_width, self.y0 * oh / self.orig_height])

        output_size = self.dest_size.round().int()
        if inverse:
            matrix = matrix.inverse()
            output_size = torch.IntTensor([self.orig_width, self.orig_height])

        # Scale up
        if self.msaa != 1:
            matrix = self._mul_affine(
                matrix,
                A=[[self.msaa,         0],
                   [        0, self.msaa]]
            )

        # Apply image transformation
        trans_output_size = tuple(output_size * self.msaa)
        image = affine_warp(image, matrix, trans_output_size)

        # Scale down to output size
        if self.msaa != 1:
            image = image.resize(tuple(output_size), Image.ANTIALIAS)

        return image

    def transform(self, image: Image.Image):
        image = self._transform_image(image, inverse=False)
        image = self._transform_colour(image)
        return image

    def untransform(self, image: Image.Image):
        return self._transform_image(image, inverse=True)


class PointTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.matrix = torch.eye(4).double()
        self.shuffle_indices = []

    def affine(self, A=None, t=None):
        aff = torch.eye(4).double()
        if A is not None:
            aff[0:3, 0:3].copy_(torch.as_tensor(A, dtype=torch.float64))
        if t is not None:
            aff[0:3, 3].copy_(torch.as_tensor(t, dtype=torch.float64))
        self.matrix = torch.mm(aff, self.matrix)

    def rotate(self, axis, theta):
        axis = axis[:3]
        axis = axis / axis.norm(2)
        ux, uy, uz = list(axis)
        cos = math.cos(theta)
        sin = math.sin(theta)
        self.affine(
            A=[[cos + ux*ux*(1 - cos),  ux*uy*(1-cos) - uz*sin, ux*uz*(1-cos) + uy*sin],
               [uy*ux*(1-cos) + uz*sin, cos + uy*uy*(1-cos),    uy*uz*(1-cos) - ux*sin],
               [uz*ux*(1-cos) - uy*sin, uz*uy*(1-cos) + ux*sin, cos + uz*uz*(1-cos)]]
        )

    def is_similarity(self):
        """Check whether the matrix represents a similarity transformation.

        Under a similarity transformation, relative lengths and angles remain unchanged.
        """
        eps = 1e-12
        A = self.matrix[:-1, :-1]
        v = self.matrix[-1]
        _, s, _ = torch.svd(A)
        return s.max() - s.min() < eps and ((v - v.new([0, 0, 0, 1])).abs() < eps).all()

    def reorder_points(self, indices):
        # Prevent shuffle indices from being set multiple times
        assert len(self.shuffle_indices) == 0
        self.shuffle_indices = indices

    def transform(self, points: torch.DoubleTensor):
        points = ensure_homogeneous(points, d=3)
        if len(self.shuffle_indices) > 0:
            index = torch.LongTensor(self.shuffle_indices).unsqueeze(-1).expand_as(points)
            points = points.gather(-2, index)
        return torch.mm(points, self.matrix.t())

    def untransform(self, points: torch.DoubleTensor):
        points = ensure_homogeneous(points, d=3)
        if len(self.shuffle_indices) > 0:
            inv_shuffle_indices = list(range(len(self.shuffle_indices)))
            for i, j in enumerate(self.shuffle_indices):
                inv_shuffle_indices[j] = i
            index = torch.LongTensor(inv_shuffle_indices).unsqueeze(-1).expand_as(points)
            points = points.gather(-2, index)
        return torch.mm(points, self.matrix.inverse().t())


class TransformerContext:
    def __init__(self, camera, image_width, image_height, msaa=2):
        self.orig_camera = camera
        self.camera_transformer = CameraTransformer()
        self.image_transformer = ImageTransformer(image_width, image_height, camera.x_0, camera.y_0, msaa=msaa)
        self.point_transformer = PointTransformer()

    def add(self, transform, camera=True, image=True, points=True):
        if points:
            transform.add_point_transform(self)
        if camera:
            transform.add_camera_transform(self)
        if image:
            transform.add_image_transform(self)

    def transform(self, camera=None, image=None, points=None):
        pairs = [
            (camera, self.camera_transformer),
            (image, self.image_transformer),
            (points, self.point_transformer),
        ]
        return tuple([t.transform(obj) if obj is not None else None for obj, t in pairs])
