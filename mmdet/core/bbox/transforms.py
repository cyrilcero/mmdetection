import numpy as np
import torch
from torch.onnx import is_in_onnx_export

from ..utils.misc import to_numpy


def clamp(x, min, max):
    if is_in_onnx_export():
        is_min_tensor = isinstance(min, torch.Tensor)
        is_max_tensor = isinstance(max, torch.Tensor)

        if is_min_tensor and is_max_tensor:
            y = x.clamp(min=min, max=max)
        else:
            device = x.device
            dtype = x.dtype

            y = x
            d = len(y.shape)

            min_val = torch.as_tensor(min, dtype=dtype, device=device)
            y = torch.stack(
                [y, min_val.view([1, ] * y.dim()).expand_as(y)], dim=d)
            y = torch.max(y, dim=d, keepdim=False)[0]

            max_val = torch.as_tensor(max, dtype=dtype, device=device)
            y = torch.stack(
                [y, max_val.view([1, ] * y.dim()).expand_as(y)], dim=d)
            y = torch.min(y, dim=d, keepdim=False)[0]
    else:
        y = x.clamp(min=min, max=max)

    return y


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal" and
            "vertical". Default: "horizontal"


    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
    return flipped


def bbox_mapping(bboxes,
                 img_shape,
                 scale_factor,
                 flip,
                 flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            rois = bboxes[:, :4].reshape(-1, 4)
            rois = torch.nn.functional.pad(rois, (1, 0, 0, 0), value=img_id)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = to_numpy(bboxes)
        labels = to_numpy(labels)
        return [bboxes[labels == i, :] for i in range(num_classes)]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = clamp(x1, min=0, max=max_shape[1])
        y1 = clamp(y1, min=0, max=max_shape[0])
        x2 = clamp(x2, min=0, max=max_shape[1])
        y2 = clamp(y2, min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)
