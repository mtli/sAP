'''
APIs for interfacing detectors (mainly mmdetection):
'''

from functools import partial
import warnings
from os.path import basename
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import pycocotools.mask as maskUtils

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes, bbox2roi, bbox_mapping, merge_aug_masks
from mmdet.models import build_detector
from mmdet.models.detectors import SingleStageDetector, TwoStageDetector
from mmdet.models.detectors.test_mixins import MaskTestMixin
from mmdet.models.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.models.detectors.htc import HybridTaskCascade

from . import parse_mmdet_result, parse_det_result


# Below are modified functions of (an old version of)
# mmdetection's preprocessing pipeline
class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose and move to GPU
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True, 
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = False # ignores input, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device).unsqueeze(0)

        return img, img_shape, pad_shape, scale_factor


class ImageTransformGPU(object):
    """Preprocess an image ON A GPU.
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.std_inv = 1/self.std
        # self.to_rgb = to_rgb, assuming already in RGB
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, device='cuda:0'):
        h, w = img.shape[:2]
        if keep_ratio:
            if isinstance(scale, (float, int)):
                if scale <= 0:
                    raise ValueError(
                         'Invalid scale {}, must be positive.'.format(scale))
                scale_factor = scale
            elif isinstance(scale, tuple):
                max_long_edge = max(scale)
                max_short_edge = min(scale)
                scale_factor = min(max_long_edge / max(h, w),
                                max_short_edge / min(h, w))
            else:
                raise TypeError(
                    'Scale must be a number or tuple of int, but got {}'.format(
                        type(scale)))
            
            new_size = (round(h*scale_factor), round(w*scale_factor))
        else:
            new_size = scale
            w_scale = new_size[1] / w
            h_scale = new_size[0] / h
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = (*new_size, 3)

        img = torch.from_numpy(img).to(device).float()
        # to BxCxHxW
        img = img.permute(2, 0, 1).unsqueeze_(0)

        if new_size[0] != img.shape[1] or new_size[1] != img.shape[2]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore the align_corner warnings
                img = F.interpolate(img, new_size, mode='bilinear')
        if flip:
            img = torch.flip(img, 3)
        for c in range(3):
            img[:, c, :, :].sub_(self.mean[c]).mul_(self.std_inv[c])

        if self.size_divisor is not None:
            pad_h = int(np.ceil(new_size[0] / self.size_divisor)) * self.size_divisor - new_size[0]
            pad_w = int(np.ceil(new_size[1] / self.size_divisor)) * self.size_divisor - new_size[1]
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            pad_shape = (img.shape[2], img.shape[3], 3)
        else:
            pad_shape = img_shape
        return img, img_shape, pad_shape, scale_factor

# Below are modified functions of (an old version of)
# mmdetection's detectors. Note that instead of creating 
# standalone detectors, we apply modular patches to mmdetection
def _get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale):
    """Get segmentation masks from mask_pred and bboxes.

    Args:
        mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multi-scale testing,
            it will be converted to numpy array outside of this method.
        det_bboxes (Tensor): shape (n, 4/5)
        det_labels (Tensor): shape (n, )
        img_shape (Tensor): shape (3, )
        rcnn_test_cfg (dict): rcnn testing config
        ori_shape: original image size

    Returns:
        list[list]: encoded masks
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.sigmoid().cpu().numpy()
    assert isinstance(mask_pred, np.ndarray)
    # when enabling mixed precision training, mask_pred may be float16
    # numpy array
    mask_pred = mask_pred.astype(np.float32)

    # cls_segms = [[] for _ in range(self.num_classes - 1)]
    segms = []
    bboxes = det_bboxes.cpu().numpy()[:, :4]
    labels = det_labels.cpu().numpy() + 1

    if rescale:
        img_h, img_w = ori_shape[:2]
    else:
        img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
        img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
        scale_factor = 1.0

    for i in range(bboxes.shape[0]):
        bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
        label = labels[i]
        w = max(bbox[2] - bbox[0] + 1, 1)
        h = max(bbox[3] - bbox[1] + 1, 1)

        if not self.class_agnostic:
            mask_pred_ = mask_pred[i, label, :, :]
        else:
            mask_pred_ = mask_pred[i, 0, :, :]
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        bbox_mask = mmcv.imresize(mask_pred_, (w, h))
        bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
            np.uint8)
        im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
        rle = maskUtils.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        # cls_segms[label - 1].append(rle)
        segms.append(rle)

    # return cls_segms
    return segms
FCNMaskHead.get_seg_masks = _get_seg_masks
# this is affecting all masks methods

def _simple_test_mask(
        self, x, img_meta,
        det_bboxes, det_labels,
        rescale=False, decode_mask=True,
    ):
    ''' 
        this is a variant of simple_test_mask from test_mixins.py
        used for mask timing
    '''
    # image shape of the first image in the batch (only one)
    ori_shape = img_meta[0]['ori_shape']
    scale_factor = img_meta[0]['scale_factor']
    if det_bboxes.shape[0] == 0:
        segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        if decode_mask:
            return segm_result
        else:
            return None
    else:
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        _bboxes = (
            det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
        mask_rois = bbox2roi([_bboxes])
        mask_feats = self.mask_roi_extractor(
            x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_shared_head:
            mask_feats = self.shared_head(mask_feats)
        mask_pred = self.mask_head(mask_feats)
        if decode_mask:
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                            det_labels,
                                            self.test_cfg.rcnn,
                                            ori_shape, scale_factor,
                                            rescale)
            return segm_result
        else:
            mask_pred = mask_pred.sigmoid().cpu().numpy()
            mask_encoded = partial(
                self.mask_head.get_seg_masks,
                mask_pred=mask_pred,
                det_bboxes=_bboxes,
                det_labels=det_labels,
                rcnn_test_cfg=self.test_cfg.rcnn,
                ori_shape=ori_shape,
                scale_factor=scale_factor,
                rescale=rescale,
            )
            # partial is a class
            # we need to transfer required attributes as well
            mask_encoded.num_classes = self.mask_head.num_classes
            mask_encoded.class_agnostic = self.mask_head.class_agnostic
            return mask_encoded
MaskTestMixin.simple_test_mask = _simple_test_mask

def _single_stage_test(self, img, img_meta, rescale=False, numpy_res=True, decode_mask=True):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
    det_bboxes, det_labels = bbox_list[0]
    if numpy_res:
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
    return det_bboxes, det_labels
SingleStageDetector.simple_test = _single_stage_test

def _two_stage_test(self, img, img_meta, proposals=None, rescale=False, numpy_res=True, decode_mask=True):
    """simple_test without bbox2result"""
    assert self.with_bbox, "Bbox head must be implemented."

    x = self.extract_feat(img)

    proposal_list = self.simple_test_rpn(
        x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

    det_bboxes, det_labels = self.simple_test_bboxes(
        x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

    if self.with_mask:
        segm_results = self.simple_test_mask(
            x, img_meta, det_bboxes, det_labels, rescale=rescale,
            decode_mask=decode_mask,
        )

    if numpy_res:
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        if self.with_mask and decode_mask:
            segm_results = np.asarray(segm_results)

    if self.with_mask:
        return det_bboxes, det_labels, segm_results
    else:
        return det_bboxes, det_labels
TwoStageDetector.simple_test = _two_stage_test

def _cascade_rcnn_simple_test(self, img, img_meta, proposals=None, rescale=False, numpy_res=True, decode_mask=True):
    assert numpy_res
    assert not self.test_cfg.keep_all_stages
    x = self.extract_feat(img)
    proposal_list = self.simple_test_rpn(
        x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

    img_shape = img_meta[0]['img_shape']
    ori_shape = img_meta[0]['ori_shape']
    scale_factor = img_meta[0]['scale_factor']

    # "ms" in variable names means multi-stage
    ms_bbox_result = {}
    ms_segm_result = {}
    ms_scores = []
    rcnn_test_cfg = self.test_cfg.rcnn

    rois = bbox2roi(proposal_list)
    for i in range(self.num_stages):
        bbox_roi_extractor = self.bbox_roi_extractor[i]
        bbox_head = self.bbox_head[i]

        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = bbox_head(bbox_feats)
        ms_scores.append(cls_score)

        if self.test_cfg.keep_all_stages:
            det_bboxes, det_labels = bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                        bbox_head.num_classes)
            ms_bbox_result['stage{}'.format(i)] = bbox_result

            if self.with_mask:
                mask_roi_extractor = self.mask_roi_extractor[i]
                mask_head = self.mask_head[i]
                if det_bboxes.shape[0] == 0:
                    mask_classes = mask_head.num_classes - 1
                    segm_result = [[] for _ in range(mask_classes)]
                else:
                    _bboxes = (
                        det_bboxes[:, :4] *
                        scale_factor if rescale else det_bboxes)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)],
                        mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats, i)
                    mask_pred = mask_head(mask_feats)
                    segm_result = mask_head.get_seg_masks(
                        mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                        ori_shape, scale_factor, rescale)
                ms_segm_result['stage{}'.format(i)] = segm_result

        if i < self.num_stages - 1:
            bbox_label = cls_score.argmax(dim=1)
            rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                img_meta[0])

    cls_score = sum(ms_scores) / self.num_stages
    det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
        rois,
        cls_score,
        bbox_pred,
        img_shape,
        scale_factor,
        rescale=rescale,
        cfg=rcnn_test_cfg)
    
    ms_bbox_result['ensemble'] = [det_bboxes, det_labels]

    if self.with_mask:
        if det_bboxes.shape[0] == 0:
            mask_classes = self.mask_head[-1].num_classes - 1
            if decode_mask:
                segm_result = [[] for _ in range(mask_classes)]
            else:
                segm_result = None
        else:
            _bboxes = (
                det_bboxes[:, :4] *
                scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            aug_masks = []
            for i in range(self.num_stages):
                mask_roi_extractor = self.mask_roi_extractor[i]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head[i](mask_feats)
                if decode_mask:
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                else:
                    aug_masks.append(mask_pred.sigmoid())
            if decode_mask:
                merged_masks = merge_aug_masks(aug_masks,
                                            [img_meta] * self.num_stages,
                                            self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            else:
                merged_masks = torch.stack(aug_masks).mean(dim=0)
                merged_masks = merged_masks.cpu().numpy()
                mask_encoded = partial(
                    self.mask_head[-1].get_seg_masks,
                    mask_pred=merged_masks,
                    det_bboxes=_bboxes,
                    det_labels=det_labels,
                    rcnn_test_cfg=rcnn_test_cfg,
                    ori_shape=ori_shape,
                    scale_factor=scale_factor,
                    rescale=rescale,
                )
                # partial is a class
                # we need to transfer required attributes as well
                mask_encoded.num_classes = self.mask_head[-1].num_classes
                mask_encoded.class_agnostic = self.mask_head[-1].class_agnostic
                segm_result = mask_encoded

        ms_segm_result['ensemble'] = segm_result

    if numpy_res:
        det_bboxes, det_labels = ms_bbox_result['ensemble']
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        if self.with_mask:
            masks = ms_segm_result['ensemble']
            if decode_mask:
                masks = np.asarray(masks)

    if self.with_mask:
        return det_bboxes, det_labels, masks
    else:
        return det_bboxes, det_labels
CascadeRCNN.simple_test = _cascade_rcnn_simple_test

def _htc_simple_test(self, img, img_meta, proposals=None, rescale=False, numpy_res=True, decode_mask=True):
    assert numpy_res
    assert decode_mask
    assert not self.test_cfg.keep_all_stages

    x = self.extract_feat(img)
    proposal_list = self.simple_test_rpn(
        x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

    if self.with_semantic:
        _, semantic_feat = self.semantic_head(x)
    else:
        semantic_feat = None

    img_shape = img_meta[0]['img_shape']
    ori_shape = img_meta[0]['ori_shape']
    scale_factor = img_meta[0]['scale_factor']

    # "ms" in variable names means multi-stage
    ms_bbox_result = {}
    ms_segm_result = {}
    ms_scores = []
    rcnn_test_cfg = self.test_cfg.rcnn

    rois = bbox2roi(proposal_list)
    for i in range(self.num_stages):
        bbox_head = self.bbox_head[i]
        cls_score, bbox_pred = self._bbox_forward_test(
            i, x, rois, semantic_feat=semantic_feat)
        ms_scores.append(cls_score)

        if i < self.num_stages - 1:
            bbox_label = cls_score.argmax(dim=1)
            rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                img_meta[0])

    cls_score = sum(ms_scores) / float(len(ms_scores))
    det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
        rois,
        cls_score,
        bbox_pred,
        img_shape,
        scale_factor,
        rescale=rescale,
        cfg=rcnn_test_cfg)

    ms_bbox_result['ensemble'] = (det_bboxes, det_labels)

    if self.with_mask:
        if det_bboxes.shape[0] == 0:
            mask_classes = self.mask_head[-1].num_classes - 1
            segm_result = [[] for _ in range(mask_classes)]
        else:
            _bboxes = (
                det_bboxes[:, :4] *
                scale_factor if rescale else det_bboxes)

            mask_rois = bbox2roi([_bboxes])
            aug_masks = []
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks,
                                            [img_meta] * self.num_stages,
                                            self.test_cfg.rcnn)
            segm_result = self.mask_head[-1].get_seg_masks(
                merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                ori_shape, scale_factor, rescale)
        ms_segm_result['ensemble'] = segm_result

    if numpy_res:
        det_bboxes, det_labels = ms_bbox_result['ensemble']
        det_bboxes = det_bboxes.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        if self.with_mask:
            masks = ms_segm_result['ensemble']
            if decode_mask:
                masks = np.asarray(masks)

    if self.with_mask:
        return det_bboxes, det_labels, masks
    else:
        return det_bboxes, det_labels
HybridTaskCascade.simple_test = _htc_simple_test

def init_detector(opts, device='cuda:0'):
    config = mmcv.Config.fromfile(opts.config)
    new_config = 'train_pipeline' in config or 'test_pipeline' in config
    if new_config:
        assert opts.in_scale is not None, "New config does not support default input scale"
        # simulate old config
        config.data.test.img_scale = 1
        config.data.test.size_divisor = 32
    if opts.in_scale is not None:
        if 'ssd' in basename(opts.config):
            # SSD
            if opts.in_scale <= 0.2:
                # too small leads to some issues
                l = round(1920*opts.in_scale)
                config.data.test.img_scale = (l, l)
                config.data.test.resize_keep_ratio = False
            else:
                config.data.test.img_scale = opts.in_scale
                config.data.test.resize_keep_ratio = True
        else:
            config.data.test.img_scale = opts.in_scale
            config.data.test.resize_keep_ratio = True
    if opts.no_mask:
        if 'mask_head' in config.model:
            config.model['mask_head'] = None
    config.model.pretrained = None

    model = build_detector(config.model, test_cfg=config.test_cfg)
    if opts.weights is not None:
        weights = load_checkpoint(model, opts.weights)
        if 'CLASSES' in weights['meta']:
            model.CLASSES = weights['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True),
        device=device,
    )
    # for update in bbox_head.py
    if type(scale_factor) is int:
        scale_factor = float(scale_factor)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

def inference_detector(model, img, gpu_pre=True, numpy_res=True, decode_mask=True):
    # assume img has RGB channel ordering instead of BGR
    cfg = model.cfg
    if gpu_pre:
        img_transform = ImageTransformGPU(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    else:
        img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    with torch.no_grad():
        data = _prepare_data(img, img_transform, cfg, device)
        result = model(return_loss=False, rescale=True, numpy_res=numpy_res, decode_mask=decode_mask, **data) 

    return result

if __name__ == "__main__":
    from PIL import Image
    img = np.asarray(Image.open('E:/Data/COCO/val2014/COCO_val2014_000000447342.jpg'))
    img_transform = ImageTransformGPU(
        size_divisor=32, mean=[0, 0, 0], std=[255, 255, 255])
    # img = np.zeros((416, 640, 3), np.uint8)
    img = img_transform(img,
        scale=(416, 416),
        keep_ratio=True,
        device='cuda')
    img = img
