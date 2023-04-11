# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER, mark
# from mmdet.models.detectors.labelstransform import simple_test_findTarget

@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.single_stage_spjgh.SingleStageDetector_SPJGH.simple_test')
def single_stage_detector_spjgh__simple_test(ctx, self, img, img_metas, rescale=False, points=None):
    """Rewrite `simple_test` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (list[dict]): Dict containing image's meta information
            such as `img_shape`.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
    """
    feat = self.extract_feat(img)
    dets, labels, keep = self.bbox_head.simple_test(
        feat, img_metas, rescale=rescale, getKeep=True)
    # return dets, labels
    # 人脸关键点识别
    face_kps = self.bbox_head.simple_test_kps(feat,  keep)
    # 人脸姿态分类
    face_zitais = self.bbox_head.simple_test_facezitai(feat, keep)
    # 人脸模糊度评估
    face_mohus = self.bbox_head.simple_test_facemohu(feat, keep)
    return dets, labels, face_kps, face_zitais, face_mohus
    # return self.bbox_head.simple_test(feat, img_metas, **kwargs)
