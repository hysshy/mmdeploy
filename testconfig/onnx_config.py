onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    #output_names=['dets', 'labels', 'face_kps', 'face_zitais', 'face_mohus'],
    output_names=['feat'],
    input_shape=None,
    optimize=True,
    dynamic_axes=dict(
        input=dict({
            0: 'batch',
            2: 'height',
            3: 'width'
        }),
        dets=dict({
            0: 'batch',
            1: 'num_dets'
        }),
        labels=dict({
            0: 'batch',
            1: 'num_dets'
        })))
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1))
backend_config = dict(type='onnxruntime')
