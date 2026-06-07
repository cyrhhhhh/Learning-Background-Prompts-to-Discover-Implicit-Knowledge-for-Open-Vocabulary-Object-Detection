_base_ = './baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini50.py'

train_cfg = dict(max_iters=100, val_interval=1000)
default_hooks = dict(
    logger=dict(interval=10),
    checkpoint=dict(interval=25, max_keep_ckpts=3))
log_processor = dict(window_size=10)
