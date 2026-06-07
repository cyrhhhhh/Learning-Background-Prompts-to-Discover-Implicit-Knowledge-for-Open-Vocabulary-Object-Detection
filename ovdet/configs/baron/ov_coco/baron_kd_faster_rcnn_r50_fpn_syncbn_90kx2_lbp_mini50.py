_base_ = './baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_smoke.py'

train_cfg = dict(max_iters=50, val_interval=1000)
default_hooks = dict(
    logger=dict(interval=5),
    checkpoint=dict(interval=25, max_keep_ckpts=2))
log_processor = dict(window_size=5)
