stylegan_weights = 'work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth'
d_reg_interval = 16
g_reg_interval = 4
g_reg_ratio = 0.8
d_reg_ratio = 0.9411764705882353
model = dict(
    type='PSPTransfer',
    src_generator=dict(
        type='StyleGANv2Generator',
        out_size=256,
        style_channels=512,
        num_mlps=8,
        pretrained=dict(
            ckpt_path=
            'work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth',
            prefix='generator_ema')),
    generator=dict(
        type='SwapStyleGANv2Generator',
        out_size=256,
        style_channels=512,
        num_mlps=8,
        pretrained=dict(
            ckpt_path=
            'work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='ADAStyleGAN2Discriminator',
        in_size=256,
        pretrained=dict(
            ckpt_path=
            'work_dirs/pre-trained/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth',
            prefix='discriminator'),
        data_aug=dict(
            type='ADAAug',
            aug_pipeline=dict(
                xflip=1,
                rotate90=1,
                xint=1,
                scale=1,
                rotate=1,
                aniso=1,
                xfrac=1,
                brightness=1,
                contrast=1,
                lumaflip=1,
                hue=1,
                saturation=1))),
    gan_loss=dict(type='GANLoss', gan_type='hinge'),
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        loss_weight=80.0,
        interval=16,
        norm_mode='HWC',
        data_info=dict(real_data='real_imgs', discriminator='disc')),
    gen_auxiliary_loss=dict(
        type='GeneratorPathRegularizer',
        loss_weight=8.0,
        pl_batch_shrink=2,
        interval=4,
        data_info=dict(generator='gen', num_batches='batch_size')),
    lpips_lambda=0.5,
    freezeD=5)
train_cfg = dict(use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657)),
    discriminator=dict(
        type='Adam', lr=0.0018823529411764706, betas=(0, 0.9905854573074332)))
dataset_type = 'UnconditionalImageDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(256, 256)),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type='UnconditionalImageDataset',
            imgs_root='data/IGFace_Dataset',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    key='real_img',
                    io_backend='disk'),
                dict(type='Resize', keys=['real_img'], scale=(256, 256)),
                dict(type='Flip', keys=['real_img'], direction='horizontal'),
                dict(
                    type='Normalize',
                    keys=['real_img'],
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_rgb=False),
                dict(type='ImageToTensor', keys=['real_img']),
                dict(
                    type='Collect',
                    keys=['real_img'],
                    meta_keys=['real_img_path'])
            ])),
    val=dict(imgs_root='data/IGFace_Dataset'))
checkpoint_config = dict(interval=400, by_epoch=False, max_keep_ckpts=20)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=100),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        start_iter=1,
        interp_cfg=dict(momentum=0.999),
        priority='VERY_HIGH')
]
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = True
cudnn_benchmark = True
imgs_root = 'data/IGFace_Dataset'
aug_kwargs = dict(
    xflip=1,
    rotate90=1,
    xint=1,
    scale=1,
    rotate=1,
    aniso=1,
    xfrac=1,
    brightness=1,
    contrast=1,
    lumaflip=1,
    hue=1,
    saturation=1)
lr_config = None
total_iters = 1600
use_ddp_wrapper = True
work_dir = '../em-dirs/experiments'
gpu_ids = [0]
