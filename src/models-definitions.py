#swin
model = monai.networks.nets.SwinUNETR(
    img_size=(512,512), 
    in_channels=3, 
    out_channels=1, 
    use_checkpoint=True, 
    spatial_dims=2,
    depths=(4,4,6,8),
    num_heads=(4,8,16,32),
    feature_size=48,
    drop_rate=0.05,            
    attn_drop_rate=0.05
).to(device)

#maxvit
model = smp.UnetPlusPlus(
    encoder_name='tu-maxvit_base_tf_512',
    encoder_weights='imagenet',
    decoder_attention_type="scse",
    in_channels=1
).to(device)


#deeplabv3 
model = smp.DeepLabV3Plus(
        encoder_name='tu-seresnextaa101d_32x8d',
        encoder_weights='imagenet',
        in_channels=1
    )

model = smp.DeepLabV3Plus(
        encoder_name='tu-seresnextaa101d_32x8d',
        encoder_weights='imagenet',
        in_channels=1,
        decoder_atrous_rates=(6,12,24),
        decoder_channels=512
    )


#unet 3.5d
#u-maxvit_base_tf_512.pth
model = smp.Unet(
        "tu-maxvit_base_tf_512",
        in_channels=3,
        classes=3,
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=(512, 256, 128, 64, 32),
        decoder_attention_type="scse"
    )


#unetplusplus-tu-maxvit_rmlp_base_rw_384.pth
model = smp.UnetPlusPlus(
        "tu-maxvit_rmlp_base_rw_384",
        in_channels=3,
        classes=3,
        encoder_weights=None,
        encoder_depth=4,
        decoder_channels=(256, 128, 64, 32),
        decoder_attention_type="scse"
    )

#unetplusplus-tu-maxvit_rmlp_base_rw_384_large.pth
model = smp.UnetPlusPlus(
        "tu-maxvit_rmlp_base_rw_384",
        in_channels=3,
        classes=3,
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=(512, 256, 128, 64, 32),
        decoder_attention_type="scse"
    )


###########################################################################################################


#se-resnext101 32x16d  
model = smp.UnetPlusPlus(
    encoder_name='resnext101_32x16d',
    encoder_weights='imagenet',
    decoder_attention_type="scse",
    in_channels=1
).to(device)

#dpn98
model = smp.UnetPlusPlus(
    encoder_name='dpn98',
    encoder_weights='imagenet',
    decoder_attention_type="scse",
    in_channels=1
).to(device)

