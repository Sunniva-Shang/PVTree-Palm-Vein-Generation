set -ex

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --dataroot ../pv_pattern_results/$1 \
    --name $2 \
    --n_samples 7  \
    --no_flip \
    --load_size 128 \
    --crop_size 128 \
    --input_nc 1\
    --output_nc 1 \
    --use_dropout \
    --upsample bilinear\
    --netG unet_128 \
    --netD basic_128_multi \
    --netE resnet_128 \
    --dataset_mode single
