#set -ex

CUDA_VISIBLE_DEVICS=0 python3 train.py  \
  --dataroot path-to-traindata  \
  --name palm_vein_128_black \
  --load_size 141   \
  --crop_size 128   \
  --input_nc 1     \
  --output_nc 1    \
  --niter 30  \
  --niter_decay 30 \
  --use_dropout   \
   --lambda_id 0.0  \
   --lr 1e-4   \
  --use_dropout \
  --upsample bilinear \
  --netG unet_128 \
  --netD basic_128_multi \
  --netE resnet_128 \
  --dataset_mode two



cd /root/gpu_trainer/
sh start0.sh 
