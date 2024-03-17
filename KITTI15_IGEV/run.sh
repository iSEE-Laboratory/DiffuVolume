#train
python train_stereo.py --logdir ./checkpoints/kitti --restore_ckpt ./pretrained_models/kitti/kitti15.pth --train_datasets kitti
#test
python evaluate_stereo.py --restore_ckpt /home/zhengdian/code/IGEV-Stereo/checkpoints/kitti/10000_igev-stereo.pth --dataset kitti