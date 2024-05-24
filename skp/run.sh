python train.py cfg_idh_from_brats_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 1
python train.py cfg_idh_from_brats_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 2
python train.py cfg_idh_from_brats_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 3
python train.py cfg_idh_from_brats_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 4
python train.py cfg_idh_from_brats_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1


python train.py cfg_idh_preop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 0
python train.py cfg_idh_preop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 1 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/202c4ac9/fold1/checkpoints/last.ckpt
python train.py cfg_idh_preop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 2 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/fd5c4ab3/fold2/checkpoints/last.ckpt
python train.py cfg_idh_preop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 3 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/37f9db35/fold3/checkpoints/last.ckpt
python train.py cfg_idh_preop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 4 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/52ac816d/fold4/checkpoints/last.ckpt


python train.py cfg_idh_postop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 0
python train.py cfg_idh_postop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 1 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/202c4ac9/fold1/checkpoints/last.ckpt
python train.py cfg_idh_postop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 2 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/fd5c4ab3/fold2/checkpoints/last.ckpt
python train.py cfg_idh_postop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 3 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/37f9db35/fold3/checkpoints/last.ckpt
python train.py cfg_idh_postop_finetune_x3d_unet --sync_batchnorm --benchmark --neptune_mode offline --num_workers 8 --devices 1 --fold 4 --load_pretrained_model /home/neurolab/ianpan/skp/experiments/cfg_idh_from_brats_x3d_unet/52ac816d/fold4/checkpoints/last.ckpt

ts --gpus 2 python train.py cfg_head_ct_age_base_2d_top25 --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_top25 --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_top25 --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_top25 --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_top25 --sync_batchnorm --benchmark --fold 4


ts --gpus 2 python train.py cfg_head_ct_age_base_2d_predict_top25 --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_predict_top25 --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_predict_top25 --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_predict_top25 --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_head_ct_age_base_2d_predict_top25 --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_head_ct_age_2d_5stack_top25 --sync_batchnorm --benchmark --fold 0 --num_workers 8
ts --gpus 2 python train.py cfg_head_ct_age_2d_5stack_top25 --sync_batchnorm --benchmark --fold 1 --num_workers 8
ts --gpus 2 python train.py cfg_head_ct_age_2d_5stack_top25 --sync_batchnorm --benchmark --fold 2 --num_workers 8
ts --gpus 2 python train.py cfg_head_ct_age_2d_5stack_top25 --sync_batchnorm --benchmark --fold 3 --num_workers 8
ts --gpus 2 python train.py cfg_head_ct_age_2d_5stack_top25 --sync_batchnorm --benchmark --fold 4 --num_workers 8



ts --gpus 2 python train.py cfg_predict_sagittal_canal_coords --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_predict_sagittal_canal_coords --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_predict_sagittal_canal_coords --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_predict_sagittal_canal_coords --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_predict_sagittal_canal_coords --sync_batchnorm --benchmark --fold 4



rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_sagittal_canal_crops .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/2773343225 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/490052995 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/3109648055 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/3387993595 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/1261271580 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/2507107985 .


ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 0 --neptune_mode debug


ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_combined_areas_patches --sync_batchnorm --benchmark --fold 4
