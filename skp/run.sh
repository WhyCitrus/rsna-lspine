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


ts --gpus 2 python train.py cfg_predict_axial_subarticular_coords --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_predict_axial_subarticular_coords --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_predict_axial_subarticular_coords --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_predict_axial_subarticular_coords --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_predict_axial_subarticular_coords --sync_batchnorm --benchmark --fold 4


rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_sagittal_canal_crops .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/2773343225 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/490052995 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/3109648055 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/3387993595 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/1261271580 .
rsync -raz --progress -e 'ssh -p 26934' ian@3.tcp.ngrok.io:/home/ian/projects/rsna-lspine/data/train_pngs/2507107985 .


ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug

python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gt_subarticular_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug

ts --gpus 2 python train.py cfg0_retinanet_mobilenetv3_foramen --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_retinanet_mobilenetv3_foramen --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_retinanet_mobilenetv3_foramen --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_retinanet_mobilenetv3_foramen --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_retinanet_mobilenetv3_foramen --sync_batchnorm --benchmark --fold 4
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 4


ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_retinanet_efficientnetv2s_foramen --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_foramen_seg_cls --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_gen_det_foraminal_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_gt_foraminal_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_gen_spinal_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_gt_spinal_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_foramina_3d_crops --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_axial_subarticular_slice_identifier_with_level_seqmodel --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
ts --gpus 2 python train.py cfg_subarticular_patches_1ch_v4--sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
ts --gpus 2 python train.py cfg_subarticular_patches_v3 --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_subarticular_patches_v3 --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_subarticular_patches_v6 --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_subarticular_patches_v5 --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_subarticular_patches_v5 --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_subarticular_patches_v5 --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_subarticular_patches_v5 --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_subarticular_patches_v5 --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg_identify_subarticular_slices_with_level --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg_identify_subarticular_slices_with_level --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg_identify_subarticular_slices_with_level --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg_identify_subarticular_slices_with_level --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg_identify_subarticular_slices_with_level --sync_batchnorm --benchmark --fold 4




ln -s "../../skp/experiments/cfg_predict_sagittal_foramina_coords/312b2196/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_foramina_coords/19164216/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_foramina_coords/ed7c9c8a/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_foramina_coords/b0a5b7d6/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_foramina_coords/c4217cb1/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg_predict_sagittal_canal_coords/016642e2/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_canal_coords/97cd59e2/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_canal_coords/a3b02859/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_canal_coords/c9e60103/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg_predict_sagittal_canal_coords/74592ed9/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg_identify_subarticular_slices_with_level/a0eaf12e/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg_identify_subarticular_slices_with_level/51176343/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg_identify_subarticular_slices_with_level/4b2e01b9/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg_identify_subarticular_slices_with_level/0a8cbac8/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg_identify_subarticular_slices_with_level/50ac81cd/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg_axial_subarticular_coords/0427e248/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg_axial_subarticular_coords/d0ab9f20/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg_axial_subarticular_coords/93cbb14f/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg_axial_subarticular_coords/2a89cdd3/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg_axial_subarticular_coords/fd2f35d2/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg0_gen_subarticular_crops/cb5473ff/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg0_gen_subarticular_crops/d82ae2ee/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg0_gen_subarticular_crops/2cdcd0ec/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg0_gen_subarticular_crops/4620efa7/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg0_gen_subarticular_crops/ea5ff58f/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg0_gen_foraminal_crops/5855fd35/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg0_gen_foraminal_crops/eca1fb70/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg0_gen_foraminal_crops/1b4003f3/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg0_gen_foraminal_crops/ac2cc4bb/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg0_gen_foraminal_crops/2d78f99c/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg0_gen_spinal_crops/13f4b060/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg0_gen_spinal_crops/f29158c9/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg0_gen_spinal_crops/f35b3636/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg0_gen_spinal_crops/3fb8ec2e/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg0_gen_spinal_crops/7b91d7b1/fold4/checkpoints/last.ckpt" fold4.pt

ln -s "../../skp/experiments/cfg0_gen_all_crops/c3a47365/fold0/checkpoints/last.ckpt" fold0.pt
ln -s "../../skp/experiments/cfg0_gen_all_crops/e593a7ab/fold1/checkpoints/last.ckpt" fold1.pt
ln -s "../../skp/experiments/cfg0_gen_all_crops/7b8cd9ca/fold2/checkpoints/last.ckpt" fold2.pt
ln -s "../../skp/experiments/cfg0_gen_all_crops/079b8672/fold3/checkpoints/last.ckpt" fold3.pt
ln -s "../../skp/experiments/cfg0_gen_all_crops/ca6ea9a6/fold4/checkpoints/last.ckpt" fold4.pt