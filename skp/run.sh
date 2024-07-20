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

python train.py cfg0_gen_det2_spinal_crops_embed_level --sync_batchnorm --benchmark --fold 0 --neptune_mode debug --backbone convnextv2_large.fcmae_ft_in22k_in1k_384
python train.py cfg0_gen_det2_spinal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug --data_dir ../data/train_x3d_generated_spinal_crops/canal
python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_gen_det2_spinal_crops --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 0 --neptune_mode debug
python train.py cfg00_gt_with_aug_spinal_crops_alt_loss --sync_batchnorm --benchmark --fold 0 --neptune_mode debug


python train.py cfg0_gen_det2_foraminal_crops_embed_level --sync_batchnorm --benchmark --fold 0
python train.py cfg0_gen_det2_foraminal_crops_embed_level --sync_batchnorm --benchmark --fold 1
python train.py cfg0_gen_det2_foraminal_crops_embed_level --sync_batchnorm --benchmark --fold 2
python train.py cfg0_gen_det2_foraminal_crops_embed_level --sync_batchnorm --benchmark --fold 3
python train.py cfg0_gen_det2_foraminal_crops_embed_level --sync_batchnorm --benchmark --fold 4

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

ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_v3_upsample_hard --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_upsample_hard_cases --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_upsample_hard_cases --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_upsample_hard_cases --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_upsample_hard_cases --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg0_foramen_dist_coord_seg_upsample_hard_cases --sync_batchnorm --benchmark --fold 4

ts --gpus 2 python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 0 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 1 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 2 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 3 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_subarticular_crops_3d --sync_batchnorm --benchmark --fold 4 --model net_r2plus1d 

ts --gpus 2 python train.py cfg0_gen_det2_spinal_crops_3d --sync_batchnorm --benchmark --fold 0 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_spinal_crops_3d --sync_batchnorm --benchmark --fold 1 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_spinal_crops_3d --sync_batchnorm --benchmark --fold 2 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_spinal_crops_3d --sync_batchnorm --benchmark --fold 3 --model net_r2plus1d 
ts --gpus 2 python train.py cfg0_gen_det2_spinal_crops_3d --sync_batchnorm --benchmark --fold 4 --model net_r2plus1d 

ts --gpus 2 python train.py cfg00_sagittal_canal_coords_3d_with_flips --sync_batchnorm --benchmark --fold 0
ts --gpus 2 python train.py cfg00_sagittal_canal_coords_3d_with_flips --sync_batchnorm --benchmark --fold 1
ts --gpus 2 python train.py cfg00_sagittal_canal_coords_3d_with_flips --sync_batchnorm --benchmark --fold 2
ts --gpus 2 python train.py cfg00_sagittal_canal_coords_3d_with_flips --sync_batchnorm --benchmark --fold 3
ts --gpus 2 python train.py cfg00_sagittal_canal_coords_3d_with_flips --sync_batchnorm --benchmark --fold 4

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




ln -s "../skp/experiments/cfg0_foramen_dist_each_level_no_rescale/8a36f1e7/fold0/checkpoints/last.ckpt" cfg0_foramen_dist_each_level_no_rescale/fold0.ckpt
ln -s "../skp/experiments/cfg0_foramen_dist_each_level_no_rescale/13026f5f/fold1/checkpoints/last.ckpt" cfg0_foramen_dist_each_level_no_rescale/fold1.ckpt
ln -s "../skp/experiments/cfg0_foramen_dist_each_level_no_rescale/a32b099d/fold2/checkpoints/last.ckpt" cfg0_foramen_dist_each_level_no_rescale/fold2.ckpt
ln -s "../skp/experiments/cfg0_foramen_dist_each_level_no_rescale/27abbc21/fold3/checkpoints/last.ckpt" cfg0_foramen_dist_each_level_no_rescale/fold3.ckpt
ln -s "../skp/experiments/cfg0_foramen_dist_each_level_no_rescale/246c107f/fold4/checkpoints/last.ckpt" cfg0_foramen_dist_each_level_no_rescale/fold4.ckpt

ln -s "../skp/experiments/cfg0_canal_dist_each_level_no_rescale/5d91f326/fold0/checkpoints/last.ckpt" cfg0_canal_dist_each_level_no_rescale/fold0.ckpt
ln -s "../skp/experiments/cfg0_canal_dist_each_level_no_rescale/8c15a109/fold1/checkpoints/last.ckpt" cfg0_canal_dist_each_level_no_rescale/fold1.ckpt
ln -s "../skp/experiments/cfg0_canal_dist_each_level_no_rescale/241d3c26/fold2/checkpoints/last.ckpt" cfg0_canal_dist_each_level_no_rescale/fold2.ckpt
ln -s "../skp/experiments/cfg0_canal_dist_each_level_no_rescale/10a48016/fold3/checkpoints/last.ckpt" cfg0_canal_dist_each_level_no_rescale/fold3.ckpt
ln -s "../skp/experiments/cfg0_canal_dist_each_level_no_rescale/162efef7/fold4/checkpoints/last.ckpt" cfg0_canal_dist_each_level_no_rescale/fold4.ckpt

ln -s "../skp/experiments/cfg0_subarticular_dist_from_each_level_no_rescale/45239f1d/fold0/checkpoints/last.ckpt" cfg0_subarticular_dist_from_each_level_no_rescale/fold0.ckpt
ln -s "../skp/experiments/cfg0_subarticular_dist_from_each_level_no_rescale/44271e71/fold1/checkpoints/last.ckpt" cfg0_subarticular_dist_from_each_level_no_rescale/fold1.ckpt
ln -s "../skp/experiments/cfg0_subarticular_dist_from_each_level_no_rescale/a52bc673/fold2/checkpoints/last.ckpt" cfg0_subarticular_dist_from_each_level_no_rescale/fold2.ckpt
ln -s "../skp/experiments/cfg0_subarticular_dist_from_each_level_no_rescale/b5c8056f/fold3/checkpoints/last.ckpt" cfg0_subarticular_dist_from_each_level_no_rescale/fold3.ckpt
ln -s "../skp/experiments/cfg0_subarticular_dist_from_each_level_no_rescale/b1825ae6/fold4/checkpoints/last.ckpt" cfg0_subarticular_dist_from_each_level_no_rescale/fold4.ckpt

ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_foramen_propagated/11751bb5/fold0/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_foramen_propagated/fold0.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_foramen_propagated/ac801ddb/fold1/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_foramen_propagated/fold1.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_foramen_propagated/e7a1ea96/fold2/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_foramen_propagated/fold2.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_foramen_propagated/163d6447/fold3/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_foramen_propagated/fold3.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_foramen_propagated/446fe81f/fold4/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_foramen_propagated/fold4.ckpt

ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_canal/227c5f54/fold0/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_canal/fold0.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_canal/4c179783/fold1/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_canal/fold1.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_canal/9fb4c459/fold2/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_canal/fold2.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_canal/a33f1372/fold3/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_canal/fold3.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_canal/7d0d225d/fold4/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_canal/fold4.ckpt

ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_subarticular_v2/ab8f3817/fold0/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_subarticular_v2/fold0.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_subarticular_v2/d70656d3/fold1/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_subarticular_v2/fold1.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_subarticular_v2/c6dfbcf4/fold2/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_subarticular_v2/fold2.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_subarticular_v2/dce74eae/fold3/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_subarticular_v2/fold3.ckpt
ln -s "../skp/experiments/cfg0_retinanet_efficientnetv2s_subarticular_v2/9588499a/fold4/checkpoints/last.ckpt" cfg0_retinanet_efficientnetv2s_subarticular_v2/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops/b17499ef/fold0/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops/5e397655/fold1/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops/9aae4608/fold2/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops/f87587b8/fold3/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops/2fc4c432/fold4/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops/f655b357/fold0/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops/aefa9900/fold1/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops/d5a82aaf/fold2/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops/0c3d91a0/fold3/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops/0685bdea/fold4/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_det2_subarticular_crops/2203f5d6/fold0/checkpoints/last.ckpt" cfg0_gen_det2_subarticular_crops/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_subarticular_crops/076f8268/fold1/checkpoints/last.ckpt" cfg0_gen_det2_subarticular_crops/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_subarticular_crops/1eba564a/fold2/checkpoints/last.ckpt" cfg0_gen_det2_subarticular_crops/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_subarticular_crops/0beeccb9/fold3/checkpoints/last.ckpt" cfg0_gen_det2_subarticular_crops/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_subarticular_crops/11436c15/fold4/checkpoints/last.ckpt" cfg0_gen_det2_subarticular_crops/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops_3d/845dc9ca/fold0/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops_3d/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops_3d/6143048b/fold1/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops_3d/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops_3d/3afde911/fold2/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops_3d/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops_3d/d4514649/fold3/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops_3d/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_foraminal_crops_3d/13410aaa/fold4/checkpoints/last.ckpt" cfg0_gen_det2_foraminal_crops_3d/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops_3d/aea5f073/fold0/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops_3d/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops_3d/dfbb0a20/fold1/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops_3d/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops_3d/6a58b496/fold2/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops_3d/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops_3d/542349a0/fold3/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops_3d/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_det2_spinal_crops_3d/401d0265/fold4/checkpoints/last.ckpt" cfg0_gen_det2_spinal_crops_3d/fold4.ckpt

ln -s "../skp/experiments/cfg0_gen_subarticular_full_slice/3e84c078/fold0/checkpoints/last.ckpt" cfg0_gen_subarticular_full_slice/fold0.ckpt
ln -s "../skp/experiments/cfg0_gen_subarticular_full_slice/57328963/fold1/checkpoints/last.ckpt" cfg0_gen_subarticular_full_slice/fold1.ckpt
ln -s "../skp/experiments/cfg0_gen_subarticular_full_slice/7755daee/fold2/checkpoints/last.ckpt" cfg0_gen_subarticular_full_slice/fold2.ckpt
ln -s "../skp/experiments/cfg0_gen_subarticular_full_slice/fd4d7eeb/fold3/checkpoints/last.ckpt" cfg0_gen_subarticular_full_slice/fold3.ckpt
ln -s "../skp/experiments/cfg0_gen_subarticular_full_slice/11e53e42/fold4/checkpoints/last.ckpt" cfg0_gen_subarticular_full_slice/fold4.ckpt