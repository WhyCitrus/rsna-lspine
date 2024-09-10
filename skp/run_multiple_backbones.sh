# debug
python train.py cfg000_genv5_foramina_crops_gt_half --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True --neptune_mode debug

python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True --neptune_mode debug
python train.py cfg000_genv5_foramina_crops_gt_half --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True --neptune_mode debug
python train.py cfg000_genv4_sag_subarticular_crops_gt --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True --neptune_mode debug

# ts
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_seq --sync_batchnorm --benchmark --fold 0 --backbone resnet18d --backbone_img_size False \
	--load_pretrained_backbone /home/ian/projects/rsna-lspine/skp/experiments/cfg000_axial_subarticular_slice_and_level_classifier/5f2734f3/fold0/checkpoints/last.ckpt --neptune_mode debug

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_seq --sync_batchnorm --benchmark --fold 1 --backbone resnet18d --backbone_img_size False \
	--load_pretrained_backbone /home/ian/projects/rsna-lspine/skp/experiments/cfg000_axial_subarticular_slice_and_level_classifier/df3fc7ba/fold1/checkpoints/last.ckpt

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_seq --sync_batchnorm --benchmark --fold 2 --backbone resnet18d --backbone_img_size False \
	--load_pretrained_backbone /home/ian/projects/rsna-lspine/skp/experiments/cfg000_axial_subarticular_slice_and_level_classifier/2cca523f/fold2/checkpoints/last.ckpt

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_seq --sync_batchnorm --benchmark --fold 3 --backbone resnet18d --backbone_img_size False \
	--load_pretrained_backbone /home/ian/projects/rsna-lspine/skp/experiments/cfg000_axial_subarticular_slice_and_level_classifier/90cb721b/fold3/checkpoints/last.ckpt

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_seq --sync_batchnorm --benchmark --fold 4 --backbone resnet18d --backbone_img_size False \
	--load_pretrained_backbone /home/ian/projects/rsna-lspine/skp/experiments/cfg000_axial_subarticular_slice_and_level_classifier/36dd10c2/fold4/checkpoints/last.ckpt


ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512 
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone resnet50d
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone tf_efficientnetv2_m

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True

ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 0 --backbone mobilenetv3_small_100 --backbone_img_size False
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 1 --backbone mobilenetv3_small_100 --backbone_img_size False
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 2 --backbone mobilenetv3_small_100 --backbone_img_size False
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 3 --backbone mobilenetv3_small_100 --backbone_img_size False
ts --gpus 2 python train.py cfg000_axial_subarticular_slice_and_level_classifier --sync_batchnorm --benchmark --fold 4 --backbone mobilenetv3_small_100 --backbone_img_size False

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k


ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_all_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True


ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_sag_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True

ts --gpus 2 python train.py cfg000_genv4_axial_spinal_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True 
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_axial_spinal_subarticular_crops_bb_bce_gt --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True --neptune_mode debug
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_ce_gt_level --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
