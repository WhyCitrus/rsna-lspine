# debug
python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True --neptune_mode debug
python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True --neptune_mode debug
python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True --neptune_mode debug
python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True --neptune_mode debug

# ts
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_foramen_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_spinal_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 0 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 1 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 2 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 3 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone coatnet_1_rw_224 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone maxvit_tiny_tf_512 --backbone_img_size True
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tf_efficientnetv2_m 
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone tiny_vit_21m_512.dist_in22k_ft_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone resnet200d
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --backbone dm_nfnet_f0.dm_in1k
ts --gpus 2 python train.py cfg000_genv4_subarticular_crops_bb_bce --sync_batchnorm --benchmark --fold 4 --model net_csn_r101 --num_input_channels 1 --convert_to_3d True

