FEATURE_PATH=/home/zhangkaicheng/RPP
ckep_num1=11865

python eval.py \
--label_gallery_path=${FEATURE_PATH}/ckep_num1/test_gallery_labels.mat \
--feature_gallery_path=${FEATURE_PATH}/ckep_num1/test_gallery_features.mat \
--label_probe_path=${FEATURE_PATH}/ckep_num1/test_probe_labels.mat \
--feature_probe_path=${FEATURE_PATH}/ckep_num1/test_probe_features.mat \
--cam_gallery_path=${FEATURE_PATH}/ckep_num1/92740/testCAM.mat \
--cam_probe_path=${FEATURE_PATH}/ckep_num1/92740/queryCAM.mat 
