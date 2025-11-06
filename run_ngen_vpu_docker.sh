#!/bin/bash
# This script runs run_ngen_vpu.py script with various arguments
# Adjust the paths and parameters as needed for your setup
# Note this script must run with MSWM and NGEN installed in the virtual environment or container
python run_ngen_vpu_docker.py \
  --vpu vpu_03S \
  --run_name kmeans \
  --work_dir /home/yuqiong.liu/repos/nwm-region-mgr/data/ \
  --start_time 2021-10-01T00:00:00 \
  --end_time 2021-11-01T00:00:00 \
  --par_file /home/yuqiong.liu/repos/nwm-region-mgr/data/outputs/test3/params/formulation_params_kmeans_conus_vpu03S.csv \
  --pair_file /home/yuqiong.liu/repos/nwm-region-mgr/data/outputs/test3/pairs/pairs_kmeans_conus_vpu03S_mswm.csv \
  --gpkg_file /home/yuqiong.liu/repos/nwm-region-mgr/data/inputs/hydrofabric/gpkg_vpu/vpu_03S_patch.gpkg \
  --config_template /home/yuqiong.liu/repos/nwm-region-mgr/sample_files/configs/mswm.config.template.docker \
  --nprocs 2 \
  --log_file /home/yuqiong.liu/repos/nwm-region-mgr/data/region_kmeans_vpu_03S.log \
  --log_level INFO