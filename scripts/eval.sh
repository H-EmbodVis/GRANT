#!/bin/bash
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

NUM_GPUS=8 
CURR_TOPK=750
CURR_QUERY=150

DATA=taskscheduling
MODEL=mask3d_lang

python main_run.py \
general.experiment_name="${MODEL}_${NUM_GPUS}GPUS" \
general.project_name="scannet200" \
general.gpus=${NUM_GPUS} \
data=${DATA} \
data/datasets=sceneverse \
data.dataset_sample_ratio=1 \
model=${MODEL}  \
data.batch_size=1 \
data.num_workers=8 \
trainer=trainer10 \
optimizer.lr=0.0008 \
general.train_mode=false \
general.timestamp=$(date +"%m-%d-%H-%M-%S") \
general.filter_scene00=false \
general.topk_per_image=${CURR_TOPK} \
general.llm_config=conf/llm/tiny_vicuna.json \
general.save_visualizations=false \
general.save_taskscheduling_visualizations=false


