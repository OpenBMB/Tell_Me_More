#! /bin/bash
# only one gpu!
export CUDA_VISIBLE_DEVICES=1  # TODO

MASTER_ADDR=localhost
MASTER_PORT=12347   # TODO
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/data"
PROJECT_PATH="${BASE_PATH}/agent"
CKPT_DIRECTORY_NAME="ckpts8" # TODO
CKPT_DIRECTORY="${PROJECT_PATH}/${CKPT_DIRECTORY_NAME}" 
subdirectory_name="step_1737" # TODO

OPTS=""
# TODO: dataset config
OPTS+=" --data_dir ${BASE_PATH}/datasets/agent_data/v2/interaction_data_test.jsonl"

OPTS+=" --max_target_length 1024"
OPTS+=" --max_train_samples 0"
# OPTS+=" --max_test_samples 100"
# model config
OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/mistral-7b"
OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"
# other config
OPTS+=" --batch_size_per_device 8"
OPTS+=" --tensorboard ${PROJECT_PATH}/tensorboard_eval/${subdirectory_name}/"`date +"%Y%m%d%H%M%S"`
# TODO:
OPTS+=" --output_dir ${PROJECT_PATH}/output/${CKPT_DIRECTORY_NAME}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/test.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

${CMD}