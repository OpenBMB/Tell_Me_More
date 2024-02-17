#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="."
PROJECT_PATH="."

OPTS=""
# dataset config
OPTS+=" --data_setting MTMD"
OPTS+=" --data_dir ${PROJECT_PATH}/data/interactions/interaction_data_train.jsonl"
# OPTS+=" --max_train_samples 30000"
# model config
OPTS+=" --max_seq_length 2048"
OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/mistral-7b" # change the path to your model weights
# training config
OPTS+=" --logging_step 5" 
OPTS+=" --batch_size_per_device 8"
OPTS+=" --save_step 400"
OPTS+=" --epochs 3"
OPTS+=" --lr 1e-6"
# OPTS+=" --train_iters 1000"
OPTS+=" --warmup_iters 0"
OPTS+=" --start_step 0"
OPTS+=" --loss_scale 6400"
OPTS+=" --tensorboard ${PROJECT_PATH}/tensorboard_sft/"`date +"%Y%m%d%H%M%S"`

OPTS+=" --save_dir ${PROJECT_PATH}/ckpts" # TODO

CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/sft.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

${CMD} 2>&1 | tee ${PROJECT_PATH}/logs/sft.log
