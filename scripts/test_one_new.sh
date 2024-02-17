#! /bin/bash
# only one gpu!
export CUDA_VISIBLE_DEVICES=0

MASTER_ADDR=localhost
MASTER_PORT=12346
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
OPTS+=" --data_dir ./data/IN3/test.jsonl"

# ====== Mistral-Interact ==
OPTS+=" --model_name mistral-interact"
OPTS+=" --model_name_or_path ./models/MI-mc" # TODO: Change the path to the model after converting to model-center weights

# ====== Llama2-Interact ==
# OPTS+=" --model_name llama2-interact"
# OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
# OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"

OPTS+=" --output_dir ./outputs"
OPTS+=" --start_from 0"

CMD="torchrun ${DISTRIBUTED_ARGS} ./src/test_one_new.py ${OPTS}"

${CMD}