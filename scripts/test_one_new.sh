#! /bin/bash
# only one gpu!
export CUDA_VISIBLE_DEVICES=0  # TODO

MASTER_ADDR=localhost
MASTER_PORT=12346   # TODO
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1


BASE_PATH="/data"
PROJECT_PATH="${BASE_PATH}/Mistral-Interact"
CKPT_DIRECTORY_NAME="mistral" # TODO
CKPT_DIRECTORY="${PROJECT_PATH}/${CKPT_DIRECTORY_NAME}" 
subdirectory_name="step_1737" # TODO

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
# TODO
OPTS+=" --data_dir ${BASE_PATH}/datasets/agent_data/vagueness_augmented_test.jsonl"

# TODO: change the model to test
# ====== Mistral-Interact ==
OPTS+=" --model_name mistral-interact"
OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/mistral-7b"
OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"

# ====== Llama2-Interact ==
# OPTS+=" --model_name llama2-interact"
# OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
# OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"

# ====== Mistral-7B-Instruct-v0.2 ======
# OPTS+=" --model_name mistral-7b-instruct-v0.2"
# OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/mistral-7b-instruct-v0.2"

# ====== Llama-2-7b-chat ======
# OPTS+=" --model_name llama-2-7b-chat"
# OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b-chat"

# TODO
OPTS+=" --output_dir ${PROJECT_PATH}/interaction_output_case_study/${CKPT_DIRECTORY_NAME}"
OPTS+=" --start_from 0" # TODO

CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/test_one_new.py ${OPTS}"

${CMD}