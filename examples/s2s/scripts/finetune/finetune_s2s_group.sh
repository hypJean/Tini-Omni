#!/bin/bash
export OMP_NUM_THREADS=1

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

base_path=your_base_path

code_dir=examples/s2s
num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))

whisper_size=small                  # tiny base small medium large-v3
speech_encoder_path="$base_path/whisper/small.pt"   # replace this with your own whisper model path (different whisper size)
llm_path="$base_path/llm/google/Gemma-3-270m"
llm_name=Gemma-3-270M

encoder_dim=768                     # 384 512 768 1024 1280
mel_size=80                         # 80 128 ( only whisper-large-v3 supports 128 )
llm_dim=640                         # 640 for Gemma-3-270M (hidden_size)

# vocabulary settings
code_layer=3                       # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers
per_layer_audio_vocabsize=4160     # the vocab size of the codec token per code layer
total_audio_vocabsize=$((per_layer_audio_vocabsize * code_layer))  # total audio vocab across code layers
llm_vocabsize=262144                # the vocab size of the LLM model (Gemma-3-270M)
text_specialtokens=4               # special tokens for text of the LLM model (Gemma-3-270M)
text_vocabsize=$((llm_vocabsize - text_specialtokens))  # text vocab size (should match tokenizer)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice
num_latency_tokens=0                # number of delay tokens (in front of the generated audio tokens)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# dataset settings
# A.使用本地arrow数据集
manifest_format=parquet             # parquet or jsonl
train_data_path=$base_path/datasets/VoiceAssistant-400K-SLAM-Omni
val_data_path=$base_path/datasets/VoiceAssistant-400K-SLAM-Omni
load_from_cache_file=false           # set to true if you have already generated the cache file, otherwise set to false
# B.如果想继续下载数据或使用缓存数据集的话，取消下面三行的注释
# train_data_path=/root/autodl-tmp/datasets
# val_data_path=/root/autodl-tmp/datasets
# load_from_cache_file=true           # set to true if you have already generated the cache file, otherwise set to false


# training settings
train_data_ratio=1.0
batch_size_training=6
# gradient_accumulation_steps=2
use_fp16=true
use_peft=false
num_epochs=3
lr=1e-4
task_type=s2s
warmup_steps=500
total_steps=28362

# validation settings
validation_interval=4726
split_size=0.02

# model settings
group_decode=true
group_decode_adapter_type=linear

# log settings
exp_name="s2s_train_v4-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-nofp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
if [ "$use_fp16" = true ]; then
    exp_name="s2s_train_v4-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-fp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
fi
exp_name="debug"
wandb_entity_name=your_wandb_entity_name
wandb_project_name=your_wandb_project_name

home_dir=$base_path/exp/debug
output_dir=$home_dir/$exp_name
# ckpt_path=/valleblob/v-wenxichen/exp/asr/asr-Qwen2-0.5b-gpu4-btz6-lr1e-4-fp16-epochs10-whisper_small-latency5-group3/s2s_epoch_5_step_3596  # this line is for resuming training

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=linear \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.text_vocabsize=$text_vocabsize \
++model_config.vocab_config.text_specialtokens=$text_specialtokens \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.code_type=$code_type \
++model_config.group_decode=$group_decode \
++model_config.group_decode_adapter_type=$group_decode_adapter_type \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=$mel_size \
++dataset_config.seed=42 \
++dataset_config.manifest_format=$manifest_format \
++dataset_config.split_size=$split_size \
++dataset_config.train_data_ratio=$train_data_ratio \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.text_vocabsize=$text_vocabsize \
++dataset_config.vocab_config.text_specialtokens=$text_specialtokens \
++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.code_type=$code_type \
++dataset_config.num_latency_tokens=$num_latency_tokens \
++dataset_config.do_layershift=$do_layershift \
++train_config.model_name=s2s \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=$lr \
++train_config.validation_interval=$validation_interval \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$batch_size_training \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.task_type=$task_type \
++train_config.use_peft=$use_peft \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=100 \
"
# ++ckpt_path=$ckpt_path/model.pt \
# ↑ this line is for resuming training


if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    if [ "$exp_name" = "debug" ]; then
        python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            $hydra_args
    else
        python $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            $hydra_args
    fi
else
    torchrun \
        --nnodes $num_nodes \
        --nproc_per_node $num_gpus_per_node \
        --master_port=29503 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        $hydra_args
fi

# for multi-machine training, you should add the following line to the torchrun command
# --node_rank=$node_rank \
# --master_addr=$master_addr \

# bash examples/s2s/scripts/finetune/finetune_s2s_group.sh