#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
# export LD_LIBRARY_PATH=/home/v-wenxichen/anaconda3/envs/slam/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

base_path=your_base_path

code_dir=examples/s2s

whisper_size=small                  # tiny base small medium large-v3
speech_encoder_path="$base_path/whisper/small.pt"   # replace this with your own whisper model path (different whisper size)
llm_path="$base_path/llm/google/Gemma-3-270m"
codec_decoder_path="$base_path/codec/CosyVoice-300M-SFT" # replace this with your own CosyVoice model path

encoder_dim=768                      # 384 512 768 896 1024 1280 
mel_size=80                          # 80 128 (128 for whisper-large only, 80 for others)
llm_dim=640

task_type=s2s

# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
per_layer_audio_vocabsize=4160     # the vocab size of the codec token per code layer
total_audio_vocabsize=$((per_layer_audio_vocabsize * code_layer))          # the vocab size of the codec token
llm_vocabsize=262144                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice
codec_decoder_type=CosyVoice
num_latency_tokens=0                # number of latency tokens (same as the number in training)
# num_latency_tokens=5                # number of latency tokens (same as the number in training)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# load the backbone model
ckpt_path=${base_path}/gemma
ckpt_name=gemma_reupload

# use peft module
use_peft=false

# model settings
group_decode=true
group_decode_adapter_type=linear

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
# decode config
max_new_tokens=3000
# force_audio_tokens=300    # 新增：当生成打到 max_new_tokens 时，截取前 100 个 audio tokens 并解码；设为 0 则禁用                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false
input_text=false

output_text_only=false
speech_sample_rate=22050
inference_online=true
online_output_dir="$base_path/examples/s2s/scripts/inference/online_output"
audio_prompt_path="$base_path/examples/s2s/audio_prompt/en/prompt_2.wav"

decode_log=$ckpt_path/s2s_decode_online_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_greedy
if [ "$do_sample" = true ] ; then
    decode_log=$ckpt_path/s2s_decode_online_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_sampling_topk${top_k}_topp${top_p}_temp${temperature}
fi

if [ "$decode_text_only" = true ] ; then
    decode_log=$decode_log"_text_only"
fi

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=Gemma-3-270M \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.inference_mode=true \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.freeze_encoder_projector=true \
        ++train_config.freeze_group_decode_adapter=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++train_config.use_peft=$use_peft \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++log_config.online_output_dir=$online_output_dir \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/gemma_reupload.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path 

# bash ./examples/s2s/scripts/inference/inference_s2s_online.sh
