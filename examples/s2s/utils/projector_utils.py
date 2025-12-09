import torch
import torch.nn as nn
from slam_llm.utils.train_utils import print_module_size
import os

class Linear_GroupDecodeAdapter(nn.Module):
    def __init__(self, audio_vocab_size, llm_hidden_size, code_layer):
        super(Linear_GroupDecodeAdapter, self).__init__()
        self.audio_vocab_size = audio_vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.code_layer = code_layer
        
        # 将LLM隐藏状态映射到多个code层的词汇表空间
        # 输入: [batch, seq_len, llm_hidden_size]
        # 输出: [batch, seq_len, code_layer * audio_vocab_size]
        self.linear = nn.Linear(llm_hidden_size, code_layer * audio_vocab_size)

    def forward(self, hidden_states):
        # 输入: hidden_states [batch, seq_len, llm_hidden_size]
        # 输出: [batch, seq_len, code_layer * audio_vocab_size]
        output = self.linear(hidden_states)
        return output


def setup_group_decode_adapter(model_config, train_config, **kwargs):
    audio_vocab_size = model_config.vocab_config.padded_audio_vocabsize  # 单个层的词汇表大小
    llm_hidden_size = model_config.llm_dim  # LLM 隐藏层维度
    code_layer = model_config.vocab_config.code_layer
    
    if model_config.group_decode_adapter_type == "linear":
        group_decode_adapter = Linear_GroupDecodeAdapter(audio_vocab_size, llm_hidden_size, code_layer)
    else:
        raise NotImplementedError

    group_decode_adapter_name = "GroupDecodeAdapter_" + model_config.group_decode_adapter_type
    print_module_size(group_decode_adapter, group_decode_adapter_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return group_decode_adapter
