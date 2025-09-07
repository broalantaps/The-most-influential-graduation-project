#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Agentic Model Implementation
包含三个组件：Compressor (Qwen3-0.6B) + Converter + Decoder (Qwen3-14B)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.activations import ACT2FN
from transformers.utils import logging
from .configuration_qwen_agentic import Qwen3AgenticConfig

logger = logging.get_logger(__name__)

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MemoryConverter(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config=config)
        self.rms_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x):
        x = self.rms_norm(x)
        return super().forward(x)


class Qwen3AgenticModel(PreTrainedModel):
    """
    Qwen3 Agentic 模型主体
    """
    config_class = Qwen3AgenticConfig
    
    def __init__(self, config: Qwen3AgenticConfig):
        super().__init__(config)
        self.config = config
        
        self._init_compressor()
        self._init_converter()
        self._init_decoder()
        
        # DEPRECATED, Seperated into outside
        # self._init_tokenizer()

        # Init weights
        self.post_init()    

    def _init_compressor(self):
        try:
            compressor_config = AutoConfig.from_pretrained(
                self.config.compressor_config.base_model if hasattr(self.config.compressor_config, "base_model") else "Qwen/Qwen3-0.6B"
            )

            self.compressor = AutoModel.from_config(compressor_config)
            # self.compressor = AutoModel.from_pretrained(
            #     self.config.compressor_config.base_model if hasattr(self.config.compressor_config, "base_model") else "Qwen/Qwen3-0.6B",
            #     dtype=compressor_config.torch_dtype
            # )
            

        except Exception as e:
            logger.info(f"Init compressor error: {e}")
    
    def _init_converter(self):
        try:
            self.converter = MemoryConverter(self.config.converter_config)
        except Exception as e:
            logger.info(f"Init converter error: {e}")
    
    def _init_decoder(self):
        try:
            decoder_config = AutoConfig.from_pretrained(
                self.config.decoder_config.base_model if hasattr(self.config.decoder_config, "base_model") else "Qwen/Qwen3-14B"

            )
            self.decoder = AutoModelForCausalLM.from_config(decoder_config)
            # self.decoder = AutoModelForCausalLM.from_pretrained(
            #     self.config.decoder_config.base_model if hasattr(self.config.decoder_config, "base_model") else "Qwen/Qwen3-14B",
            #     dtype=decoder_config.torch_dtype
            # )
        except Exception as e:
            logger.info(f"Init decoder error: {e}")
    
    # DEPRECATED, Seperated into outside
    # def _init_tokenizer(self):
    #     try:
    #         self.tokenizer = AutoTokenizer.from_pretrained(
    #             self.config.compressor_config.base_model if hasattr(self.config.compressor_config, "base_model") else "Qwen/Qwen3-0.6B"
    #         )
            
    #         all_special_tokens = ["<|mem_start|>", "<|mem_end|>"] + [f"<|mem_{i}|>" for i in range(self.config.compressor_config.mem_token_num)]
    #         logger.info(self.tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens}))
    #         self.compressor.resize_token_embeddings(len(self.tokenizer))

    #         # TODO: init embedding weights
    #         logger.info("Waiting to test init embedding weihts, not implement now!!!")

    #     except Exception as e:
    #         logger.info(f"Init tokenizer error: {e}")

    def compress_memory(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            Comp
            
        Returns:
            memory_tokens: 压缩后的记忆tokens [batch_size, mem_token_num, hidden_size]
        """
        # 使用压缩器处理输入
        compressor_outputs = self.compressor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 获取隐藏状态
        hidden_states = compressor_outputs.last_hidden_state  # [batch_size, seq_len, 1024]
        
        # 根据配置进行记忆压缩
        max_segment_length = self.config.compressor_config.get("max_segment_length", 2048)
        mem_token_num = self.config.compressor_config.get("mem_token_num", 512)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= max_segment_length:
            # 如果序列长度小于等于最大段长度，直接压缩
            memory_tokens = self._compress_sequence(hidden_states, mem_token_num)
        else:
            # 如果序列长度超过最大段长度，分段处理
            memory_tokens = self._compress_long_sequence(
                hidden_states, max_segment_length, mem_token_num
            )
        
        return memory_tokens
    
    def _compress_sequence(self, hidden_states: torch.Tensor, mem_token_num: int) -> torch.Tensor:
        """
        压缩单个序列
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            mem_token_num: 目标记忆token数量
            
        Returns:
            memory_tokens: [batch_size, mem_token_num, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= mem_token_num:
            # 如果序列长度小于等于记忆token数量，直接返回并padding
            padding_length = mem_token_num - seq_len
            if padding_length > 0:
                padding = torch.zeros(
                    batch_size, padding_length, hidden_size,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                memory_tokens = torch.cat([hidden_states, padding], dim=1)
            else:
                memory_tokens = hidden_states
        else:
            # 使用平均池化压缩
            # 将序列重塑为 [batch_size, mem_token_num, -1, hidden_size]
            tokens_per_memory = seq_len // mem_token_num
            remainder = seq_len % mem_token_num
            
            if remainder == 0:
                # 可以整除
                reshaped = hidden_states[:, :tokens_per_memory * mem_token_num, :].view(
                    batch_size, mem_token_num, tokens_per_memory, hidden_size
                )
                memory_tokens = reshaped.mean(dim=2)  # 平均池化
            else:
                # 不能整除，处理余下的tokens
                main_part = hidden_states[:, :tokens_per_memory * mem_token_num, :].view(
                    batch_size, mem_token_num, tokens_per_memory, hidden_size
                ).mean(dim=2)
                
                # 将余下的tokens分配到最后几个memory tokens中
                remainder_part = hidden_states[:, tokens_per_memory * mem_token_num:, :]
                remainder_mean = remainder_part.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
                
                # 将余下部分加到最后一个memory token上
                memory_tokens = main_part.clone()
                memory_tokens[:, -1:, :] = (memory_tokens[:, -1:, :] + remainder_mean) / 2
        
        return memory_tokens
    
    def _compress_long_sequence(
        self, 
        hidden_states: torch.Tensor, 
        max_segment_length: int, 
        mem_token_num: int
    ) -> torch.Tensor:
        """
        压缩长序列（分段处理）
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            max_segment_length: 最大段长度
            mem_token_num: 每段的记忆token数量
            
        Returns:
            memory_tokens: [batch_size, total_mem_tokens, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算段数
        num_segments = (seq_len + max_segment_length - 1) // max_segment_length
        
        segment_memories = []
        
        for i in range(num_segments):
            start_idx = i * max_segment_length
            end_idx = min((i + 1) * max_segment_length, seq_len)
            
            segment = hidden_states[:, start_idx:end_idx, :]
            segment_memory = self._compress_sequence(segment, mem_token_num)
            segment_memories.append(segment_memory)
        
        # 拼接所有段的记忆
        memory_tokens = torch.cat(segment_memories, dim=1)
        
        return memory_tokens
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            其他参数同标准transformers模型
            
        Returns:
            模型输出
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 第一步：使用压缩器压缩输入
        memory_tokens = self.compress_memory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )  # [batch_size, mem_token_num, 1024]
        
        # 第二步：使用转换器转换维度
        converted_memory = self.converter(memory_tokens)  # [batch_size, mem_token_num, 5120]
        
        # 第三步：使用解码器生成输出
        # 注意：这里需要将转换后的记忆作为解码器的输入
        # 具体实现取决于你想如何使用这些记忆tokens
        
        # 简单实现：将转换后的记忆作为inputs_embeds传给解码器
        decoder_outputs = self.decoder(
            inputs_embeds=converted_memory,
            attention_mask=None,  # 记忆tokens通常不需要mask
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            return decoder_outputs
        
        return CausalLMOutputWithPast(
            logits=decoder_outputs.logits if hasattr(decoder_outputs, 'logits') else None,
            past_key_values=decoder_outputs.past_key_values if hasattr(decoder_outputs, 'past_key_values') else None,
            hidden_states=decoder_outputs.hidden_states if hasattr(decoder_outputs, 'hidden_states') else None,
            attentions=decoder_outputs.attentions if hasattr(decoder_outputs, 'attentions') else None,
        )


class Qwen3Agentic(Qwen3AgenticModel):
    """
    Qwen3 Agentic 模型 (用于因果语言建模)
    """
    def __init__(self, config: Qwen3AgenticConfig):
        super().__init__(config)
        
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        return outputs