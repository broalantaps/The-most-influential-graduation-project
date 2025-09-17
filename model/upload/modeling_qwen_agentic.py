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

from configuration_qwen_agentic import Qwen3AgenticConfig

logger = logging.get_logger(__name__)

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_hidden_size = config.input_hidden_size
        self.output_hidden_size = config.output_hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.input_hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MemoryConverter(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config=config)
        self.rms_norm = nn.RMSNorm(config.output_hidden_size, eps=config.rms_norm_eps)
        
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

            # self.compressor = AutoModel.from_config(compressor_config)
            self.compressor = AutoModel.from_pretrained(
                self.config.compressor_config.base_model if hasattr(self.config.compressor_config, "base_model") else "Qwen/Qwen3-0.6B",
                dtype=compressor_config.torch_dtype
            )
            

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
                self.config.decoder_config.base_model if hasattr(self.config.decoder_config, "base_model") else "Qwen/Qwen3-8B"
            )
            # self.decoder = AutoModel.from_config(decoder_config)
            # logger.info(f"Init decoder from {self.config.decoder_config.base_model}")
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.config.decoder_config.base_model if hasattr(self.config.decoder_config, "base_model") else "Qwen/Qwen3-8B",
                dtype=decoder_config.torch_dtype
            )
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        self.compressor()
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


class Qwen3AgenticForCausalLM(Qwen3AgenticModel):
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