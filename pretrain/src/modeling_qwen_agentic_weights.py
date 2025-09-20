
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.activations import ACT2FN
# from transformers.utils import logging
from modeling_qwen3 import Qwen3Model
from transformers.generation import GenerationMixin
from configuration_qwen_agentic import Qwen3AgenticConfig

from transformers.utils import logging, TransformersKwargs, auto_docstring, can_return_tuple
from transformers.processing_utils import Unpack
# logging.set_verbosity_info()
# logging.enable_explicit_format()


# if not logger.handlers:
#     logger_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler = logging.StreamHandler()
#     handler.setFormatter(logger_format)
#     logger.addHandler(handler)

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
    config_class = Qwen3AgenticConfig
    def __init__(self, config: Qwen3AgenticConfig):
        super().__init__(config)
        self.config = config
        
        print(f"Init compressor...")
        self._init_compressor(config=config.compressor_config)
        print(f"Init compressor success")

        print(f"Init converter...")
        self._init_converter(config=config.converter_config)
        print(f"Init converter success")

        print(f"Init decoder...")
        self._init_language_model(config=config)
        print(f"Init decoder success")

        # Init weights
        self.post_init()    

    def _init_compressor(self, config):
        # self.compressor = Qwen3Model(config=config)     
        model_name = getattr(config, "base_model", "Qwen/Qwen3-0.6B")
        self.compressor = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to('cuda')


    def _init_converter(self, config):
        self.converter = MemoryConverter(config=config)
        self.converter.apply(self._init_weights)

    def _init_language_model(self, config): 
        # self.decoder = Qwen3Model(config=config) 
        model_name = getattr(config, "base_model", "Qwen/Qwen3-8B")
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to('cuda')


    # DEPRECATED, Seperated into outside
    # def _init_tokenizer(self):
    #     try:
    #         self.tokenizer = AutoTokenizer.from_pretrained(
    #             self.config.compressor_config.base_model if hasattr(self.config.compressor_config, "base_model") else "Qwen/Qwen3-0.6B"
    #         )
            
    #         all_special_tokens = ["<|mem_start|>", "<|mem_end|>"] + [f"<|mem_{i}|>" for i in range(self.config.compressor_config.mem_token_num)]
    #         print(self.tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens}))
    #         self.compressor.resize_token_embeddings(len(self.tokenizer))

    #         # TODO: init embedding weights
    #         print("Waiting to test init embedding weihts, not implement now!!!")

    #     except Exception as e:
    #         print(f"Init tokenizer error: {e}")
    
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


class Qwen3AgenticForCausalLM(Qwen3AgenticModel, GenerationMixin):
    def __init__(self, config: Qwen3AgenticConfig):
        super().__init__(config)
        
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.vocab_size, bias=False)
        # self.config = config.decoder_config
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == "__main__":
    config = Qwen3AgenticConfig.from_pretrained("/root/The-most-influential-graduation-project/model/9B-config.json")
    model = Qwen3AgenticForCausalLM(config=config).to("cuda")
    # print(model)

    Qwen38b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).to('cuda')
    model.lm_head = Qwen38b.lm_head
    
    print("Compressor params:", sum(p.numel() for p in model.compressor.parameters()))
    print("Converter params:", sum(p.numel() for p in model.converter.parameters()))
    print("Decoder params:", sum(p.numel() for p in model.model.parameters()))
    print("LM Head params:", sum(p.numel() for p in model.lm_head.parameters()))
    print("Total params:", sum(p.numel() for p in model.parameters()))
    # 保存权重
    model.save_pretrained("/root/The-most-influential-graduation-project/ckpt/Qwen3-Agentic-9B")
    model.push_to_hub('BroAlanTaps/Qwen3-Agentic-9B')
    # # Test Qwen3AgenticForCausalLM Inference
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    # prompt = "OpenAI是一家什么公司？"
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    
    # # conduct text completion
    # # generated_ids = model.generate(
    # #     **model_inputs,
    # #     max_new_tokens=100
    # # )
    # # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # # # parsing thinking content
    # # try:
    # #     # rindex finding 151668 (</think>)
    # #     index = len(output_ids) - output_ids[::-1].index(151668)
    # # except ValueError:
    # #     index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    # print("content:", content)