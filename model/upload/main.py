# 预训练代码

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.configuration_qwen_agentic import CompressorConfig, ConverterConfig, DecoderConfig
from model.modeling_qwen_agentic import Qwen3AgenticConfig, Qwen3AgenticModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Qwen3AgenticConfig.from_pretrained("./model")

# model = Qwen3AgenticModel.from_pretrained("/home/dyh/The-most-influential-graduation-project/checkpoints/Qwen3-Agentic-9B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/home/dyh/The-most-influential-graduation-project/checkpoints/Qwen3-Agentic-9B")
# print(model)
# model.push_to_hub("BroAlanTaps/Qwen3-Agentic-9B")
tokenizer.push_to_hub("BroAlanTaps/Qwen3-Agentic-9B")