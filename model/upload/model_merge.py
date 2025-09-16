# 预训练代码

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.configuration_qwen_agentic import CompressorConfig, ConverterConfig, DecoderConfig, Qwen3AgenticConfig
from model.modeling_qwen_agentic import Qwen3AgenticModel, Qwen3AgenticForCausalLM

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Qwen3AgenticConfig.from_pretrained("./model")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = Qwen3AgenticForCausalLM(config)

# model.save_pretrained
total_param = 0

for _, param in model.named_parameters():
    total_param += param.numel()

print(f"Total parameters: {total_param}")

# 保存模型
# model.save_pretrained("/home/dyh/The-most-influential-graduation-project/checkpoints/Qwen3-Agentic-9B")
tokenizer = AutoTokenizer.from_pretrained("/home/dyh/The-most-influential-graduation-project/checkpoints/Qwen3-8B")
tokenizer.save_pretrained("/home/dyh/The-most-influential-graduation-project/checkpoints/Qwen3-Agentic-9B")
# # 加载模型
# model = Qwen3AgenticForCausalLM.from_pretrained("./model_merge")