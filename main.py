# 预训练代码


from transformers import AutoTokenizer, AutoModelForCausalLM
from model.configuration_qwen_agentic import CompressorConfig, ConverterConfig, DecoderConfig
from model.modeling_qwen_agentic import Qwen3AgenticConfig, Qwen3AgenticModel, Qwen3Agentic


config = Qwen3AgenticConfig.from_pretrained("./model")