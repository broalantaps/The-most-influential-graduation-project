# 配置文件生成使用说明

本目录包含用于生成和管理 Qwen3 Agentic 模型配置的脚本。

## 文件说明

### 1. `configuration_qwen_agentic.py`
主要的配置类文件，包含：
- `CompressorConfig`: 压缩器配置
- `ConverterConfig`: 转换器配置  
- `DecoderConfig`: 解码器配置
- `Qwen3AgenticConfig`: 主配置类，组合上述三个配置

### 2. `config_merger.py` 
完整的配置合并器（需要 transformers 库）：
- 支持从现有配置文件加载
- 可以自定义各种配置参数
- 生成符合 HuggingFace 标准的 config.json

### 3. `simple_config_generator.py`
简化的配置生成器（无依赖）：
- 不需要安装 transformers 库
- 直接生成干净的 config.json
- 适合快速生成标准配置

## 使用方法

### 方法一：使用简化生成器（推荐）

```bash
python3 simple_config_generator.py
```

这会生成一个干净的 `config.json` 文件，包含：
- 标准的 HuggingFace 字段
- 你的模型特有字段（max_segment_length, mem_token_num）
- 子配置（converter_config, decoder_config）
- 自动映射信息

### 方法二：使用完整合并器

首先安装依赖：
```bash
pip install transformers==4.41.2
```

然后运行：
```bash
python3 config_merger.py
```

### 自定义配置

你可以修改 `simple_config_generator.py` 中的 `create_clean_config()` 函数来调整配置参数：

```python
def create_clean_config() -> Dict[str, Any]:
    config = {
        # 修改这些值来自定义你的配置
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        # ... 其他参数
    }
    return config
```

## 生成的配置结构

生成的 `config.json` 具有以下结构：

```json
{
  "model_type": "qwen3_agentic",
  "architectures": ["Qwen3Agentic"],
  "vocab_size": 151936,
  "hidden_size": 4096,
  // ... 标准 HuggingFace 字段
  
  "max_segment_length": 2048,  // 你的模型特有字段
  "mem_token_num": 512,
  
  "converter_config": {         // 子配置
    "model_type": "converter",
    "hidden_size": 3072,
    "intermediate_size": 5120
  },
  
  "decoder_config": {           // 子配置
    "model_type": "decoder",
    // ... decoder 参数
  },
  
  "auto_map": {                 // 自动映射
    "AutoConfig": "configuration_qwen_agentic.Qwen3AgenticConfig",
    "AutoModel": "modeling_qwen_agentic.Qwen3Agentic",
    "AutoModelForCausalLM": "modeling_qwen_agentic.Qwen3Agentic"
  }
}
```

## 注意事项

1. 生成的配置已经移除了内部使用的字段（如 `layer_types`），只保留标准的 HuggingFace 字段
2. 配置文件符合 HuggingFace 的标准格式，可以直接用于模型加载
3. 如果需要修改配置，建议修改生成脚本而不是直接编辑 JSON 文件
4. `auto_map` 字段指向你的自定义配置和模型类，需要确保这些类存在

## 加载配置

在你的模型代码中，可以这样加载配置：

```python
from transformers import AutoConfig
from configuration_qwen_agentic import Qwen3AgenticConfig

# 方法1：使用 AutoConfig
config = AutoConfig.from_pretrained("./")

# 方法2：直接使用自定义配置类
config = Qwen3AgenticConfig.from_pretrained("./")
```
