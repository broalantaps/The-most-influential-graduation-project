#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终配置生成脚本
使用真实的 Qwen3-0.6B 和 Qwen3-14B 官方配置参数
"""

import json
import os
from typing import Dict, Any
from configuration_qwen_agentic import CompressorConfig, ConverterConfig, DecoderConfig

def load_qwen3_config(model_name: str) -> Dict[str, Any]:
    """
    加载 Qwen3 模型配置
    优先从 HuggingFace 加载，失败时使用预定义配置
    """
    try:
        from transformers import AutoConfig
        if model_name == "Qwen/Qwen3-0.6B":
            config = CompressorConfig.from_pretrained(model_name)
        elif model_name == "Qwen/Qwen3-14B":
            config = DecoderConfig.from_pretrained(model_name)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        print(f"从 HuggingFace 加载 {model_name} 成功")
        config_dict = config.to_dict()
        # 移除不需要的字段
        if "layer_types" in config_dict:
            del config_dict["layer_types"]
        return config_dict
    except ImportError:
        print(f"transformers 库未安装，使用 {model_name} 的预定义配置")
        return get_official_config(model_name)
    except Exception as e:
        print(f"从 HuggingFace 加载 {model_name} 配置失败: {e}")
        print(f"使用 {model_name} 的预定义配置")
        return get_official_config(model_name)


def get_official_config(model_name: str) -> Dict[str, Any]:
    """
    获取官方配置（基于真实的 HuggingFace 配置）
    """
    if "0.6B" in model_name:
        # 基于真实的 Qwen3-0.6B 配置
        return {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": 28,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "rope_theta": 1000000,
            "rope_scaling": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": True,
        }
    elif "14B" in model_name:
        # 基于真实的 Qwen3-14B 配置
        return {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "hidden_size": 5120,
            "intermediate_size": 13696,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": 28,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "rope_theta": 1000000,
            "rope_scaling": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": True,
        }
    else:
        raise ValueError(f"未知的模型名称: {model_name}")


def create_qwen3_agentic_config() -> Dict[str, Any]:
    """
    创建 Qwen3Agentic 配置
    基于真实的官方模型配置
    """
    print("正在加载 Qwen3-0.6B 配置...")
    qwen3_0_6b_config = load_qwen3_config("Qwen/Qwen3-0.6B")
    
    print("正在加载 Qwen3-14B 配置...")
    qwen3_14b_config = load_qwen3_config("Qwen/Qwen3-14B")
    
    # 创建转换器配置
    converter_config = {
        "model_type": "converter",
        "architectures": [
            "Qwen3MLP"
        ],
        "input_hidden_size": qwen3_0_6b_config["hidden_size"],   # 1024
        "output_hidden_size": qwen3_14b_config["hidden_size"],   # 5120
        "intermediate_size": (qwen3_0_6b_config["hidden_size"] + qwen3_14b_config["hidden_size"]) // 2,  # 3072
        "hidden_act": qwen3_0_6b_config["hidden_act"],
        "rms_norm_eps": qwen3_0_6b_config["rms_norm_eps"],
    }
    
    # 创建主配置
    main_config = {
        # 基本信息
        "model_type": "qwen3_agentic",
        "architectures": ["Qwen3Agentic"],
        # 子配置
        "compressor_config": {
            "base_model": "Qwen/Qwen3-0.6B",
            "model_type": "compressor",
            "architectures": [
                qwen3_0_6b_config['architectures'][0]
            ],
            "attention_bias": qwen3_0_6b_config["attention_bias"],
            "attention_dropout": qwen3_0_6b_config["attention_dropout"],
            "bos_token_id": qwen3_0_6b_config["bos_token_id"],
            "eos_token_id": qwen3_0_6b_config["eos_token_id"],
            "head_dim": qwen3_0_6b_config["head_dim"],
            "hidden_act": qwen3_0_6b_config["hidden_act"],
            "hidden_size": qwen3_0_6b_config["hidden_size"],
            "initializer_range": qwen3_0_6b_config["initializer_range"],
            "intermediate_size": qwen3_0_6b_config["intermediate_size"],
            "max_position_embeddings": qwen3_0_6b_config["max_position_embeddings"],
            "max_window_layers": qwen3_0_6b_config['max_window_layers'],
            "num_attention_heads": qwen3_0_6b_config["num_attention_heads"],
            "num_hidden_layers": qwen3_0_6b_config["num_hidden_layers"],
            "num_key_value_heads": qwen3_0_6b_config["num_key_value_heads"],
            "rms_norm_eps": qwen3_0_6b_config["rms_norm_eps"],
            "rope_scaling": qwen3_0_6b_config["rope_scaling"],
            "rope_theta": qwen3_0_6b_config["rope_theta"],
            "sliding_window": qwen3_0_6b_config["sliding_window"],
            "tie_word_embeddings": qwen3_0_6b_config["tie_word_embeddings"],
            "torch_dtype": qwen3_0_6b_config["dtype"],
            "transformers_version": "4.56.1",
            "use_cache": qwen3_0_6b_config["use_cache"],
            "use_sliding_window": qwen3_0_6b_config["use_sliding_window"],
            "vocab_size": qwen3_0_6b_config["vocab_size"],
            "max_segment_length": qwen3_0_6b_config["max_segment_length"],
            "mem_token_num": qwen3_0_6b_config["mem_token_num"],
        },
        
        "converter_config": converter_config,
        
        "decoder_config": {
            "model_type": "decoder",
            "base_model": "Qwen/Qwen3-14B",
            "architectures": [
                qwen3_14b_config['architectures'][0]
            ],
            "attention_bias": qwen3_14b_config["attention_bias"],
            "attention_dropout": qwen3_14b_config["attention_dropout"],
            "bos_token_id": qwen3_14b_config["bos_token_id"],
            "eos_token_id": qwen3_14b_config["eos_token_id"],
            "head_dim": qwen3_14b_config["head_dim"],
            "hidden_act": qwen3_14b_config["hidden_act"],
            "hidden_size": qwen3_14b_config["hidden_size"],
            "initializer_range": qwen3_14b_config["initializer_range"],
            "intermediate_size": qwen3_14b_config["intermediate_size"],
            "max_position_embeddings": qwen3_14b_config["max_position_embeddings"],
            "max_window_layers": qwen3_14b_config['max_window_layers'],
            "num_attention_heads": qwen3_14b_config["num_attention_heads"],
            "num_hidden_layers": qwen3_14b_config["num_hidden_layers"],
            "num_key_value_heads": qwen3_14b_config["num_key_value_heads"],
            "rms_norm_eps": qwen3_14b_config["rms_norm_eps"],
            "rope_scaling": qwen3_14b_config["rope_scaling"],
            "rope_theta": qwen3_14b_config["rope_theta"],
            "sliding_window": qwen3_14b_config["sliding_window"],
            "tie_word_embeddings": qwen3_14b_config["tie_word_embeddings"],
            "torch_dtype": qwen3_14b_config["dtype"],
            "transformers_version": "4.56.1",
            "use_cache": qwen3_14b_config["use_cache"],
            "use_sliding_window": qwen3_14b_config["use_sliding_window"],
            "vocab_size": qwen3_14b_config["vocab_size"],
        },
        
        # 自动映射
        "auto_map": {
            "AutoConfig": "configuration_qwen_agentic.Qwen3AgenticConfig",
            "AutoModel": "modeling_qwen_agentic.Qwen3Agentic",
            "AutoModelForCausalLM": "modeling_qwen_agentic.Qwen3Agentic"
        },
        
        # 其他标准字段
        "torch_dtype": "bfloat16",
        "transformers_version": "4.56.1"
    }
    
    return main_config


def save_config(config: Dict[str, Any], output_path: str = "config.json"):
    """保存配置到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 配置已保存到: {output_path}")


def main():
    """主函数"""
    print("🚀 开始生成 Qwen3 Agentic 配置...")
    
    try:
        # 创建配置
        config = create_qwen3_agentic_config()
        
        # 保存配置
        save_config(config, "config.json")
        
        # 打印详细信息
        print("\n" + "="*60)
        print("📋 Qwen3 Agentic 配置概览")
        print("="*60)
        
        print(f"🔧 模型类型: {config['model_type']}")
        print(f"🏗️  架构: {config['architectures']}")
        
        
        print("\n" + "-"*40)
        print("🗜️  COMPRESSOR (基于 Qwen3-0.6B)")
        print("-"*40)
        comp_config = config['compressor_config']
        print(f"   📦 基础模型: {comp_config['base_model']}")
        print(f"   🧠 Hidden Size: {comp_config['hidden_size']}")
        print(f"   🏢 Intermediate Size: {comp_config['intermediate_size']}")
        print(f"   📚 层数: {comp_config['num_hidden_layers']}")
        print(f"   👁️  注意力头数: {comp_config['num_attention_heads']}")
        print(f"   🔑 KV头数: {comp_config['num_key_value_heads']}")
        
        print("\n" + "-"*40)
        print("🔄 CONVERTER (尺寸转换器)")
        print("-"*40)
        conv_config = config['converter_config']
        print(f"   📥 输入维度: {conv_config['input_hidden_size']} (来自 0.6B)")
        print(f"   📤 输出维度: {conv_config['output_hidden_size']} (到 14B)")
        print(f"   ⚙️  中间层大小: {conv_config['intermediate_size']}")
        
        print("\n" + "-"*40)
        print("🎯 DECODER (基于 Qwen3-14B)")
        print("-"*40)
        dec_config = config['decoder_config']
        print(f"   📦 基础模型: {dec_config['base_model']}")
        print(f"   🧠 Hidden Size: {dec_config['hidden_size']}")
        print(f"   🏢 Intermediate Size: {dec_config['intermediate_size']}")
        print(f"   📚 层数: {dec_config['num_hidden_layers']}")
        print(f"   👁️  注意力头数: {dec_config['num_attention_heads']}")
        print(f"   🔑 KV头数: {dec_config['num_key_value_heads']}")
        
        print("\n" + "-"*40)
        print("🎛️  自定义参数")
        print("-"*40)
        # print(f"   💾 内存段长度: {config['max_segment_length']}")
        # print(f"   🎫 内存token数量: {config['mem_token_num']}")
        print(f"   🔢 数据类型: {config['torch_dtype']}")
        
        print("\n" + "="*60)
        print("✨ 配置生成完成！")
        print("="*60)
        
        return config
        
    except Exception as e:
        print(f"❌ 配置生成失败: {e}")
        raise


if __name__ == "__main__":
    config = main()
