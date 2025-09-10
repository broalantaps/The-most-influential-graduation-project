import runpy
from vllm import ModelRegistry
from modeling_qwen_agentic import Qwen3AgenticForCausalLM

ModelRegistry.register_model("Qwen3AgenticForCausalLM", Qwen3AgenticForCausalLM)

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")