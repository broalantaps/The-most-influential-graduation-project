# 预训练代码


from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", trust_remote_code=True)