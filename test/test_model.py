from transformers import GPT2LMHeadModel,GPT2Config
from parameter_config import *

params = ParameterConfig()
model_config = GPT2Config.from_json_file(params.config_json)
model = GPT2LMHeadModel(model_config)
parameter = model.parameters()
num = 0
for param in parameter:
    num += param.numel()
print(num)