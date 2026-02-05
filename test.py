from qwen3mem import Qwen3MemConfig, Qwen3MemModel, Qwen3ForCausalLM, Qwen3MemDecoderLayer,Qwen3MemAttention,register_customized_qwen3,StreamingSinkCache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
register_customized_qwen3()
device = torch.device(os.environ.get("CUDA_DEVICE", "cuda:4") if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

MODEL_DIR = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model.eval()
prompt = "你好!"
messages = [
  {"role": "system", "content": "You are Qwen-Compressor, created by yongggao Xiao. You will use the compressed mem token to help you answer the user's question. You are a helpful assistant."},
  {"role": "user", "content": f"{prompt}"}
]
inputs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
cache = StreamingSinkCache(window_length=512,num_sink_tokens=4)
with torch.no_grad():
    outputs = model.generate(
        **tok(inputs, return_tensors="pt").to(device),
        past_key_values=cache,
        max_new_tokens=128,
        num_beams=1,
        # top_p=0.9,
    )

pred = tok.decode(outputs[0], skip_special_tokens=False)
print(pred)
