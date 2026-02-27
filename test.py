# make sure the "swaa_patch" folder is in your PYTHONPATH or sys.path, for example:
# import sys
# sys.path.append("./sliding-window-attention-adaptation")

# Then, before running your code, import the function hack_hf_swaa to patch transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from swaa_patch import SWAAConfig,hack_hf_swaa,hack_vllm_swaa
hack_hf_swaa(training=False)
...
# then you can load the model as usual
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map={"": device_id},
                                             dtype="bfloat16",
                                             trust_remote_code=True,
                                             attn_implementation="flash_attention_2",
                                             ).eval()
...
# finally, set your SWAA config and add it to the model's config,for example:
swaa_config = SWAAConfig(
    sliding_window_size=2048,
    keep_first=100,
    force_fa_decode=True,
    non_sliding_layers=[1,3,5,7,9,11],
)
model.config.swaa_config=swaa_config # attach SWAA config to model config. This is an informal temporary solution.
...
# now you can use the model as usual
prompt="Who are you?"
inputs = tokenizer([prompt], return_tensors="pt").to(device_id)
outputs = model.generate(**inputs)