import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def run_test():
    print("="*20 + " 环境核心环节测试 " + "="*20)

    # 1. 硬件与版本检查
    print(f"[1/4] 正在检查基础环境...")
    print(f"    - PyTorch 版本: {torch.__version__}")
    print(f"    - CUDA 版本: {torch.version.cuda}")
    print(f"    - GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"    - 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 2. Flash Attention 算子检测
    print(f"\n[2/4] 正在检测 Flash Attention...")
    try:
        import flash_attn
        print(f"    - Flash Attention 版本: {flash_attn.__version__}")
        # 模拟一个 BF16 张量测试
        q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
        from flash_attn import flash_attn_func
        _ = flash_attn_func(q, q, q, causal=True)
        print("    - Flash Attention 算子运行: 正常 ✅")
    except Exception as e:
        print(f"    - Flash Attention 运行失败: {e} ❌")

    model_id = "Qwen/Qwen3-1.7B"
    print(f"\n[3/4] 正在从 HuggingFace 加载模型 (使用 Flash Attention 2)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0", # 强制单卡，避免 auto 导致的跨卡张量冲突
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        
        input_text = "你好，请介绍一下 A100 GPU 的优势。"
        # 核心修正：确保 inputs 跟着 model 走
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        print(f"    - 模型运行在设备: {model.device}")
        print("    - 开始推理生成...")
        outputs = model.generate(**inputs, max_new_tokens=50)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    - 推理成功! 输出摘要: {response}...")
        print("    - 模型推理测试: 正常 ✅")
    except Exception as e:
        print(f"    - 模型加载或推理失败: {e} ❌")

    # 4. 多卡并行与通信检查 (针对无 NVLink 场景)
    print(f"\n[4/4] 正在检测多卡通信 (NCCL over PCIe)...")
    if torch.cuda.device_count() < 2:
        print("    - 警告: 检测到的 GPU 数量少于 2，跳过通信测试。")
    else:
        try:
            # 简单测试第 0 卡和第 1 卡的 P2P 访问
            can_p2p = torch.cuda.can_device_access_peer(0, 1)
            print(f"    - GPU 0/1 P2P 访问能力: {can_p2p} (无 NVLink 通常为 True 但带宽较低)")
            print("    - 多卡状态检测: 正常 ✅")
        except Exception as e:
            print(f"    - 多卡通信异常: {e} ❌")

    print("\n" + "="*20 + " 测试完成 " + "="*20)

if __name__ == "__main__":
    run_test()