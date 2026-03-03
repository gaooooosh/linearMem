# RULER 长文本评测

简化的 RULER 评测系统，专注于长文本能力测试。

## 快速开始

```bash
cd scripts

# 基础评测 (32K上下文)
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768

# 禁用 linear memory
python run_ruler.py --model Qwen/Qwen3-1.7B --no-linear-mem

# 快速测试
python run_ruler.py --model Qwen/Qwen3-1.7B --limit 10
```

## 目录结构

```
eval/
├── scripts/
│   ├── run_ruler.py       # 评测入口脚本
│   └── eval_swaa_model.py # 模型封装 (内部使用)
├── docs/
│   └── RULER_GUIDE.md     # 详细使用文档
└── results/               # 评测结果输出
```

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--context-length` | 32768 | 上下文长度 |
| `--enable-linear-mem` | True | 启用 Linear Memory |
| `--no-linear-mem` | - | 禁用 Linear Memory |
| `--sliding-window` | 2048 | 滑动窗口大小 |
| `--keep-first` | 4 | 保留前N个token |
| `--batch-size` | auto | 批次大小 |

## 更多信息

详见 [RULER_GUIDE.md](docs/RULER_GUIDE.md)
