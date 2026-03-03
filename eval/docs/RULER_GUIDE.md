# RULER 评测使用指南

## 快速开始

```bash
# 进入评测目录
cd eval/scripts

# 基础评测 (32K上下文，启用linear memory)
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768

# 禁用 linear memory 对比测试
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768 --no-linear-mem

# 自定义滑动窗口
python run_ruler.py --model Qwen/Qwen3-1.7B --sliding-window 4096 --keep-first 8

# 快速测试 (少量样本)
python run_ruler.py --model Qwen/Qwen3-1.7B --limit 10
```

## 核心参数

### 上下文与采样
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--context-length` | 32768 | 上下文长度 (tokens) |
| `--limit` | 100 | 每任务样本数 |

### Linear Memory 配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-linear-mem` | True | 启用 Linear Memory |
| `--no-linear-mem` | - | 禁用 Linear Memory |
| `--sliding-window` | 2048 | 滑动窗口大小 |
| `--keep-first` | 4 | 保留前N个token |
| `--non-sliding-layers` | [] | 非滑动层索引 (如: 0,1,2,3) |

### 模型配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | Qwen/Qwen3-1.7B | 模型名称或路径 |
| `--device` | cuda:0 | 设备 |
| `--dtype` | bfloat16 | 数据类型 |
| `--attn` | flash_attention_2 | 注意力实现 |
| `--batch-size` | auto | 批次大小，支持 `auto` 或数字 |

## 输出结果

每次评测会在 `eval/results/` 下创建目录，格式为：
```
ruler_{上下文K}_{linear_mem状态}_{时间戳}/
├── config.json        # 完整配置
├── results.json       # 完整结果 (含配置)
├── results_table.txt  # 结果表格
└── report.md          # 详细报告
```

### 示例输出目录名
- `ruler_32k_lm_20240101_120000/` - 32K上下文，启用linear mem
- `ruler_16k_no_lm_20240101_130000/` - 16K上下文，禁用linear mem

## 测试任务

RULER 包含 10 个长文本评测任务：

| 任务 | 说明 |
|------|------|
| niah_single_1/2/3 | 单针检索 |
| niah_multikey_1 | 多键检索 |
| niah_multivalue | 多值检索 |
| niah_multiquery | 多查询检索 |
| passkey | 密码回忆 |
| ruler_vt | 变量追踪 |
| ruler_cwe | 常见词提取 |
| ruler_fwe | 频繁词提取 |

## 典型使用场景

### 1. 对比 Linear Memory 效果
```bash
# 启用 linear memory
python run_ruler.py --context-length 32768 --output-dir results/with_lm

# 禁用 linear memory
python run_ruler.py --context-length 32768 --no-linear-mem --output-dir results/without_lm
```

### 2. 不同上下文长度测试
```bash
# 16K
python run_ruler.py --context-length 16384

# 32K
python run_ruler.py --context-length 32768

# 64K
python run_ruler.py --context-length 65536
```

### 3. 不同滑动窗口配置
```bash
# 默认窗口
python run_ruler.py --sliding-window 2048

# 更大窗口
python run_ruler.py --sliding-window 4096
```

### 4. 指定非滑动层
```bash
# 前四层不使用滑动注意力
python run_ruler.py --non-sliding-layers 0,1,2,3
```

## 结果解读

- **>90%**: 优秀
- **70-90%**: 良好
- **50-70%**: 中等
- **<50%**: 需改进
