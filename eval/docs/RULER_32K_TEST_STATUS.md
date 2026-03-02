# RULER 32K 评测测试运行状态

## ✅ 测试已启动

**启动时间**: 2026-03-02 13:21

### 📊 测试配置

- **上下文长度**: 32K tokens
- **样本数/任务**: 100 samples
- **总任务数**: 10 tasks
- **总样本数**: 1000 samples
- **设备**: cuda:7
- **批次大小**: 1

### 🎯 测试任务

1. **niah_single_1** - NIAH 单针检索 #1
2. **niah_single_2** - NIAH 单针检索 #2
3. **niah_single_3** - NIAH 单针检索 #3
4. **niah_multikey_1** - NIAH 多键检索
5. **niah_multivalue** - NIAH 多值检索
6. **niah_multiquery** - NIAH 多查询检索
7. **passkey** - 密钥检索
8. **ruler_vt** - 变量跟踪
9. **ruler_cwe** - 代码词提取
10. **ruler_fwe** - 频繁词提取

### 🔧 模型配置

- **模型**: Qwen/Qwen3-1.7B
- **SWAA 滑动窗口**: 2048 tokens
- **Keep First**: 4 tokens
- **数据类型**: bfloat16
- **注意力实现**: flash_attention_2

### ⏱️ 预计时间

- **单样本时间**: ~2-5 秒 (32K 上下文)
- **单个任务 (100样本)**: ~3-8 分钟
- **总时间**: ~30-80 分钟

### 📁 输出目录

```
eval_results/ruler_32k_20260302_132117/
├── results.json          # 完整评测结果
├── results_table.txt     # 结果表格
└── ruler_32k_report.md   # 详细报告
```

### 🔄 查看进度

```bash
# 查看实时日志
tail -f /tmp/ruler_32k_test_v2.log

# 查看最新进度
tail -50 /tmp/ruler_32k_test_v2.log

# 检查是否有结果文件
ls -lh eval_results/ruler_32k_20260302_132117/
```

### 📊 预期结果

对于表现良好的长文本模型，在 32K 上下文长度下：

- **NIAH 任务**: 0.6-0.8 准确率
- **Passkey**: 0.7-0.9 准确率
- **VT/CWE/FWE**: 0.5-0.7 准确率
- **平均准确率**: 0.6-0.8

### ⚠️ 注意事项

1. **内存使用**: 32K 长度需要约 8-16GB GPU 内存
2. **速度**: 由于上下文较长，评测速度会比短文本慢
3. **SWAA 优势**: 滑动窗口注意力可以显著减少内存使用

### 📝 测试状态

- [x] 配置测试参数
- [x] 启动测试
- [ ] 等待测试完成 (预计 30-80 分钟)
- [ ] 分析结果
- [ ] 生成报告

---

**状态**: 🟡 测试运行中...

**后台任务 ID**: b3bf77d

**日志文件**: `/tmp/ruler_32k_test_v2.log`

**最后更新**: 2026-03-02 13:22
