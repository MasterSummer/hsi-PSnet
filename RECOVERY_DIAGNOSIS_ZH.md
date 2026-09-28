# 补齐七条观测：先定位重叠样本的处理差异

补发包包含 8 组 RGB/HSI，其中 `Plant69_Infected_Leaf3_Day2` 已在归档队列中，另外七条对应缺失观测。若全部满足一致的处理和配对要求，预计总观测数由 569 变为 576；2 dpi 由 185 条变为 192 条、由 93 株变为 96 株。重叠观测保留旧条目，不重复添加。

## 只读诊断

`diagnose_recovery.py` 从已失败作业的 `base_metadata.csv` 定位重叠样本，只加载其所在 PT，并与 `recovered_raw` 中的原始 NPY 比较。它会检查数值、零值分布、有限的轴翻转/reshape 假设。不会修改原数据、拟合缩放、生成补齐队列或启动训练。

```bash
(
  set -euo pipefail
  git fetch origin codex/psnet-revision-package
  PSNET_DIAG_SCRIPT=$(mktemp /tmp/psnet-diagnose.XXXXXX)
  trap 'rm -f "$PSNET_DIAG_SCRIPT"' EXIT
  git show FETCH_HEAD:diagnose_recovery.py > "$PSNET_DIAG_SCRIPT"
  python -u "$PSNET_DIAG_SCRIPT" \
    --job "$PWD/revision_jobs/job_20260928_160317_ermLu6" \
    --output "$PWD/revision_jobs/recovery_diagnosis_$(date +%Y%m%d_%H%M%S)_$$"
)
```

诊断会打印摘要和输出目录。`summary.json` 是摘要；`diagnosis.json` 是各假设的完整比较；`bands.csv` 是逐波段均值和旧数据零值比例；`overlap_pair.npz` 仅包含这一条旧张量及对应的新张量，便于后续分析，避免传输整套 PT。

`exact_matching_hypotheses` 表示某个排列假设能复现张量；`nonzero_preserving_hypotheses` 表示旧非零值与对应排列相符，可能涉及置零。二者均只是线索，不证明原处理流程，也不提供其余七条数据的有效掩膜。不能把一片叶子的掩膜复用于其他叶子。若旧非零值也不同，需要继续确认校正、裁切/重建或源样本身份。

## 确认处理后的合并

确认并记录处理流程后，将包括重叠样本在内的 8 份 NPY 用同一流程处理，写入新目录。保留原始输入和处理记录。用后台程序的 `--processed-dir` 替代 `--zip`，指定实际 CHW/HWC 排列，并保持 `--data-policy require-recovered` 和 `--preset paper-minimal`。仅在检查通过后加入七条新观测，重新生成植物独立五折，再完成合并日期分类、光谱分析、2 dpi 单独分类。

若仍无法恢复旧处理流程，另一条完整路径是获取全部原始样本并统一重建 576 条队列。仅有补发的 8 份原始数据不足以完成这一步。当前工具交付是诊断能力，不代表服务器上的数据已经补齐。
