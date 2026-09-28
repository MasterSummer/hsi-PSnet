# 主实验与植物级数据组成表

`main` 只运行 8 个主对比模型，不包含结构消融：PSNet、plain multimodal、simple multimodal、RGB ResNet-18、RGB ResNet-34、HSI Transformer、1D spectral CNN、compact 3D CNN。
四个任务为 2 dpi、4 dpi、6 dpi、联合 2+4 dpi；三个种子为 157、257、357；每个任务做植物独立五折，共 480 次训练。

在服务器原有 Git 仓库目录、已激活的 GPU 训练环境中执行。当前目录应包含 `split_4re`，其中已有两个 PT 和两个补发 ZIP；旧 PT 中的 RGB 路径应仍有效。

```bash
(
  set -euo pipefail
  PSNET_SPLIT="$PWD/split_4re"
  test -f "$PSNET_SPLIT/trainval.pt"
  test -f "$PSNET_SPLIT/cleantest.pt"
  test -f "$PSNET_SPLIT/新数据.zip"
  test -f "$PSNET_SPLIT/wetransfer_hypercubes-mat-files_2026-09-21_1153.zip"
  git fetch origin codex/psnet-revision-package
  PSNET_COMMIT=$(git rev-parse FETCH_HEAD)
  PSNET_LAUNCH_FILE=$(mktemp /tmp/psnet-main.XXXXXX)
  trap 'rm -f "$PSNET_LAUNCH_FILE"' EXIT
  git show "$PSNET_COMMIT:launch_revision.sh" > "$PSNET_LAUNCH_FILE"
  PSNET_REVISION_REF="$PSNET_COMMIT" bash "$PSNET_LAUNCH_FILE" \
    --preset main \
    --split-dir "$PSNET_SPLIT" \
    --zip "$PSNET_SPLIT/wetransfer_hypercubes-mat-files_2026-09-21_1153.zip" \
    --rgb-zip "$PSNET_SPLIT/新数据.zip" \
    --data-policy require-recovered \
    --device cuda:0
)
```

这里通过 `git fetch` 下载分支，再从确定的提交生成独立代码快照。无需切换当前分支或覆盖未提交文件。`nohup` 后台运行，输出目录由启动器打印。

若旧 RGB 路径已失效，在启动参数中追加 `--rgb-root '/服务器上原RGB目录'`。补发的 8 张 RGB 不能替代整套旧 RGB。

程序先输出原队列的 `results/composition_archived/`，再导出 ZIP、校验重叠样本，通过后生成本次实际队列的 `results/composition_used/`，随后预检 GPU 并训练。组成表、缺失观测清单、英文图注均按对应清单计算；若补齐七条观测，实际队列图注不会继续声称缺失七条。

已有归档核查提示原始 MAT 与旧数据预处理可能不同。`require-recovered` 在完整张量不匹配时停止，不回退到旧队列训练。原始导出不等于完成旧预处理；若失败，检查 `results/recovered_cohort/overlap_audit.json` 和 `results/FAILED.txt`，找回原预处理后用 `--processed-dir /实际处理后NPY目录 --layout HWC` 替代 `--zip`。

查看启动器打印的作业目录：

```bash
PSNET_JOB='/启动器打印的作业目录'
tail -n 80 "$PSNET_JOB/job.log"
cat "$PSNET_JOB/results/STATUS.json"
```

完成后 `EXIT_CODE.txt` 应为 0，状态应为 `completed`。主实验汇总在 `results/summary/summary.csv`，逐种子指标在 `results/summary/metrics_by_seed.csv`。若状态为 `failed`，读取 `results/FAILED.txt`；启动成功不代表训练完成。
