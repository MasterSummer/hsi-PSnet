# PSNet 五折、光谱分析与 2 dpi 单独分类

当前精简流程使用后台入口的 `paper-minimal`，按以下顺序运行：

1. 仅 PSNet，全部日期合并为一个接种/mock 二分类任务，种子 157，植物独立五折。
2. 按论文方法重做植物级 whole-cube 光谱统计，分 2、4、6 dpi 比较当天两组，输出 Welch 检验、Hedges' g 和 BH-FDR。使用与分类相同的数据清单，不训练波段筛选模型。
3. 仅 PSNet，只用 2 dpi 的接种与 mock 数据，种子 157，单独植物独立五折。

总计 10 次训练。组成表继续输出。单种子结果不能替代论文三种子均值及 SD。此流程通过 `run_unattended_revision.py --preset paper-minimal` 执行；单独的 `run_revision.py` 仅生成分类训练命令，不包含光谱步骤。

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
    --preset paper-minimal \
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

完成后 `EXIT_CODE.txt` 应为 0，状态应为 `completed`。全日期任务汇总在 `results/summary/all_dpi/summary.csv`，2 dpi 在 `results/summary/dpi_2/summary.csv`；对应种子目录下的 `evaluation/` 含植物级 OOF 评估。光谱输出在 `results/spectral/`，其中 `summary_by_dpi.csv` 是各日期汇总，`within_dpi_band_statistics.csv` 是逐波段统计，`plant_mean_spectra.csv` 是植物均值光谱。若状态为 `failed`，读取 `results/FAILED.txt`；启动成功不代表训练完成。
