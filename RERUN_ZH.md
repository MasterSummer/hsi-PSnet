# PSNet 修回实验与命令

本文档配套 `manuscript/PSNet_revision_20260928.docx`。论文使用已核验的归档结果；本次整理没有启动完整训练，也没有把补发数据对应的结果写成已完成。

## 先决定要解决什么

| 目标 | 入口 | 默认训练次数 | 何时需要 |
|---|---|---:|---|
| 补上四项机制消融及同协议 full 对照 | `controls` | 25 | 回应结构作用；目前缺真实结果 |
| 复核主要早期结论 | `priority` | 45 | 2 dpi、4 dpi、联合 2+4 dpi；PSNet、RGB18、plain multimodal；单种子 |
| 恢复缺失观测后重跑受影响任务 | `recovered` | 135 | 同三模型，2 dpi、联合、全日期；三种子 |
| 重建完整主表 | `full` | 720 | 12 模型 × 4 任务 × 3 种子 × 5 折；时间充足再做 |

已有三种子的主表可以直接用于这份修订稿，**不需要为交稿先重做 720 次训练**。若只能补一组，优先考虑 25 次结构消融；它仍是一个种子、日期混合的机制实验，不能单独证明症状前诊断。不同任务不能为了好看只选择最优结果。训练时间取决于设备和早停，本仓库不承诺完成时长。

## 环境

在 GPU 服务器的新目录使用 Python 3.11；CUDA 版 PyTorch 必须与服务器驱动匹配。若已有可用的项目环境，可直接复用。

```bash
git clone --branch codex/psnet-revision-package https://github.com/MasterSummer/hsi-PSnet.git
cd hsi-PSnet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r reviewer_experiments/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s reviewer_experiments/tests -v
```

将下面两个路径改成服务器上的实际位置。PT 必须是你自己已有、可信的包；旧包加载包含 Python 反序列化。

```bash
export PSNET_SPLIT_DIR=/absolute/path/to/split_4re
export PSNET_RGB_ROOT='/absolute/path/to/rgb_jpg/Run 1'
```

`PSNET_SPLIT_DIR` 下需要 `trainval.pt` 和 `cleantest.pt`。RGB 可为可解码 JPG/PNG；不能把原始 RGB888 改扩展名假装 JPG。RGB 文件名需要保留 `Plant69_Infected_Leaf3_Day2` 等样本标识。不同文件夹中不能有同名的两份 RGB。

## 时间有限时的运行命令

先打印计划，不训练：

```bash
python run_revision.py --preset controls \
  --split-dir "$PSNET_SPLIT_DIR" --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/controls_20260928
```

确认输出目录是这一次实验专用后，执行同一命令加 `--execute`：

```bash
python run_revision.py --preset controls \
  --split-dir "$PSNET_SPLIT_DIR" --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/controls_20260928 --device cuda --execute
```

主结论复核（45 次；不要与旧三种子平均值混为同一实验）：

```bash
python run_revision.py --preset priority \
  --split-dir "$PSNET_SPLIT_DIR" --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/priority_20260928 --device cuda --execute
```

预算允许时，把 `priority` 换为 `full` 并使用新输出目录。`--seeds 157,257,357` 可把单种子计划扩展到三种子。

所有预设明确设置 dim=64、depth=5、heads=4、dropout=0.5、RGB ImageNet 初始化、AdamW、lr=3e-4、weight_decay=1e-4、batch=8、最多 100 epoch、patience=12。不要直接使用通用 `reviewer_experiments.cli matrix` 的架构默认值来代替这些命令。每个外层训练集内再留出 20% 的植物做验证；波段选择和标准化仅使用 fitting plants。

## 补发 ZIP 怎么处理

它包含植物 69–72 在 2 dpi 的 8 组 MAT/hypercube。其中植物 69 的叶 3 在旧队列中已有，实际候选新增 7 条。不要只挑几株加入；若决定恢复，应使用一致的处理方式加入全部合格缺失观测。

当前核查发现，重叠样本的原始 MAT 均值约 0.74336，旧归档均值约 0.68243，最大逐波段均值差约 0.45835。因此，**把原始 MAT 直接转成 NPY 并拼进旧 PT 不成立**。原始数据导出不等于恢复了旧预处理。

```bash
python recover_revision_data.py export \
  --zip /absolute/path/to/wetransfer_hypercubes-mat-files_2026-09-21_1153.zip \
  --output outputs/recovered_raw
```

输出 HWC、float32 的 NPY、原始名义波长和每文件审计。没有暗校正、归一化或掩膜的推测性处理。名义波长也不是历史 PT 波段映射的证明。

在服务器从可信旧 PT 生成当前机器的路径清单：

```bash
python -m reviewer_experiments.cli ingest-split4re \
  --split-dir "$PSNET_SPLIT_DIR" --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/base_metadata.csv
```

只有找回原始预处理、把包括重叠样本在内的所有补发数据按同一流程处理后，再执行：

```bash
python recover_revision_data.py merge \
  --metadata outputs/base_metadata.csv \
  --recovered-dir /absolute/path/to/recovered_processed_npy \
  --layout HWC --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/recovered_cohort
```

合并要求每个重叠样本的完整张量在 `rtol=1e-5, atol=1e-6` 下匹配；没有重叠、数值不匹配或元数据不一致就中止。不会覆盖旧 PT，也不重复加入旧样本。不要为了过检查拟合一个缩放系数。单个重叠通过也不能证明每条新数据均正确，仍需保留真实处理脚本和记录。

若找不回旧预处理，可用统一流程从**全部原始数据**重新建队列，然后全部重训；不能只变换新样本。时间不够时继续提交归档队列版本，论文如实说明数据范围。

合并成功后：

```bash
python run_revision.py --preset recovered \
  --task-dir outputs/recovered_cohort/tasks \
  --output outputs/recovered_main --device cuda --execute
```

新队列会重新生成植物分层五折。仅恢复 2 dpi 时，受影响的是 2 dpi、联合 2+4 dpi 和全日期任务；4/6 dpi 不必因此重跑，前提是这些任务的数据、划分和处理没有变化。恢复后的 `full` 预设用于更新全部 12 模型主表；`recovered` 只覆盖三个关键模型，不能拿它宣称全部表格已更新。

## 重跑波段筛选和无需训练的统计

```bash
python run_band_experiments.py \
  --split-dir "$PSNET_SPLIT_DIR" --rgb-root "$PSNET_RGB_ROOT" \
  --output outputs/topk_20260928 --device cuda

python -m reviewer_experiments.spectral_reanalysis \
  --split-dir "$PSNET_SPLIT_DIR" --region whole \
  --output outputs/spectral_20260928
```

波段 runner 默认 full/top3/top10/top30，各五折，共 20 次训练。`top-k` 必须在每折 fitting plants 内选择，不能先用全体数据选波段。重跑前以 `--help` 查看输入选项；现有论文只使用计算波段索引，没有未验证的生理波长解释。

## 输出、续跑与检查

每个种子目录有 `experiment_manifest.json`、源码快照、任务快照、参数统计；每折有 `best.pt`、`run.json`、`predictions.csv`。全部五折齐全且样本 ID、植物 ID、标签和概率检查通过后，才生成 `evaluation/oof_metrics.csv` 和条件性植物 bootstrap 区间。一个种子的 SD 是缺失值，不是零。

保持代码、任务 CSV、原始数据及运行参数不变，可用相同命令续跑。已有预测但缺 checkpoint、结果损坏或旧输出来源不清时，应使用新的输出目录，不能把半套预测补成论文结果。当前 manifest 对代码和任务表做哈希，未对每个原始输入文件内容做哈希；请将输入数据保持只读并单独保留校验和。

每个种子的指标单独保存。全部完成后可用下面的命令核查各目录的协议和完整性，再计算同模型/任务的植物级 OOF 均值与样本 SD（ddof=1）：

```bash
python summarize_revision.py --runs-root outputs/controls_20260928 \
  --output outputs/controls_summary
```

换成相应的多种子输出目录即可汇总主表。程序报告实际种子数，不能把折当成独立实验重复。历史 plain multimodal 联合任务只完成两个种子，不应把缺失第三种子计为零。

本次本地验证包括单元测试、真实补发包导出、命令计划与 Word 渲染；没有本地旧 PT 全量数据，因此没有声称完成真实合并或 GPU 训练。
