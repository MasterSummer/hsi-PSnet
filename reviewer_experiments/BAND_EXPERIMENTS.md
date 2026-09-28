# 光谱重分析与部分光谱输入实验

两个入口都直接支持已有 `split_4re/trainval.pt` 和 `cleantest.pt`。统计无需 RGB 原图；多模态训练需要原 RGB 图像。PT 数据须为本项目的可信数据包，即 `(rgb_path, hsi_tensor, class_label)` 记录列表。

两个入口会联合检查所有 PT 记录，自动识别两种标签编码：`mock=0，2/4/6 dpi接种=1/2/3`，或旧编码 `2/4/6 dpi接种=0/1/2，mock=3`。只有整个输入集合与其中一种编码一致才继续；混用或未知编码会报错并列出对应计数。处理组、日期、植株身份由文件名解析；二分类标签由处理组生成，不能直接使用 `stored_label > 0`。统计输出 `pt_label_audit.json`，训练导入的 `metadata.summary.json` 保存编码与计数，逐条元数据保留原标签。无需改写 PT 文件。

## 1. 安装

在仓库根目录，用现有 PyTorch 环境：

```bash
python -m pip install -r reviewer_experiments/requirements.txt
```

## 2. 重新做光谱统计

全部313个输入维度，同日期内比较接种和mock组：

```bash
python -m reviewer_experiments.spectral_reanalysis \
  --split-dir /data/split_4re \
  --expected-bands 313 \
  --output /data/results/spectral_full
```

仅分析指定维度（例：原始索引0–103）：

```bash
python -m reviewer_experiments.spectral_reanalysis \
  --split-dir /data/split_4re \
  --expected-bands 313 --bands '0:104' \
  --output /data/results/spectral_first
```

也可复用刚提取的植株级光谱，避免反复读取大型PT文件：

```bash
python -m reviewer_experiments.spectral_reanalysis \
  --plant-csv /data/results/spectral_full/plant_mean_spectra.csv \
  --bands '104:208' --output /data/results/spectral_middle
```

统计单位是植株：先平均同叶片的重复采集，再平均同株同日期的叶片。输出每个原始波段的均值、点位95% CI、Welch检验、Hedges’ g及每日期内BH-FDR；全日期检验族FDR另列敏感性结果。输出波段编号保持原始索引，不重编号为0起点。统计表另有 `absolute_mean_difference`、`difference_rank` 和 `descriptive_top3/top10/top30`：这是各日期的描述性排名，不供分类实验直接选波段。

默认是整幅立方体空间平均。如有已对齐的叶片二值掩膜，可加 `--region leaf --mask-root /data/masks`；`--region background` 使用背景。掩膜为与原RGB文件名同stem的 `.npy`，形状为HSI的H×W，1为叶片、0为背景。不自动生成掩膜。CSV复算无法补做空间区域提取。

真实波长表可用 `--wavelengths wavelengths.csv` 提供，字段为 `band,wavelength_nm`。`--calibrated-only` 只保留明确映射的输入位置。不可从范围端点推算313个波长，也不可默认前273个位置就是标定集合。本入口不自动计算红边指标。

## 3. 完整光谱与部分维度训练对照

```bash
python run_band_experiments.py \
  --split-dir /data/split_4re \
  --rgb-root /data/RGB \
  --output /data/results/band_experiments \
  --device cuda
```

默认 **PSNet、全部日期合并、一个种子157、五折按植株交叉验证**，比较四组输入，共20次训练：

| 名称 | 313维输入时的原始索引 | 维度数 |
|---|---|---:|
| full | 0–312 | 313 |
| top3 | 每折拟合集平均光谱差异最大的3个原始索引 | 3 |
| top10 | 每折拟合集平均光谱差异最大的10个原始索引 | 10 |
| top30 | 每折拟合集平均光谱差异最大的30个原始索引 | 30 |

按两组平均光谱之差的绝对值排序，选择单个波段维度，并非连续区间。每折先划分拟合/早停验证/外层测试植株，**只用拟合植株**选波段；验证、测试均不参与。先计算每个立方体的空间均值，再平均同株所有可用日期/叶片观测，最后对两组分别按植株等权平均。观测缺失会改变株内日期权重。按原始数值差异排名，不按p值或标准化效应量；并列时优先较小索引。选中后按原始光谱顺序输入模型。每折所选维度可以不同，同折Top3包含于Top10和Top30。可通过 `--top-k 3,10,30` 显式指定规模。完整输入参照也重新训练，以保持协议一致。改变输入维度会改变入口层和token投影参数量，参数数值随每组输出记录；不是等参数容量实验。

自定义区域、组合或间隔采样：

```bash
python run_band_experiments.py \
  --split-dir /data/split_4re --rgb-root /data/RGB \
  --band-set full=all \
  --band-set selected=0:100,200:313 \
  --output /data/results/band_custom --device cuda
```

以上是2组×5折=10次训练。`--band-set sparse=0:313:2` 可每隔一维采样。索引从0开始，冒号范围右端不包含；重复、倒序、越界或空集合会报错。非连续选择会把选中维度压紧作为模型输入，并保留原始索引映射，需在方法中说明。

如果还要对照HSI单模态，加 `--models psnet_full,hsi_transformer`；训练量相应翻倍。其他模型名见 `reviewer_experiments/models.py`。该训练数据管道仍读取配对RGB图像，即使选择HSI-only分类器，也需可解析的RGB文件。

已有上一轮任务表时，可用 `--task-dir /data/reviewer_results/tasks` 替代 `--split-dir` 和 `--rgb-root`，复用 `all_dpi.csv` 的精确划分。需要表内的RGB路径和PT URI在当前服务器有效。默认不按日期拆分训练；通过 `--tasks` 可另行指定其他任务。

首次可加 `--prepare-only`，只检查输入、生成划分和记录参数，不启动训练；之后去掉该参数续跑同一命令。配置、源码和任务表一致时，会跳过已完成的折；未完成折从头训练。修改波段、模型、任务或训练配置时必须换输出目录。

## 4. 输出和解释

- 光谱：`summary_by_dpi.csv`、`within_dpi_band_statistics.csv`、`plant_mean_spectra.csv`、`sample_qc.csv`（PT模式）、`analysis_notes.json`。
- 模型对照总表：`band_comparison.csv`。
- 每波段组：`full/`、`top3/`、`top10/`、`top30/` 或自定义名称。内部有 `evaluation/oof_metrics.csv`、逐折预测、最佳checkpoint、训练/验证/测试观测表、源码快照和 `parameter_counts.csv`。
- `band_experiment_manifest.json` 记录选择规则、Top-K规模、源码哈希和共同任务表哈希；每折 `band_ranking.csv` 保存组均值、绝对差异、排名、是否选中；checkpoint的 `band_stats` 与 `config` 保存所选索引和原始输入维度。

训练、验证、测试和checkpoint推理使用同一波段选择；每折只在内层拟合植物上估计所选波段均值/SD。每株所有日期与叶片归入同一折。先合并五折OOF预测，再对每株所有可用观测平均概率，计算最终指标和植物bootstrap CI；单种子SD留空。由于全日期结果包含6 dpi，它用于输入维度比较，不表示无症状早期诊断。

**不要根据全数据光谱显著性挑选“最好波段”后，把同一数据上的分类性能当作无偏测试结果。** 本入口默认在每折拟合集内重新选Top-K，并保留独立测试；全数据描述统计不参与训练选波段。固定索引子集仍可通过 `--band-set` 指定。

PT训练导入沿用已有Run 1 mock源编号取模规则以合并跨日期身份，需要真实采集记录支持。统计入口保留源编号、分别在每日期内分析，不推断跨日期mock身份。元数据/源码哈希不等于原始PT内容哈希。

## 5. 测试

```bash
python -m unittest discover -s reviewer_experiments/tests -v
```

包含原始索引解析、子集标准化、checkpoint推理一致性、遮挡索引映射、PT光谱提取、五折完整性与跨日期植物隔离。真实数据训练须在数据所在服务器执行，代码测试不能代替实验结果。
