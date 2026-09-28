# 2026 年 9 月 28 日论文修改说明

本稿基于 v6 counts-verified Word 修改，保留作者、主体内容、硬件和叶片图，以及已经核验的分类结果。它是可供共同作者审阅的完整研究叙述，不代表所有审稿问题都已由新实验解决。

## 实质修改

- 摘要和结论收窄为单次实验中的接种组区分，避免把处理标签等同于逐株确认感染。
- 方法改为实际 binary reimplementation：三层 Haar、先空间均值、313 系数截断、64 学习通道、512 patch tokens、4 heads、5 layers，以及 CAF 的准确输入位置。
- 训练表区分归档可核实的设置与新的重跑协议，删除无依据的 Adam、学习率 0.001、16 heads 和 warm-up 声明。
- 补入完成的整立方体光谱分析、Figure 7 和 Table 12；明确没有验证叶片掩膜及历史波长映射。
- 用完成的 top-k 实验替换全是 Pending 的 Table 10；其 full 对照与历史 pooled full 独立，不混用。
- 补全基线参数量，使用已重新计算的完整五折预测填写 Table S5。
- 保留真实数据范围、缺失观察、重复植物 ID 映射和采集顺序混杂，不以补发文件代替补跑结果。
- 成本改为 £489.20 部件小计，不再承诺缺乏打印成本依据的整机总价。
- 补全 Furzer 文献的书目信息。来源为 [University of Bath 原始研究条目](https://researchportal.bath.ac.uk/en/publications/an-improved-assembly-of-the-albugo-candida-ac2v-genome-reveals-th/)，DOI 10.1094/MPMI-04-21-0075-R；本次没有独立复核原实验是否逐步执行了该文染色方案。

## 证据范围

`evidence/verified_archived_metrics.csv`：126 个完整模型/任务/种子组合，630 份折预测，252 个 BA/AUROC 值与归档参考一致；同时检查样本集合、标签及植物折划分。

`evidence/band_comparison.csv` 与 `paired_differences_vs_full.csv`：独立的一种子、五折 top-k 实验及固定预测的配对 bootstrap。所有 BA 差值区间跨零。

`evidence/spectral_summary.csv`：每个 dpi 内独立 BH 校正的汇总。不要使用跨 dpi 校正列代替论文所述分析。

重跑工具经过本地测试，但真实 GPU 重训未在本次执行。源 v6 文件不重复公开打包；`build_revision.py --source /path/to/v6.docx` 可从作者保留的原稿重建。`revision.diff` 与 `revision_diff.html` 是实际文本差异，包括表格和公式文字；它们不表达图片像素或 Word 样式差异。

## 提交前确实仍需作者确认

1. 全体作者的利益冲突声明、CRediT 分工以及 AI 使用披露；助手不能代替共同作者作出声明。
2. 染色的样本数、mock 对照及逐株阳性记录，如能找回可增强论证；当前稿已明确未建立这项参考标准。
3. 原始植物标记记录、校准帧时序和 upstream preprocessing，如没有记录，保留本文的限制说明。
4. 原审稿人要求的单独机制消融、归因、独立批次等，现有材料不能称为已全部满足。新的实验完成后应更新相关结果和逐点回复，而不是只删除限制。

本次没有发邮件，没有运行完整训练，也没有把原始数据或邮件正文纳入 Git 提交。
