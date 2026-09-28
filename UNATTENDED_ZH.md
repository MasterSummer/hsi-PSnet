# 服务器后台运行（无需重新打包 PT）

旧 `trainval.pt`、`cleantest.pt` 保留；训练读取清单中的 `ptbundle:` 条目。补发 MAT 可以导出 NPY；同一清单可混合引用旧 PT 和新 NPY，**无需重新生成整套 PT**。转换文件格式不能消除旧预处理与原始 MAT 之间的差异，恢复数据仍须通过重叠检查。

## 现在直接执行

先进入服务器原来的 Git 仓库目录，激活原来能训练的 Python 环境，并处于可使用 GPU 的会话。下面假定当前目录下有 `split_4re/trainval.pt`、`split_4re/cleantest.pt`，且旧 PT 记录的 RGB 路径仍有效。

```bash
(
  set -euo pipefail
  test -f split_4re/trainval.pt
  test -f split_4re/cleantest.pt
  git fetch origin codex/psnet-revision-package
  PSNET_LAUNCH_FILE=$(mktemp /tmp/psnet-launch.XXXXXX)
  git show FETCH_HEAD:launch_revision.sh > "$PSNET_LAUNCH_FILE"
  PSNET_REVISION_REF=FETCH_HEAD bash "$PSNET_LAUNCH_FILE" \
    --split-dir "$PWD/split_4re" --data-policy archived --device cuda:0
)
```

这里 fetch 最新代码，然后固定该提交的独立代码副本来训练，不切换当前分支，也不覆盖原仓库未提交的修改。每次启动使用新目录。**这段命令只训练旧队列的 controls：5 个模型 × 5 折 × 1 个种子，共 25 次训练。它不会自动增加 45 或 720 次实验。**

`launch_revision.sh` 使用当前 `python`；如训练环境的解释器另有路径，启动前设置 `export PSNET_PYTHON=/absolute/path/to/python`。不会自动升级或改动现有 PyTorch/CUDA 环境。若缺依赖，日志会指出具体 import 错误；在正确环境安装对应依赖后重新启动。

输出目录会打印到终端，形如 `revision_jobs/job_日期_时间_随机字符/`。随后可以断开 SSH；程序自动完成：

1. 校验两个旧 PT、逐个 RGB 解码、HSI 的维度/有限值、CUDA 与 ResNet34 预训练权重。
2. 固定植物五折划分，记录代码版本、原始输入 SHA-256 和实际队列。
3. 顺序完成 25 次训练，使用原定早停；不会在失败时自动降低 batch 或更换协议。
4. 核验五折预测、生成单种子指标和汇总，再检查原始输入是否被修改。
5. 成功写 `results/DONE.txt`、`results/RESULTS.md`；失败写 `results/FAILED.txt`。启动器总会在正常退出或可捕获错误后写 `EXIT_CODE.txt`。

`nohup` 可应对 SSH 断开，不能保证服务器重启、调度器取消作业或强制杀进程后继续运行。GPU 作业若受调度器管理，应在分配的作业中运行，并遵守其时限。

## 之后一次性查看结果

无需持续看日志。回到同一仓库，执行：

```bash
PSNET_LAST_JOB=$(ls -dt "$PWD"/revision_jobs/job_* | head -n 1)
cat "$PSNET_LAST_JOB/EXIT_CODE.txt"
cat "$PSNET_LAST_JOB/results/STATUS.json"
cat "$PSNET_LAST_JOB/results/RESULTS.md"
```

`EXIT_CODE.txt` 为 `0` 且状态为 `completed` 才代表完成；没有退出文件时可能仍在运行，也可能被系统强制中止，结合 PID 和调度器状态判断。失败原因在 `results/FAILED.txt` 或 `job.log`。训练汇总为 `results/summary/summary.csv`。

## 若旧 PT 内的 RGB 路径失效

PT 内存的是 RGB 文件路径，不一定包含 RGB 像素。需要原 RGB 文件。给启动命令追加：

```bash
--rgb-root '/absolute/path/to/rgb_jpg/Run 1'
```

该目录及子目录需包含可解码 JPG/PNG，文件名保留植物/叶/日期信息。缺 RGB 时程序会停在预检，不会假装多模态实验已完成。

## ZIP 传到服务器之后

本地 Mac 的 `/Users/...` 不是服务器路径。先把 ZIP 传到服务器任意确定位置；启动参数改成：

```bash
(
  set -euo pipefail
  git fetch origin codex/psnet-revision-package
  PSNET_LAUNCH_FILE=$(mktemp /tmp/psnet-launch.XXXXXX)
  git show FETCH_HEAD:launch_revision.sh > "$PSNET_LAUNCH_FILE"
  PSNET_REVISION_REF=FETCH_HEAD bash "$PSNET_LAUNCH_FILE" \
    --split-dir "$PWD/split_4re" \
    --rgb-root '/absolute/path/to/rgb_jpg/Run 1' \
    --zip '/absolute/path/to/recovered.zip' \
    --data-policy prefer-recovered --device cuda:0
)
```

将 RGB 和 ZIP 路径改为服务器实际位置；不要把“现在直接执行”和本例同时启动，以免重复占用 GPU。

- `prefer-recovered`：自动导出、检查；成功合并才用新队列。失败自动使用旧队列训练，明确记录原因。不会把新增结果当成已经纳入。
- `require-recovered`：恢复失败就停止，不训练旧队列。
- `archived`：使用旧队列，与上面“现在直接执行”的命令一致。

当前已有均值核查提示原始 MAT 与旧归档可能不一致，所以**不要期待换成 NPY 就能直接加入**。若找回原始预处理，用 `--processed-dir /path/to/processed_npy --layout HWC` 代替 `--zip`；不得通过人为拟合缩放只为通过检查。新队列实验需要连同其 full 对照一起报告，不能直接与旧队列分数混合。

后台程序不是完整重训矩阵，也不自动续跑失败作业。后续需要续跑已完成的折时，使用同一代码快照里的 `run_revision.py` 对 `results/experiments` 执行原协议；先处理日志中的错误，不能忽略损坏或缺失的 checkpoint。更多矩阵和恢复说明见 [RERUN_ZH.md](RERUN_ZH.md)。
