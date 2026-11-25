# 智训引擎 V2

本目录是校赛 V2。V1 的运行代码仍在同级 `../app`，提交镜像和文档由服务器 `../work/delivery` 留档。V2 输出到 `../work/v2/delivery`。两版共用赛方原始 MLP 读写器、少量通用数值函数与 SDK 1.0.4。V2 的作品文档内嵌可复现源码和演示。

## 开发与验收

以下命令从服务器的 `校赛` 目录执行。依赖、赛方数据、crane、固定摘要的基础镜像和 NumPy 2.2.6 wheel 的准备步骤见 [V1 构建说明](../README.md)。开发环境为 Python 3.12 和 NumPy。部署镜像使用 Linux/amd64、CPU 单线程、无外部训练依赖。

```sh
cd "/opt/data/private/mzx/mzx/AI +/校赛"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai-plus
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B prepare.py
python -B V2/bench_gas.py
python -B V2/bench_bearing.py
python -B V2/bench_engine.py
python -B V2/bench_engine_confirm.py
python -B V2/bench_external.py
PYTHONPATH=work/sdk python -B V2/replay.py
python -B V2/test_loop.py
python -B build_image.py --version v2
python -B verify_image.py --version v2
```

气体和发动机的 V1 十轮对照读取 `work/validation10` 中原始记录。在全新环境中可用 `python -B rehearse.py control --rounds 10 --name validation10` 创建。开发侧 `PYTHONPATH=work/sdk python -B V2/replay.py --live` 对每项做一次真实 Qwen 首轮验证，保存至 `work/v2/live`，调用计入 PyTRIO 账号；已有回执时拒绝覆盖。此前的单调用对照使用 20 个输出 token 上限。最终镜像每轮通过赛方网关调用两次真实 27B，规划和评审分别最多输出 56 和 32 个 token。输出报告使用 M2Gaze 的 PyMuPDF 环境及 FFmpeg。

## 面向客观分的部署设计

每轮恢复赛方给出的实际父检查点和 Adam 状态。按公开训练与验证输入的均值偏移识别漂移最明显的两个特征，报告父模型及最差组表现，按工具统计正收益和实际采纳次数。27B 第一份回执给出可检验的归因假设、训练工具和全量或半量学习率。程序解析模型发出的白名单指令，真实执行重拟合任务分支、保留已有分支续训或只训练 MLP 中的一项。每轮只训练并向宿主上传一个候选。候选须改善主指标；对于公开验证中最差类别召回，允许至多 0.01 的波动，对于最差设备或子集 RMSE，允许相对 2% 的波动。候选未达到部署条件时，从最近两轮的获选记录中至多取回一个不同于父模型、且被宿主声明可见的 V2 历史产物，使用当前公开投影重新评估。27B 第二份回执根据候选与历史产物的实测收益选择采用候选、保留父模型或回滚。程序拒绝不可见、未复核或未过保护条件的部署。决策、归因、学习率、具体执行步数、两份真实回执均进入训练记录。容差是本作品的部署选择。赛方未公布由这些量兑换资源分和恢复分的具体公式。

训练输入仅来自本轮公开训练行。训练日志中的设备号、子集和顺序信息仅用于分组诊断与留出。B 有校正头时，再在当前公开集测一次关闭它后的主指标和最差组变化，供第一步规划归因。预测使用单条授权输入和自己的检查点，逐条刷新 SDK 输出。开发扩展实验与镜像构建走不同入口；构建器只收录五个 V2 运行模块、两个共用运行模块、SDK 及其运行依赖。

固定公开回放使用整份公开验证集合选型。轮换压力测试在每个公开组内每五条抽取一条用于审计，其他样本按顺序分四段，按 0,0,1,1,2,2,3,3,0,1 的顺序轮换作为选型验证。每轮再在同一段的审计行测量相对静态起点的改善。此试验重放验证分布变化，训练投影固定。项目已经在开发时接触过这份公开验证数据。单独的交叉验证由各任务脚本从训练数据划分，按折重新训练基础模型。赛方的实际阶段和隐藏评分由平台独立生成。

## 研究依据与来源

- A 在不同传感器之间比较同次测量的相对响应，对绝对响应和相对形状采用不同程度的协方差收缩，再与 MLP 分类结果结合。公开源数据来自 16 个传感器、六种气体，按采集批次存在漂移。[UCI 原始数据说明](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) · [2026 年跨批次评测研究](https://doi.org/10.3390/informatics13010015)
- B 将振动强度与剩余寿命先验的可信程度关联。使用训练对学习对数特征残差，训练集的高振动分位区间逐渐降低总体寿命先验权重。XJTU-SY 原始实验有三种工况，跨设备检验保留整根轴承。[数据作者说明](https://biaowang.tech/xjtu-sy-bearing-datasets/) · [2026 年双通道振动健康指标研究](https://www.mdpi.com/2075-1702/14/10/1119)
- C 依据当前运行设置消除传感器的工况差异，再从传感器中学习对当前 MLP 寿命估计的非线性修正。[RUL Adapt 开源工况适配](https://krokotsch.eu/rul-adapt/examples/conditional/) · [工况标准化研究](https://arxiv.org/abs/2611.28759)
- 自动试验借鉴 AIDE 的候选验证，以及 MLE-Ideator 对实验设计与执行的区分。文献提出待测假设，各模块参数以本项目实测为依据。模块由队伍结合 AI 辅助编写；基础训练器和 SDK 来自组委会，运行时保留 Python、Debian、NumPy 的分发说明。[AIDE 论文](https://arxiv.org/abs/2502.13138) · [MLE-Ideator，EACL 2026](https://aclanthology.org/2026.eacl-short.32/)

外部电子鼻试验读取作者提供的特征表（36 个采集日），缓存由 SHA-256 校验。前 12 日训练、后 24 日检验，参数沿用 A 的判别结构。62 个传感器和 3 个标签使用各自的输入输出。外部数据及权重保存在服务器实验区。[原始数据与许可](https://zenodo.org/records/15681119)

## 最终双阶段公开实测

从校赛目录运行 `PYTHONPATH=work/sdk python -B V2/agent_replay.py A`；B 和 C 替换最后一个参数。此入口产生真实费用。每项执行一个固定公开首轮和十轮轮换压力测试，单轮各取一份真实规划及裁决。运行记录和回执分别保存在 `work/v2/agent`、`work/v2/agent_artifacts`，可从已完成轮次恢复。该入口在 BJTU 执行生产 train 及实际训练工具，并从落盘的权重独立重算公开分数。宿主替身不含赛方官方系统提示、排队、分块传输或隐藏打分。纯 CPU 对照及此前单调用对照在 `V2/control.py` 和 `V2/replay.py`。最终打包命令为 `/root/miniconda3/envs/M2Gaze/bin/python -B V2/deliver.py`。

[各任务结果、成本与失败实验](实测结果.md)。运行 `PYTHONPATH=work/sdk python -B V2/bench_resources.py` 可复核首轮配对 CPU 实验。
