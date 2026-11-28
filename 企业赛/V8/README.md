# 企业赛 V8

冻结运行使用 PyTRiO Qwen/Qwen3.8-27B、赛方 SDK 1.0.4 和当前父工件。工业任务头仅从本轮已释放的训练集学习，模型依据可见验证结果复核候选与父模型。部署可复用历史工件，三类任务均支持单条推理。

| 任务 | 生产模块 | 公开检验 |
|---|---|---|
| A | gas.py、prototype.py | 类内动力学聚类，训练组间均值惩罚，稀有组温度；八个顺序组、全年电子鼻与 DS-PS |
| B | bearing.py | 设备终点与截尾分开，低能量侧保留设备内相对结构；XJTU 整机、FEMTO、KAIST 与十轮回放 |
| C | engine.py | 发布轨迹成熟度控制参数平均，稳定轮由分组误差和双侧 NASA 风险约束校准；公开整机及 N-CMAPSS DS02 |

三项研究的候选、消融、失效条件与文献分别见[气体实验](气体实验.md)、[轴承实验](轴承实验.md)、[发动机实验](发动机实验.md)。图像构建时只写入十个运行模块与赛方接口。报告、校验清单及图像保存在服务器 `企业赛/work/v8/delivery`。

## 复算

在本目录及服务器项目环境执行下列命令。运行前将 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`BASELINE_CPU_THREADS` 均设为 1。普通评估器及制镜使用 ai-plus 环境 CPython 3.12，中文 PDF 使用服务器 M2Gaze 环境的 PyMuPDF。

```bash
python -B qualify.py register
python -B qualify.py freeze
python -B qualify.py dev
python -B qualify.py confirm
python -B qualify.py summary
python -B audit.py
python -B -m unittest discover
python -B build_image.py
python -B verify_image.py
python -B artifact_audit.py
```

协议在模型冻结前登记。测试集包含 8 个预留样本划分、4 套公开协议、3 个任务和 3 个版本。另有 2 个开发划分。独立复算把每条流对应的静态起点、样本摘要和逐轮指标一一配对。真实模型回放命令为 `python -B live_replay.py B v8`，仅在发出新的请求时使用 PyTRiO 额度。原确认划分为 201，轴承另外以 203、207 做真实复核。这两个诊断在观察 201 后增加。

V5 的正式五项客观成绩来自平台报告。V7、V8 试运行返回隐藏终态指标。公开回放采用公开样本、推定阶段和可见的发布比例。各协议的平均和最低收益、遗忘、恢复及成本在技术报告分别给出。

## 研究实现与对照

气体以跨日期复现和组间分布变化为依据；轴承结合设备完整度及健康退化阶段；发动机比较工况归一化与非对称代价，并借鉴优化轨迹参数平均。论文用于提出候选，生产方案由项目内配对实验决定。

- 气体跨日复现与代码 [E-nose](https://github.com/SuperAIfuture/E-nose)
- 轴承自适应退化指标 [Machines 2026](https://www.mdpi.com/2075-1702/14/10/1119)，公开对照实现 [rul-datasets](https://github.com/tilman151/rul-datasets)
- 发动机工况与代价对照 [Adeel 2026](https://arxiv.org/abs/2611.28759)，参数平均 [Izmailov 等](https://proceedings.mlr.press/r16/izmailov18a.html)

## 平台额度

每次试运行或正式评测均须由队伍明确指定版本、类型和次数。现有运行状态可用只读入口 `python -B platform_watch.py` 查看；该入口要求已有运行回执。
