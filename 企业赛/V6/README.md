# 企业赛 V6 · 工业漂移阶段收益验证

本版沿用 Qwen3.8-27B 驱动的 Harness 与三个工业预测任务，从企业赛 V5 正式运行 `80900572e91d7d33d6637088915dc5e6` 的逐阶段报告定位低谷。V5 官网为 23.034385 / 35，其中三项任务的最差阶段收益均为负。V5 网站回执、V4/V5 试运行对照及本地 V5 报告的对应关系在证据包 `official-reference.json`。实验室比较的基准是独立加载的冻结 V5。

## 重现实验

工作目录为服务器仓库根目录。训练与 SDK 镜像用 ai-plus 环境中的 CPython 3.12；报告生成使用安装 PyMuPDF 的环境。模型与 SDK 输入沿用项目的 `校赛/work/{A,B,C}` 和 `校赛/work/tools`，补充公开数据位于 `Supp_Datasets`。运行时仅依赖赛方提供的训练、验证、历史产物与 Qwen 控制接口。外部数据用于离线检验，外部设备标签、样本顺序分组和赛方报告的阶段收益均不进入线上预测。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
v=企业赛/V6
python -B "$v/qualify.py" register
python -B "$v/qualify.py" dev --workers 12
python -B "$v/qualify.py" freeze
python -B "$v/qualify.py" confirm --workers 12
python -B "$v/qualify.py" summary
python -B "$v/audit.py"
python -B "$v/scorecard.py"
for t in A B C; do
  python -B "$v/opening_study.py" "$t" v5
  python -B "$v/opening_study.py" "$t" v6
  python -B "$v/opening_history.py" "$t" v5
  python -B "$v/opening_history.py" "$t" v6
  python -B "$v/resource_bench.py" "$t"
  python -B "$v/serving_study.py" suite "$t"
done
python -B "$v/build_image.py"
python -B "$v/verify_image.py" > 企业赛/work/v6/image-proof-0.log
python -B "$v/bearing_partition_grid.py" --workers 8
python -B "$v/sdk_latency.py"
python -B "$v/sdk_latency.py" --repeat-session
python -B "$v/resource_summary.py"
python -B -m unittest discover -s "$v" -p 'test*.py'
```

预留顺序在生产代码和九个公开输入冻结后方可用于确认。程序自动核验文件摘要。开发可见的旧划分为 0–13，主比较报告其中 7 个开发顺序和冻结后执行的 6 个预留编号（B 的 3 个窗口与开发期重复，另以全部 27 种窗口组合复查）。六种协议共用官方公开样本，两版共享每个顺序的训练释放、静态参照、选型样本、工况和审计目标。完整首批、整组排除在选型之外的首轮检查、跨域补充数据分别报告。单条测试包含完整 SDK 入口的 JSON 解析、进程往返、反序列化与预测。

## 选型与依据

A 在完整数据支持下按响应幅度分层选择每类物理核原型，小样本尾部专家按最弱类的可见样本数调节骨干证据。B 依据末端样本覆盖率，在训练片段间校准残差方向；删失严重时向骨干收缩。C 在工况归一化传感器上计算二次投影，沿用旧分支初始化，拟合对最弱寿命组和双侧 NASA 风险有约束的读出层。分任务报告给出公式、替代路线、补充设备和失效条件。

对照中还比较了置信区间分层的尾部风险排序、细化静态混合权重、按剩余寿命稀有度或截断残差重采样。A 的细化权重在完整首批整组留出上压低低谷，B 残差重采样在第二组开发顺序未重复抬高低谷。B 的额外局部支持回退在 14 个开发顺序的三工况低谷上全部退化；详见 bearing_floor_study.py。生产版保留 V5 的候选混合与均匀回放，不引入额外的在线控制模型调用。该选型由开发顺序完成。

相关原始资料包括 [气体识别的浓度与漂移评测](https://www.sciencedirect.com/science/article/pii/S0026265X26032674)、[电子鼻复现项目](https://github.com/SuperAIfuture/E-nose/commit/51ceaf593d35f9b97b9b688a5185d37877ad5f4b)、[XJTU-SY 发布者仓库](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)、[分阶段轴承寿命目标的研究](https://arxiv.org/abs/2607.28115)、[C-MAPSS 域偏移中的精度与风险](https://arxiv.org/abs/2611.28759) 和 [CruiseBench](https://github.com/NostalgiaJohn/CruiseBench)。这些研究用于形成可验证的假设，项目中的数值来自本地对照。

## 交付与分数

服务器 `企业赛/work/v6/delivery` 包含冻结镜像、源码、公开证据、技术报告和验证汇总。证据包中 `freeze.json` 固定生产来源与数据摘要，`audit.json` 保存逐对重算，`scorecard.json` 给出六种协议的阶段均值、低谷、回撤、恢复和 C 的 NASA。目录 `gas`、`bearing`、`engine` 保留路线消融。预测容器不加载这些实验报告。

官网隐藏流与本地公开分组没有已知一一映射。A 的八个时间顺序片段全部来自一个公开验证批次，因此另按日期留出外部设备。B 的三个工况是转速、载荷分组。C 的四个 FD 子集和整机留出用于检验跨条件迁移。实际 V6 五项正式成绩需平台评定；本次只运行本地验证。

## 冻结后的本地结果

按 V5 官网的十轮阶段访问顺序，在六个预留划分上，A 的最低阶段 7.79% → 7.79%，B −9.82% → −0.75%，C 2.82% → 3.88%。阶段平均分别为 A 33.90% → 34.36%、B 3.38% → 7.09%、C 9.36% → 10.31%。B 的 27 种不同窗口中 25 种低谷改善；C 的六种协议最低阶段均值均改善，另外五种的阶段均值略降。A 的主协议低谷持平，完整官方起点协议 11.11% → 13.96%。完整比较、实际训练资源及官网编号对应限制见交付目录的验证汇总与分项报告。
