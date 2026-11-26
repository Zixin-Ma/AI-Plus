# 企业赛 V1 · 工业漂移下的模型续训
本阶段源码位于服务器 AI +/企业赛/V1，镜像、实验数据及发布件在企业赛/work。继承校赛 V2 的三项 CPU 模型、父检查点与 Adam 状态读写以及 SDK 1.0.4。比赛运行使用宿主提供的 Qwen/Qwen3.8-27B，温度 0，规划和裁决输出最多 56、32 Token。线上工作区由赛方挂载。宿主启动镜像的 describe、train 或 predict 子命令，模型和历史由 SDK 传入。predict 接收每条授权输入并立即写出分类分数或寿命值。建议从技术报告的冻结压力表开始阅读。

## 复现
从服务器项目根进入 企业赛/V1。使用 ai-plus 环境，设置 OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1，所有 Python 命令加 -B。已有校赛/work 提供官包解出的公开 JSONL、官方检查点、SDK、校验过的基础镜像与 NumPy 2.2.6 wheel。全新环境可按校赛/README.md 准备这些公共依赖并运行其中的 prepare.py。本阶段的写入目录均在企业赛。来源及开放组件说明见三份分项研究记录以及原始 SDK 授权材料；Python/Debian、NumPy 的分发说明保留在基础镜像/安装目录。开发用 NumPy，DS-PS 的 MATLAB 和发动机 HDF5 扩展实验分别需要 SciPy、h5py。报告脚本使用 M2Gaze 环境的 PyMuPDF 和 FFmpeg。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B test_loop.py
for key in A B C; do
  for v in campus enterprise; do
    for split in 0 1 2; do python -B benchmark.py "$key" "$v" "$split"; done
    python -B benchmark.py "$key" "$v" 1 --setting official
  done
done
python -B resource_bench.py
python -B kaist_shift.py
python -B build_image.py
python -B verify_image.py
/root/miniconda3/envs/M2Gaze/bin/python -B release.py
```

gas_study.py、bearing_study.py、engine_study.py 的数据准备与命令见相应的 study.md。ablate.py 将预设的守门策略对照保存在 work/ablation。public_smoke.py 每个任务执行真实 27B 公开首轮，需要账户凭据并会产生费用，相同请求复用企业赛/work/live 下的回执。当前发布所用实验在服务器已有完整日志。

## 回放口径
drift 每个训练组按顺序从 55% 逐轮释放到 100%，初始网络只用首批数据重建。每条公开验证记录只在同一条回放的选型侧或审计侧出现。A 按气体、B 按轴承沿记录顺序划四阶段，C 按整机划选型与审计，再按循环划四阶段。上台模型只能看到当前阶段的公开选型行，所有任务的设备元数据在进入 Harness 前移除。静态参照始终是该条回放的首个检查点。split 0 为开发，split 1 交换验证侧并改变顺序，split 2 使用中段或另一组发动机、以末期先到的顺序形成额外压力。多条回放共享公开来源。official 固定官方检查点和完整公开训练池，轮换选型池。所有这些 CPU 对照用脚本规划与裁决，允许生产保护门拒绝；history 回滚分支另有专门协议测试。真实 API 的三任务首轮另列。

每轮对当前阶段的主指标计算相对静态改善（F1 增长、RMSE 下降为正）。每条流分别取十轮均值及最小值；遗忘是已见阶段相对自己历史最高改善率的最大回撤；恢复是在五次阶段切换中，当前相对收益重新非负前经历的负收益轮数，阶段结束时截尾。这三个口径只用于公开代理比较。资源表的 CPU 为训练闭环进程时间，含候选序列化及宿主替身复核，模型调用为本地应答；单条预测含输出 JSON 序列化。均不含赛方排队、网络与隐藏测试计量。

## 运行与回退
部署镜像标签 bjtu-rsi:enterprise-v1，Non-root UID 65534，CPU 单线程。推理进程使用只读父产物，现场诊断应观察缺失字段、非有限数、偏离训练工况和弱类支持量。输入或权重不满足契约时向宿主报告失败并排查对应资产。裁决无效会保留父产物。仅在宿主可见并通过当前公开数据重测时可选择历史产物。生产上线时按镜像摘要和产物 ID 冻结版本，异常时将部署恢复到上一批准的镜像与检查点。寿命预测用于设备巡检排程，检修动作由既有设备规程与人员复核决定。
