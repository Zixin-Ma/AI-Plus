# 企业赛 V7 工业漂移验证

V7 以冻结的企业赛 V6 为本地基准。队伍最近一次平台正式报告是 V5，编号 `80900572e91d7d33d6637088915dc5e6`。该报告的五项客观分依次为漂移流 9.079221/15、最差阶段 0/5、恢复 3.968254/5、稳定性与抗遗忘 4.986910/5、资源效率 5/5。公开回放观察到的相对收益不能换算成平台分数，公开协议中仍有负收益的任务，正式最低阶段能否得分应由平台检验。

## 方案与实验依据

- A 用响应强度覆盖动力学原型，调节传感器共模距离，小样本时向跨类共享的响应斜率收缩。公开首批八组留出、跨日、跨重复及单样本兼容性详见[气体实验](气体实验.md)。
- B 针对观察窗口截断与寿命长尾，将设备贡献与训练轨迹成熟度关联，保留已有预测回退通道。按三台验证设备留工况、全部 27 种旧窗口及额外设备 RUL 作配对检查。
- C 用源轨迹成熟度控制寿命正则项的适用范围，独立核算过早与过迟预测造成的 NASA 代价，并从不可变头参数编译标量预测。
- 控制模型为 PyTRiO 的 Qwen/Qwen3.8-27B。沿用 V6 完整诊断规划提示，用明确列出的可接受动作复核部署。工业预测器以公开选型指标及历史记忆约束候选。从旧模型及新专家分别验证分组下降风险惩罚。各路线的配对结果及未采用方案见本目录的分项实验报告和公开证据包。训练与预测容器通过 SDK 接受宿主调用，研究侧的真实控制轨迹使用服务器网络连接 PyTRiO，逐条公开回执另行存档。

## 预登记的复现协议

在服务器仓库根目录运行。依赖 ai-plus 环境 CPython 3.12、NumPy 2.2.6，Harness SDK 1.0.4 与官方起点位于校赛工作区，外部设备位于 `Supp_Datasets`。研究人员可使用阶段标识划分验证、审计集合，评测 Harness 收到的行已经去除设备及阶段元数据。全部工件保存在 `企业赛/work/v7`。

开发顺序 0、5、8、13、16、19，单独预留 101–108。三任务每种顺序有七种发布与选型协议，每条流执行十轮，通过同轮的训练、可见选型和审计行编号摘要校验配对。A、B 的新顺序改变连续审计窗口位置和宽度，C 以整机留出。主比较先对同一阶段的访问取均值，再对各阶段取最小值，最后对八条流取均值。均使用相同公开库，窗口会重叠。官网日程代理按 V5 已公开的计数比例更新训练池，并沿用观测到的阶段时序和工具动作。额外的 transfer 模式从选型集合剔除当前审计工况。official 模式由真正的完整官方检查点起步；其余模式以首批可见行重建静态基线。完整首批另用全规模训练及整组选型留出复验。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
v=企业赛/V7
w=企业赛/work/v7
python -B "$v/qualify.py" register
python -B "$v/qualify.py" dev --workers 12
python -B "$v/qualify.py" freeze
python -B "$v/qualify.py" confirm --workers 12
python -B "$v/qualify.py" summary
python -B "$v/audit.py"
python -B "$v/scorecard.py"
python -B "$v/bearing_partition_grid.py" --workers 6
for t in A B C; do
  for version in v6 v7; do python -B "$v/opening_history.py" "$t" "$version"; done
  python -B "$v/serving_study.py" suite "$t"
  python -B "$v/resource_bench.py" "$t"
done
python -B "$v/build_image.py"
python -B "$v/verify_image.py" > "$w/image-proof.log"
python -B "$v/artifact_audit.py"
python -B "$v/sdk_latency.py"
python -B "$v/sdk_latency.py" --repeat-session
python -B "$v/resource_summary.py"
python -B -m unittest discover -s "$v" -p 'test*.py' -v > "$w/tests.log" 2>&1
```

## PyTRiO 真实控制轨迹

查看现有结果只需解压公开证据包。服务器 `cloud/receipts` 记录请求摘要、回复和 Token，每个请求按实际内容去重复用。对新输入执行下列命令会调用队伍的 PyTRiO 账户。账户配置只在服务器用户目录。两版运行相同的公开数据和动作约束，实际模型可自行选择规划。已知划分使用两种协议，另在冻结前规定从八个主协议预留划分取 101、104、107，以真实控制器和三项工业模型再做九对完整回放。

```bash
for t in A B C; do
  for version in v6 v7; do
    python -B "$v/live_replay.py" "$t" "$version" official 13
    python -B "$v/live_replay.py" "$t" "$version" schedule 17
  done
done
python -B "$v/cloud_confirmation.py" --workers 4
# 冻结确认后，为复查 FD004 的 NASA 尾部代价追加两个定向案例
for split in 103 108; do
  for version in v6 v7; do python -B "$v/live_replay.py" C "$version" schedule "$split"; done
done
python -B "$v/evidence.py"
```

## 交付、运行与回滚

`企业赛/work/v7/delivery` 包含 Linux/amd64 Docker-save 镜像、模型配置、源码、PDF 报告、Markdown 汇总和公开实验包，摘要列在 SHA256SUMS.txt。镜像入口 `python3 /app/harness.py` 支持 describe、train、predict；启动示例为 `docker load -i harness.tar`、`docker run --rm --network none bjtu-rsi:enterprise-v7 describe`。训练经宿主提供的 SDK 工作区与模型网关启动；流式预测从标准输入逐行接受 JSON，并输出逐行结果。镜像核验覆盖三个任务各三轮、从上轮父产物恢复和无特权预测。

冻结的 V6 镜像、代码和历史实验保存在同级目录。部署回滚可重新加载 V6 交付的镜像与原始父工件。两场 SDK 吞吐测量包含 V6 检查点在 V7 上的逐样本数值回归检查。研究版通过新旧检查点的同一接口恢复。补充数据仅供开发和独立迁移诊断，隐藏数据和账户信息均不进入镜像或源码包。

## 软件及来源

本队实现各任务校正头、工业控制策略和研究工具，开发使用 AI 编程辅助。三任务骨干与 SDK 从赛方提供的 train_origin 和 Harness_SDK 1.0.4 恢复，按竞赛用途使用。镜像包含 Python 3.12 slim-bookworm 基础层及发行方版权声明；NumPy 与 wheel 所含原生库的声明保留在包内。crane 仅用于服务器构建及核验。项目源码包附校赛《来源声明》。[CPython 许可](https://docs.python.org/3.12/license.html) · [NumPy 许可](https://numpy.org/doc/2.2/license.html)

跨设备实验分别使用原生传感器空间和寿命时间单位。正式运行完全依赖已释放的赛方输入。对数据来源、每项启发工作的作者和实现使用范围，参阅本目录三份分任务报告。

## 阅读冻结结果

公开验证采用 294 对十轮发布流，含开发 126 对、冻结确认 168 对。八个新日程划分使用固定动作比较工业方案；其中 101、104、107 在冻结前另行登记，使用实际控制模型复核。103、108 是观察到发动机 NASA 增量后选取的风险案例，单独统计。真实模型的开发流、预留流及事后风险案例均按同轮的公开数据进行有对应版本的复算，原始请求按内容摘要缓存。

冻结后最新结果见交付目录的验证汇总与技术报告，关联的公开数据、代码及指标摘要由 `artifact_audit.py`、`audit.py` 和 `evidence.py` 分别检查。留出的多个窗口仍共享同一批基础数据，尤其 B 的 27 种旧窗口只来自三台独立验证设备。针对实际工厂的迁移判断应优先参考分设备回放与日期留出结果。
