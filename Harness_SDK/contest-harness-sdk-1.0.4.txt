# 选手 Harness SDK 使用说明

SDK 版本：`contest_harness` 1.0.4。协议：`contest.harness.workspace.v2`。提示词版本：`industrial-workspace-flexible-v3`。

这份说明介绍如何用 SDK 读取平台提示、公开数据和父模型，以及如何让平台调用镜像。评分、隐藏标签和计量由平台负责；SDK 不包含分数，也不替选手设计训练算法。

## 提交里要有什么

1. 模型身份。持久 Sampler 填写完整 `trio://` 路径；基础模型只填写平台白名单中的名称，并完成二次确认。PyTRiO API Key 只填在平台密钥栏，不写入镜像、源码、文档或训练记录。
2. 训练与预测代码。算法、续训方式和实验记录由选手自己决定。
3. SDK。镜像内安装本包，并把 `template/harness.py` 放到 `/app/harness.py`。选手模块默认是 `/app/contest_submission.py`；使用其他模块名时，设置环境变量 `CONTEST_SUBMISSION`。

SDK 推荐使用以下固定入口：

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

`describe` 向 stdout 打印一行 JSON，至少包含：

```json
{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","tasks":["classification","regression"]}
```

`tasks` 只能包含镜像实际支持的 `classification` 和 `regression`。`describe` 会导入选手模块并检查 `train`、`predict` 和任务声明；导入或检查失败、协议字段不符合要求时，平台不会进入训练。通过 `describe` 不代表训练代码已经过完整审查，也不保证训练成功。

## 目录与权限

| 路径 | 权限 | 内容 |
|---|---|---|
| `/app` | 只读 | 选手代码、SDK、`/app/harness.py` |
| `/data/train.jsonl`、`/data/validation.jsonl` | 只读 | 当前阶段公开投影 |
| `/previous` | 只读 | 当前父产物，路径与清单一致 |
| `/state` | 只读 | 上一检查点保存的原生状态；首轮可能不存在 |
| `/model` | 只读 | 只在预测时挂载被选中的产物 |
| `/workspace` | 可写 | 本轮工作目录。候选、状态副本和临时工程放这里 |
| `/tmp` | 可写 | 临时文件。运行环境可能禁止在这里执行程序 |

不要在 `/app` 下创建目录或改文件。不要假设镜像根目录可写。

当前新建CCE沙箱的CPU配额为2核、内存上限为4GiB；不是无限内存。工作卷4GiB、输出和临时卷各1GiB，使用磁盘存储。容器内存不足会被内核终止；不要仅凭退出码137推断原因，应结合平台保留的运行诊断。

## 公开数据

训练和验证文件是 UTF-8 JSONL。空行跳过。每一行是一个对象，字段固定为：

```json
{"id":"样本标识","inputs":{"特征":0},"answer":"标签或非负数值"}
```

按随基线发布的训练指南，梯度训练与拟合统计量只使用训练集。公开验证集可用于测量、模型选择和决策，但不能并入梯度训练；这与隐藏/未释放标签的访问限制是两条独立边界。参考超参数和profile模板不限制选手的算法设计。

`id` 是字符串。`inputs` 由当前数据集的冻结说明决定。`answer` 只出现在授权的公开训练集和公开验证集。隐藏集不会挂进训练容器，也不能从文件名、样本号或匿名标识反推。

`Round.data.training_rows()` 与 `Round.data.validation_rows()` 按文件顺序逐行读取，不把整份数据一次性拼成另一份“官方包”。平台给出的路径写在 `rnd.task["public_projection"]`。当前默认就是上面的两个文件。

需要 CSV 或其他目录布局时，在 `/workspace` 里自行转换。转换规则由选手代码写明，平台不按队伍猜测列名。

## 父模型

`rnd.parent.checkpoint_sha256` 是本轮必须续训的父检查点摘要。`rnd.parent.path(相对路径)` 只打开清单里声明过的文件，并核对摘要。路径形如 `checkpoint.npz` 或 `baselines/model.py`，来自父产物自己的 `checkpoint_paths` 和 `code_paths`。摘要不一致时 SDK 抛出 `PARENT_DIGEST_MISMATCH`。

不要搜索 `/app` 来找权重，不要改 `/previous` 里的字节，也不要另造一份 `baseline-info.json` 去凑历史报告里的哈希。

## 一轮训练

`train(rnd)` 收到的 `rnd` 来自平台写入 stdin 的一帧 `start`。选手需要使用的字段已经放在对象上：

- `rnd.prompt`：本轮固定任务提示，只读。
- `rnd.prompt_version`：提示词版本。
- `rnd.dataset_key`：`A`、`B` 或 `C`。
- `rnd.round_index`：当前检查点序号，从 1 开始。它不是“原生程序内部只能训练一次”。
- `rnd.task`：冻结的任务类型、指标、指标方向、数据系列、目标单位和公开投影。任务类型以这里为准，不要把数据集代号当成 `classification`。
- `rnd.model_name`：本轮绑定的模型名。
- `rnd.max_output_tokens`：本轮允许的单次输出上限。
- `rnd.limits`：候选上传、状态上传和元数据的字节上限。

官方提示由宿主在请求发往模型前加到第一条 system 消息中，并附上本轮运行、数据集、检查点和任务合同。`rnd.prompt` 可供选手代码读取，但不要把它再放进 system 或 user 消息，否则会重复。自己的系统提示可以放在第一条 system；宿主会把官方提示加在它前面。

使用 SDK 时，可以直接调用模型：

```python
receipt = rnd.model(
    [{"role": "user", "content": "这里只放本轮数据、代码状态和你自己的问题"}],
    max_tokens=1024,
    temperature=0,
)
```

省略 `max_tokens` 时使用本轮授权上限，不会默认改成 4096。`receipt.call_id` 是平台回执号，`receipt.response` 是完整响应，`receipt.text` 是第一条文本回复。密钥、真实端点和计费由平台持有，不要把 PyTRiO 密钥或外部采样客户端放进镜像。

已有 OpenAI 兼容客户端时，可以使用本地传输层，保留客户端的消息、采样和响应解析方式：

```python
with rnd.transport() as transport:
    client = OpenAI(base_url=transport.base_url, api_key=transport.api_key)
    client.chat.completions.create(model=rnd.model_name, messages=messages, max_tokens=1024)
```

`transport.api_key` 是本地占位符，不是 PyTRiO 密钥。`transport.base_url` 提供 `GET /v1/models` 和 `POST /v1/chat/completions`。传输层支持工具调用、多个 choices，以及按 `stream_options.include_usage` 请求流式 usage。流式请求会先完成一次非流式平台调用，再把完整回执转成事件流；它不是实时 token 流。调用完成后，`rnd.call_ids` 中会记录真实回执号。

若模型未返回结束原因，SDK 会为事件流补齐结束帧：有工具调用时使用 `tool_calls`，否则使用 `stop`。这是桥接层的默认处理，不代表模型报告了该原因；模型明确返回的原因（如 `length`）保持不变。

`rnd.history()` 返回本数据集已经完成的检查点记录。结束本轮时引用实际读过的 `id`。记录可以是空的。

候选从 `/workspace` 的一个新目录提交。目录里只放普通文件，单次候选或状态最多 64 个文件、合计 512MiB；每轮候选累计 1GiB、状态累计 512MiB，具体以 `rnd.limits` 为准。至少声明一个检查点文件。`predict_argv` 必须是平台稍后会原样执行的命令：

沙箱工作区是临时空间：SDK 先分块计算摘要，再分块上传到宿主的持久产物存储；提交完成后宿主才验证、保存和返回回执，上传期间不要修改文件，也不必更换检查点格式。预算在开始上传时预留，包括未完成的上传。容量、摘要或传输错误是资源/协议问题，不是算法违规。SDK 1.0.3 需要支持分块上传的新宿主；不兼容旧宿主时明确报错，不会退回大消息内联上传。旧版 SDK 的内联协议保持原有 64MiB 限制。

SDK 1.0.4 中，RPC 发出后如被信号、超时或异常中断，宿主是否已处理该请求未知。此轮会话不再可用，后续 RPC 报 `RPC_OUTCOME_UNKNOWN`，不会发送新请求、消费迟到响应或自动重试。不要捕获中断后继续调用；候选提交包括宿主验证，选手自设超时不保证足够。

```python
submitted = rnd.submit_candidate(
    candidate_dir,
    hypothesis="本轮假设",
    change_summary="相对父模型实际改了什么",
    configuration={"epochs": 1},
    seed=1,
    checkpoint_paths=["checkpoint.npz"],
    code_paths=["baselines/model.py"],
    predict_argv=["python3", "/app/harness.py", "predict"],
)
```

`hypothesis` 和 `change_summary` 必须是非空文本，各自不超过 4000 字节。`configuration` 是对象，序列化后不超过 64KiB。`seed` 是整数。这三个值由选手如实填写，平台不据此改分数。

平台测量后返回 `artifact_id` 和公开验证摘要。测量数字以返回值为准，不要在代码里另报一套分数。

本轮要留给下一轮的原生状态，先复制再写：

```python
state = rnd.state_workspace("/workspace/state")
# 只写 state 里面的文件
rnd.save_state(state)
```

`/state` 本身只读。不保存则沿用上一份状态；保存空目录表示清空。同一轮多次保存时，每次保存都以本轮开始时的状态为父状态，不以前一次保存的结果为父状态。

同一数据集里更早的产物用 `rnd.restore_artifact(artifact_id, "/workspace/history/某id")` 取回。只能取本帧 `artifacts` 里出现过的标识，取回后核对摘要。

结束本轮：

```python
rnd.finish(
    selected_artifact_id=submitted["artifact_id"],
    hypothesis="本轮假设",
    change_summary="相对父模型实际改了什么",
    reason="根据本轮真实模型回复和平台测量做出的选择。",
    configuration={"epochs": 1},
    seed=1,
    history_ids=[item["id"] for item in history],
    model_call_ids=list(rnd.call_ids),
)
```

`model_call_ids` 不能为空，而且必须是本轮 `rnd.model()` 或传输层实际返回的调用号。因此每一轮都要有至少一次真实模型调用。选择当前新候选、保留父模型，或选回更早的产物，都由这次调用里的决定说明。`reason` 写实际理由。调用 `finish` 并收到平台确认后进程退出。不要在确认前再发协议帧。

stdout 只写协议 JSONL。日志写 stderr。不要在训练进程里等待 stdin 结束才开始输出。

## 预测

平台按选中产物的 `predict_argv` 启动进程，只把该产物挂载到 `/model`。SDK 传给 `predict` 的 `model_dir` 是 `/model` 根目录，不是检查点文件。预测没有训练用的 start 帧，stdin 第一行就是样本。平台还会设置：

- `CONTEST_ARTIFACT_PATH`：选中的检查点文件路径，不是产物目录。
- `CONTEST_ARTIFACT_MANIFEST`：产物清单。
- `CONTEST_PREDICTION_CONTEXT`：一行 JSON，协议 `contest.predict-context.v1`，字段只有 `dataset_key`、`task`、`artifact`、`artifact_id`。
- `CONTEST_DATASET_KEY` 与 `CONTEST_TASK`：与上面的上下文一致。`task` 仍是 `classification` 或 `regression`，不是 `A`、`B`、`C`。

`sample.dataset_key`、`sample.artifact_id` 和 `sample.context` 来自这个上下文。上下文缺失时只使用样本行里的 `task`。样本行的 `task` 与上下文不一致时，SDK 拒绝该行。

stdin 每一行是：

```json
{"task":"classification","inputs":{"特征":0}}
```

或

```json
{"task":"regression","inputs":{"特征":0}}
```

`task` 是冻结合同里的 `classification` 或 `regression`，不是数据集代号 `A`、`B`、`C`。

进程必须在读取下一行之前写出本行结果并刷新。分类：

```json
{"outputs":[{"label":"normal","score":1}]}
```

每个 `label` 是非空字符串，`score` 是 0 到 1 的有限数，标签不能重复，最高分只能有一个。回归：

```json
{"outputs":{"value":0}}
```

`value` 是有限的非负数。缺行、等待 EOF、超时或形状不对，这一整次测量失败，平台不会补预测值。

```python
def predict(sample, model_dir):
    # model_dir 是 /model 根目录；检查点路径见 CONTEST_ARTIFACT_PATH
    if sample.task == "regression":
        return {"value": 0}
    return [{"label": "normal", "score": 1}]
```

## 本地检查

在 SDK 目录运行离线测试：

```sh
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

这些测试使用 JSONL 宿主替身，不会调用真实模型。真实宿主的协议回归测试位于仓库的 `runtime/tests/test_sdk_host_contract.py`；SDK 下载包不包含宿主。`tests/test_live.py` 是另行运行的真实模型测试，设置 `PYTRIO_API_KEY` 后可能产生费用；设置 `CONTEST_PI` 指向精简 pi 的 `pi.mjs` 时，还可通过 SDK 传输层运行握手。运行真实模型测试前，请确认密钥和环境。单元测试通过不代表比赛中能得到实际分数。

示例候选只展示协议和模型调用，不是可得分的训练方案，也不能证明工业模型成绩。

## 其他接入方式

旧的 `contest_sdk` HTTP 客户端使用 `/runtime/v1/runs/...`，与本 SDK 协议不同。固定的 `/app/harness.py` 入口是 SDK 推荐接法，并不表示宿主完全禁用了其他 adapter。无论如何接入，都应遵守平台的协议、模型访问和目录权限要求。
