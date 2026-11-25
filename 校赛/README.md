# 校赛 · 智训引擎

本目录集中维护校赛作品。运行位置是 BJTU 服务器的本目录。赛方 SDK 和训练包使用仓库同级目录的原件，公开投影、权重、运行记录和提交归档保存在 Git 忽略的 `work/`。宿主入口是 `python3 /app/harness.py describe|train|predict`。三项任务共享一个 Linux/amd64 镜像；作品选择基础模型 `Qwen/Qwen3.8-27B`。

## 复现实验

使用项目的 Python 3.12 环境，已安装 PyTRIO 0.2.9。开发对照使用 NumPy 2.5.4，镜像固定为 2.2.6。真实调用使用 PyTRIO 提供的 OpenAI 兼容接口和该账号已有的登录配置。

```sh
cd "/opt/data/private/mzx/mzx/AI +/校赛"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai-plus
python -B prepare.py
python -B rehearse.py control --rounds 10 --name repeat-cpu
python -B rehearse.py live --keys ABC --rounds 3 --name repeat-qwen
python -B publish_records.py repeat-qwen
```

每次实验使用新的输出名称。最后两条分别消耗 PyTRIO 推理额度、发布到已登录账号的 SwanLab 公共项目。比赛容器内仅使用 SDK 的宿主网关，SwanLab 记录上传在 BJTU 开发侧执行。平台产生的正式记录由对应运行页面导出。

## 构建与验证

有 Docker 时，从本目录运行 `python -B prepare.py`，再运行以下命令。源码仓库含 Dockerfile 和构建说明。无 Docker 的 BJTU 采用第二种等价打包方式。

```sh
docker build --platform linux/amd64 -t bjtu-rsi:campus-v1 .
docker save -o work/delivery/harness.tar bjtu-rsi:campus-v1
```

在 BJTU，将 Google go-containerregistry 官方发布的 crane 置于 `work/tools/crane`，校验其发行版 SHA-256，然后执行：

```sh
mkdir -p work/tools
work/tools/crane pull --platform linux/amd64 public.ecr.aws/docker/library/python@sha256:1475c4f5b6bcb7d35e57661d6f5d7429ec4041275633c15f2b49350ac898276d work/tools/base.tar
python -m pip download --no-deps --only-binary=:all: numpy==2.2.6 -d work/tools
python -B build_image.py
python -B verify_image.py
```

镜像构建器校验基础镜像配置、各层及 NumPy wheel 摘要。镜像中只有运行时、算法和 SDK。验收程序需要可用的 chroot，以 UID 65534 执行三个 SDK 入口，覆盖两轮连续的父权重读取、分块上传、决策回执和逐样本预测。该验收使用回放决策。

## 作品与来源

- 方法、实验结果、论文及开源项目比较见 [研究与实验](研究与实验.md)。
- 开源与赛方材料使用范围见 [来源声明](来源声明.md)。
- [公开 SwanLab 实验记录](https://swanlab.cn/@Zixin-Ma/ai-plus-campus-rsi)按数据集和验证方式命名。
- [参赛平台](https://rsi-index.115.zone/)接收镜像 tar、训练记录链接、源码仓库链接和作品文档。评测得分以宿主回执为准。

## 交付包

开发侧从本目录执行 `/root/miniconda3/envs/M2Gaze/bin/python -B make_materials.py`，使用 PyMuPDF 1.28.2 和 FFmpeg，将真实实验记录与镜像验收记录编入技术报告和 60 秒演示。运行 `package_delivery.py` 将源码、运行说明、研究依据与来源声明打包，并生成内嵌源码与视频的独立作品文档（PDF 附件面板可打开）。提交镜像独立上传。公开实验与本次归档均在 BJTU 上生成。
