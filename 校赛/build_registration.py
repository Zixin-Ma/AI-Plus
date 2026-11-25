"""Generate seven separately uploadable registration materials for the frozen campus v1."""
from pathlib import Path
import argparse, hashlib, html, json, zipfile
import pymupdf as pdf
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from docx import Document
from docx.shared import Pt as DocPt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from make_materials import page, SWAN

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"work/registration-v1"
HASH="a6d6fea006d4cc43adef522a3bc42a04dc80c6a4700466809ac24fb9ed3d9d38"
REPO="https://github.com/Zixin-Ma/AI-Plus/tree/main/%E6%A0%A1%E8%B5%9B"
RUN=[
("运行环境与依赖",[
"本材料对应 McCarthy 队已经提交的 v1 镜像。Linux/amd64，Python 3.12，NumPy 2.2.6，contest_harness SDK 1.0.4，协议 contest.harness.workspace.v2。模型选择 Qwen/Qwen3.8-27B，由 PyTRIO 官方基础模型服务提供。",
"平台每项任务运行 10 轮。SDK 当前沙箱提供 2 核 CPU、4 GiB 内存、4 GiB 工作卷和各 1 GiB 的临时卷与输出卷。镜像以 UID/GID 65534 执行，OPENBLAS_NUM_THREADS 和 OMP_NUM_THREADS 均为 1。",
"宿主通过 host_gateway_rpc 提供真实模型请求、官方提示和用量计量。每轮一次调用，temperature=0，enable_thinking=false，输出上限为本轮授权值与 160 的较小值。服务密钥仅配置在平台专用字段。",
"镜像自带运行依赖。开发复现使用赛方 train_origin 和 Harness_SDK 原始包，Python 3.12 的 ai-plus 环境。开发依赖与文档制作依赖分别列在源码包中。"]),
("启动命令与挂载路径",[
"导入镜像执行 docker load -i harness.tar。镜像标签 bjtu-rsi:campus-v1。独立离线声明检查执行 docker run --rm --network none bjtu-rsi:campus-v1 describe。",
"固定入口分别为 python3 /app/harness.py describe、python3 /app/harness.py train、python3 /app/harness.py predict。train 和 predict 由赛方宿主通过标准输入输出启动并按 SDK 合同提供协议帧。",
"/app、/data、/previous、/state 只读，/model 在预测时只读。候选写入 /workspace，CONTEST_WORK 可覆盖候选根目录。选手模块默认 contest_submission，CONTEST_SUBMISSION 可覆盖模块名。",
"预测上下文由 CONTEST_PREDICTION_CONTEXT、CONTEST_DATASET_KEY、CONTEST_TASK 和产物清单提供。CONTEST_ARTIFACT_PATH 是检查点文件路径，predict 的 model_dir 参数是产物根目录。",
"stdout 专用于协议 JSONL，日志写 stderr。独立声明检查可离线执行，完整训练需要宿主网关和真实模型调用。"]),
("输入输出及配置参数",[
"公开数据为 UTF-8 JSONL，每行包含 id、inputs、answer。特征顺序由父模型清单固定。A 输入 128 项特征，输出六类气体概率。B 输入 34 项快照统计量，输出剩余分钟。C 输入 21 个传感器及 3 个运行设置，输出剩余循环。",
"预测输入包含 task 和 inputs。A 的 task 为 classification，B/C 为 regression。A 返回 label/score 数组，B/C 返回 value。SDK 包装 outputs 并逐行刷新。回归输出截断为非负数。",
"A/B/C 的基础学习率分别为 0.001、0.0008、0.0006，每轮训练 epoch 数分别为 3、1、1。五轮循环的学习率倍率为 1、0.45、0.15、1.5、0.7。恢复父模型权重、输入与目标尺度及 Adam 状态。",
"残差使用 96 维余弦投影，A 另加最多 160 个训练锚点的相似度。新增基函数种子为 2026 加当前轮次。已部署基函数保持固定。三种残差更新比较收缩系数 0.35、0.85、0.90 及随轮次变化的正则系数，具体以 app/industrial.py 为准。",
"训练集拟合特征变换与残差，验证集测量并选型。收益小于等于 1e-8 时保护规则保留父产物。"]),
("离线复现与镜像重建",[
"在校赛目录激活 Python 3.12 的 ai-plus 环境，执行 python -B prepare.py 读取赛方原始包，再执行 python -B rehearse.py control --rounds 10 --name repeat-cpu。name 使用新的名称以保留既有实验。",
"control 模式用数值收益选择产物，复现候选与保护规则。真实模型联调命令为 python -B rehearse.py live --keys ABC --rounds 3 --name repeat-qwen。开发期每项最多 3 轮，调用产生 PyTRIO 推理费用。",
"开发联调读取账号外部登录配置。上传实验记录命令为 python -B publish_records.py repeat-qwen，使用已登录的 SwanLab 账号。",
"有 Docker 时执行 docker build --platform linux/amd64 -t bjtu-rsi:campus-v1 .，再执行 docker save -o work/delivery/harness.tar bjtu-rsi:campus-v1。无 Docker 的 BJTU 使用 build_image.py，固定基础镜像和 NumPy wheel 摘要，准备命令见 README.md。",
"执行 python -B verify_image.py 验证镜像归档，需要可用的 chroot。该验收以 UID 65534 执行 A/B/C 各两轮、候选上传与逐样本预测，使用宿主和模型决策回放。平台全流程实测证据见试运行报告。"]),
("异常处理与回滚方法",[
"正常回退由 rnd.finish 选择父产物。下一轮从宿主绑定的父检查点续训。候选收益为正且 Qwen 选择 candidate 时采纳，指标退化、收益过小或回复无法解析时保留父产物。",
"SDK 核验父文件摘要。摘要不符、空数据、非有限公开输入、任务不符或模型身份不符会终止该轮，由平台保留诊断。核查挂载、公开投影和配置后，使用平台允许的重试流程恢复。",
"模型网络错误和 RPC 中断会向宿主传播。SDK 1.0.4 在结果未知时终止会话，平台确认状态后才可按其规则重试。上传期间保持候选文件字节不变，避免破坏摘要。",
"每次完成记录引用实际历史 ID 和本轮真实模型调用 ID。候选检查点、代码、配置、种子、假设、修改和部署理由共同构成复核链。凭据由宿主持有，试运行与正式评测以赛方计量为准。"])
]
PROJECT=[
("工业模型在漂移下持续更新",[
"McCarthy 队，智训引擎，当前参赛版本 v1，材料日期 2026 年 11 月 25 日。",
"针对气体传感器批次漂移、轴承跨设备寿命差异与发动机工况变化，系统从赛方小型 CPU 模型持续更新。Qwen/Qwen3.8-27B 读取候选实验与历史，判断是否部署。",
"核心设计是保留父模型和优化器状态的小步训练，比较正则残差候选，并以公开验证保护控制退化部署。训练与决策运行于 SDK 1.0.4 和赛方网关。",
"平台试运行已完成三项任务各 10 轮，生成 14,773 条隐藏样本预测。第 1 次正式评测已于当日 11:13:56 启动。本文冻结已取得的试运行证据，正式得分以主办方发布为准。"]),
("系统架构与自主实验",[
"赛方宿主绑定当前公开投影和父产物，SDK 验证摘要。CPU 训练器恢复原始 MLP、标准化参数与 Adam 状态，生成纯续训、继承残差和不同正则强度的候选。",
"诊断模块量化训练与验证的均值差，按训练标准差归一化，记录偏移最大的三个特征。该描述为部署判断提供上下文。更新序列采用预设的小步策略并通过验证集选型。",
"本地比较候选后向宿主上传一个候选。宿主测量公开指标，Qwen 读取父指标、候选指标、有符号收益以及最近两轮摘要，输出 candidate 或 parent 及理由。",
"保护规则要求严格正收益且模型同意后才部署。每轮从实际选中的产物恢复，保留完整谱系、调用回执和历史引用。三项各十轮形成连续自主实验记录。"]),
("漂移特征与模型改进",[
"A 的公开验证特征 41 相对训练均值偏移约 11.46 个标准差。对 16 个传感器的 8 类响应做有符号 log1p 变换，减去同类响应中位数，使用响应形状和训练锚点相似度进行校正。",
"B 使用双通道振动快照的 34 项统计量。对长尾特征做 log1p 压缩，比较普通残差和降低长寿命样本权重的稳健残差。训练与验证按整根轴承隔离，目标为剩余分钟。",
"C 仅用训练行确定常见工况中心与传感器均值方差，将当前运行设置匹配最近工况。使用工况下传感器偏差校正剩余循环。寿命标签保持原始值。",
"线性残差基于训练变换和余弦随机投影拟合，正则和收缩控制更新幅度。部署后基函数固定，系数可增量更新。所有新统计量从训练集拟合，公开验证用于测量与选择。"]),
("相同公开划分的静态基线对照",[
"A 的 Macro-F1 从静态起点 0.714030 提升至十轮 CPU 对照的 0.725782，增加 0.011752。",
"B 的 RMSE 从 681.1598 分钟下降至 649.7214 分钟，降低约 4.62%。C 的 RMSE 从 52.8686 循环下降至 48.6173 循环，降低约 8.04%。",
"公开验证集 A/B/C 分别为 1,244、1,679、23,049 行。CPU 对照每项十轮，按验证指标自动选择，用于衡量候选序列与部署保护。验证划分参与选型。",
"真实 27B 开发联调的 A/B 各三轮选中指标为 0.724210 与 649.7214。C 初版三轮保留基线，提示修正后独立一轮为 48.7802。初版失误与修正记录均保留。",
"B 的 1,515 条公开验证行来自同一根轴承，聚合误差对该轴承较敏感。跨设备和后续阶段表现由平台独立评测检验。"]),
("平台试运行的完整运行结果",[
"赛方试运行开始时间为 2026-11-25 10:40:25，北京时间。状态已完成，三项均标记记录完整，全部 30 轮完成。预测量共 14,773。",
"A 隐藏测试 4,886 条，Macro-F1 为 0.891003，Accuracy 为 0.896234。最终使用第十轮候选。",
"B 隐藏测试 200 条，RMSE 为 570.153626 分钟，MAE 为 403.873233 分钟。最终使用第二轮候选。",
"C 隐藏测试 9,687 条，RMSE 为 54.486098 循环，MAE 为 38.925696 循环，NASA 非对称评分为 83,402,551,853.39285。最终使用第五轮候选。",
"报告未公布同一隐藏集静态基线及达标判定。本页呈现实测指标，综合得分、排名和门槛通过情况以主办方正式发布为准。"]),
("算力、标注与模型用量",[
"试运行 CPU 用时 A 为 787.139381 秒、B 为 349.663831 秒、C 为 1,365.731502 秒，总计约 2,502.53 秒。对应单任务运行用时分别为 1,089.02、654.83、1,468.41 秒，三项并行，整场墙钟时间独立计量。",
"试运行已结算输入 21,907 token、输出 942 token，合计 22,849。开发期十次真实推理共 3,176 输入与 410 输出 token。正式评测用量以其后续报告为准。",
"开发期按当时 PyTRIO 未缓存公开价估算约 0.0204 元，实际费用以账号结算为准。单轮一次调用、关闭思考、最多 160 输出 token 用于控制推理开销。",
"训练使用赛方已授权公开标签，代码没有新增标注请求，新增标注申请为 0。平台配额消耗以宿主计量为准。CPU 工业模型训练用时与 GPU 时长分别记录。"]),
("受控机制与工业应用价值",[
"正常回滚使用父产物。保护规则在无正收益或无法解析回复时保留父模型。SDK 校验检查点摘要并按轮次绑定模型，RPC 中断后遵循会话终止语义。",
"初版 C 提示曾把 RMSE 从 52.87 降至 48.78 判断为变差。当前 v1 显式展示带符号收益，独立真实复核正确采纳。完整实验保留该修改依据。",
"容器以非特权用户执行，模型只接收聚合诊断、指标和历史摘要。运行服务密钥由宿主持有。公开验证、隐藏评测、原始日志按赛方授权访问。",
"应用方向包括传感器批次变化校正、轴承跨设备预测与发动机工况适配。当前验证了比赛环境中的训练与部署闭环，现场应用需结合采样规范、告警策略及运维流程开展设备验证。"]),
("复现、研究依据与来源",[
"源码包包含训练、评估、漂移诊断和 Harness 集成，提供配置、依赖清单与运行入口。README.md 说明公开投影准备、CPU 对照、真实联调、构建和镜像验收。",
"研究比较见研究与实验.md，覆盖气体跨批漂移、跨工况轴承寿命、发动机工况标准化，以及 AIDE、MLE-IDEATOR 等自动化实验方法。文献用于方案对照，具体实现采用赛方基线、SDK 与 NumPy。",
"CPU 基线与 SDK 由赛方提供，按竞赛用途使用。Python、Debian、NumPy 保留原始分发许可和第三方声明。参赛算法与集成经 AI 辅助开发。完整声明见来源声明.md。",
"云端实验记录 "+SWAN,
"赛方报告 https://rsi-index.115.zone/#runs 需要队伍授权登录。源码 "+REPO])
]
SLIDES=[
("工业模型在漂移下持续更新","从父模型续训，以真实实验判断部署",["McCarthy 队  智训引擎","当前参赛版本 v1","Qwen/Qwen3.8-27B 与 CPU 工业模型"]),
("三类工业观测带来持续更新需求","任务和评价指标",["气体传感器批次变化，六类识别以 Macro-F1 衡量","轴承设备间寿命差异，输出剩余分钟","发动机运行工况变化，输出剩余循环"]),
("宿主核验候选，Qwen 判断部署","系统架构",["父检查点及公开数据进入 CPU 候选实验","宿主测量后，27B 读取收益、漂移与历史","验证保护控制采纳，父产物支持恢复"]),
("任务特征校正保留父模型状态","方法设计",["A  相对响应和训练锚点相似度","B  对数变换和长尾寿命的稳健残差","C  训练集工况统计和传感器偏差校正","96 维余弦投影，固定基函数，增量更新系数"]),
("公开验证中三项主指标均有改善","固定公开划分的十轮 CPU 对照",["A Macro-F1  0.714030 → 0.725782","B RMSE 分钟  681.1598 → 649.7214  降低 4.62%","C RMSE 循环  52.8686 → 48.6173  降低 8.04%","验证划分参与选型，跨设备表现由独立评测检验"]),
("平台试运行完成全部 30 轮","赛方报告已完成，三项记录完整",["A  4,886 条预测  Macro-F1  0.891003","B  200 条预测  RMSE  570.153626 分钟","C  9,687 条预测  RMSE  54.486098 循环","共 14,773 条预测，正式得分以赛方发布为准"]),
("历史与决策形成可复核记录","至少三次连续自主实验，当前已完成三项各十轮",["每轮留存假设、修改、配置、种子与调用回执","Qwen 读取父分数、候选分数与最近历史","C 指标方向误读促成有符号收益提示修正"]),
("单轮一次调用控制推理开销","试运行总 token 为 22,849",["21,907 输入 token，942 输出 token，平台已结算","单次最多 160 输出 token，关闭思考，temperature=0","试运行 CPU 总量约 2,502.53 秒，三项并行","使用公开标签，新增标注申请为 0，配额以平台计量为准"]),
("部署保护绑定实际指标收益","安全与可恢复性",["收益小于等于 1e-8 或回复无法解析时保留父模型","父文件摘要校验及非特权 UID 65534 运行","RPC 中断后按 SDK 会话终止语义交由宿主处理","服务密钥由赛方宿主持有"]),
("单镜像支持训练和逐样本预测","运行与复现",["Linux/amd64，Python 3.12，NumPy 2.2.6，SDK 1.0.4","固定入口 describe、train、predict，单线程 BLAS","镜像验收覆盖 A/B/C 各两轮，公开对照可复现","完整源码提供依赖、配置、运行说明和来源声明"]),
("工业应用以可恢复的小步更新为基础","应用价值",["传感器批次校正、轴承跨设备预测、发动机工况适配","平台完成连续训练、候选选择与逐样本输出","现场落地结合采样、告警和运维开展设备验证"]),
("实验和源码支持独立复核","当前材料冻结到完整试运行的可核实结果",["实验记录  "+SWAN,"源码见 AI-Plus 仓库校赛目录","正式评测第 1 次已启动，成绩以主办方发布为准","研究依据和第三方声明见项目说明书及源码"])
]

def make_pdf(name, chapters):
    doc=pdf.open()
    for title, paragraphs in chapters:
        body="".join("<p>"+html.escape(s)+"</p>" for s in paragraphs)
        page(doc,"McCARTHY / CAMPUS V1",title,body)
    doc.subset_fonts()
    doc.save(OUT/name,garbage=4,deflate=True)
    doc.close()

def text_box(s,text,x,y,w,h,size,color="183733",bold=False):
    box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    box.text_frame.word_wrap=True
    for i,line in enumerate(text.split("\n")):
        p=box.text_frame.paragraphs[0] if i==0 else box.text_frame.add_paragraph()
        p.text=line
        p.font.name="Noto Sans CJK SC"
        p.font.size=Pt(size)
        p.font.bold=bold
        p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(18)

def make_deck():
    prs=Presentation()
    prs.slide_width,prs.slide_height=Inches(13.333),Inches(7.5)
    for i,(title,subtitle,lines) in enumerate(SLIDES):
        s=prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb=RGBColor.from_string("F8F9F4")
        text_box(s,title,.65,.55,12,1.0,30,"096A5E",True)
        text_box(s,subtitle,.72,1.55,11.9,.65,17,"47625C")
        text_box(s,"\n".join(lines),.78,2.55,11.7,3.9,25 if i!=11 else 22)
        text_box(s,str(i+1),12.1,7,.5,.25,10,"47625C")
        s.notes_slide.notes_text_frame.text="当前版本 v1。公开验证与平台试运行采用各自数据口径。详见项目说明书及研究与实验.md。"

    # Keep numeric comparisons and the architecture editable in PowerPoint.
    for idx, rows in [(4,[
        ["任务主指标","静态起点","十轮 CPU 对照","改善"],
        ["A Macro-F1","0.714030","0.725782","+0.011752"],
        ["B RMSE 分钟","681.1598","649.7214","降低 4.62%"],
        ["C RMSE 循环","52.8686","48.6173","降低 8.04%"]]),
        (5,[["任务","轮次","隐藏样本","隐藏主指标"],
        ["A","10/10","4,886","Macro-F1 0.891003"],
        ["B","10/10","200","RMSE 570.153626 分钟"],
        ["C","10/10","9,687","RMSE 54.486098 循环"]])]:
        slide=prs.slides[idx]
        body=slide.shapes[2]._element
        body.getparent().remove(body)
        table=slide.shapes.add_table(4,4,Inches(.78),Inches(2.6),Inches(11.7),Inches(2.7)).table
        for row, values in zip(table.rows,rows):
            for cell, value in zip(row.cells,values):
                cell.text=value
                for para in cell.text_frame.paragraphs:
                    para.font.name="Noto Sans CJK SC"
                    para.font.size=Pt(20)
        table.columns[0].width=Inches(2.8)
        table.columns[1].width=Inches(2.3)
        table.columns[2].width=Inches(2.6)
        table.columns[3].width=Inches(4.0)
        note="验证划分参与选型，跨设备表现由独立评测检验。" if idx==4 else "共 14,773 条预测。隐藏静态基线未公布，正式得分以赛方发布为准。"
        text_box(slide,note,.85,5.8,11.7,1.0,19,"47625C")
    slide=prs.slides[2]
    body=slide.shapes[2]._element
    body.getparent().remove(body)
    for x,title,detail in [
        (.8,"CPU 候选实验","父检查点与 Adam 状态\n公开训练数据拟合\n比较多种残差更新"),
        (4.95,"27B 部署判断","宿主测量的公开指标\n漂移摘要和历史\n一次 JSON 决策"),
        (9.0,"受控上线","严格正收益门槛\n采纳或保留父模型\n下一轮恢复检查点")]:
        text_box(slide,title,x,2.7,3.5,.65,25,"096A5E",True)
        text_box(slide,detail,x,3.65,3.5,2.6,22)
    from pptx.enum.shapes import MSO_CONNECTOR
    for start,end in [(4.1,4.75),(8.25,8.8)]:
        line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(start),Inches(3),Inches(end),Inches(3))
        line.line.color.rgb=RGBColor.from_string("096A5E")
        line.line.width=Pt(2)
    prs.save(OUT/"06_答辩PPT.pptx")

def make_image_doc(url):
    d=Document()
    st=d.styles["Normal"]
    st.font.name="Noto Sans CJK SC"
    st.font.size=DocPt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"),"Noto Sans CJK SC")
    d.styles["Title"].font.size=DocPt(20)
    from docx.shared import RGBColor as DocRGB
    d.styles["Title"].font.color.rgb=DocRGB(0,0,0)
    for border in list(d.styles.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    d.add_paragraph("校赛 v1 系统镜像下载说明",style="Title")
    d.add_paragraph("参赛队伍  McCarthy")
    d.add_paragraph("本附件提供当前 v1 的 Linux/amd64 单镜像 Docker save 归档信息。该归档与已提交赛方平台的 v1 镜像相同。")
    d.add_paragraph("文件名  harness.tar")
    d.add_paragraph("大小  187,013,120 字节，约 178.3 MiB")
    d.add_paragraph("SHA-256  "+HASH)
    if url:
        p=d.add_paragraph("Google Drive 下载链接  ")
        h=OxmlElement("w:hyperlink")
        h.set(qn("r:id"),p.part.relate_to(url,"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",is_external=True))
        r=OxmlElement("w:r")
        t=OxmlElement("w:t")
        t.text=url
        r.append(t)
        h.append(r)
        p._p.append(h)
        name="02_Harness镜像下载说明.docx"
    else:
        d.add_paragraph("下载链接待补。镜像原件位于 BJTU 服务器校赛/work/delivery/harness.tar。待 Google Drive 连接恢复并上传后，将可供评委下载的链接写入此附件。")
        name="02_Harness镜像下载说明_待补链接.docx"
    d.add_paragraph("核对文件大小和 SHA-256 后，执行 docker load -i harness.tar。镜像标签为 bjtu-rsi:campus-v1。")
    d.add_paragraph("声明检查执行 docker run --rm --network none bjtu-rsi:campus-v1 describe。完整训练需要赛方宿主提供公开数据、父检查点和模型网关，操作与异常恢复见系统运行说明。")
    d.save(OUT/name)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image-share-url")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(version="campus-v1",team="McCarthy",model="Qwen/Qwen3.8-27B",model_source="PyTRIO official base model",sampler=None,model_access="host_gateway_rpc",sdk="1.0.4",protocol="contest.harness.workspace.v2",temperature=0,enable_thinking=False,max_output_tokens=160,calls_per_round=1,rounds_per_dataset=10,datasets=["A","B","C"],entrypoints={n:["python3","/app/harness.py",n] for n in ["describe","train","predict"]},runtime=dict(platform="linux/amd64",python="3.12",numpy="2.2.6",uid=65534,blas_threads=1),api_key_handling="Contest platform dedicated credential field only",source_repository=REPO,experiments=SWAN)
    (OUT/"01_模型配置.txt").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"03_云端实验记录.txt").write_text(SWAN+"\n赛方记录 https://rsi-index.115.zone/#runs",encoding="utf-8")
    assert len((OUT/"03_云端实验记录.txt").read_text())<100
    assert len(SWAN)<100
    make_pdf("04_系统运行说明文档.pdf",RUN)
    make_pdf("05_项目说明书.pdf",PROJECT)
    make_deck()
    make_image_doc(args.image_share_url)
    if args.image_share_url:
        draft=OUT/"02_Harness镜像下载说明_待补链接.docx"
        draft.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT/"07_项目源代码.zip","w",zipfile.ZIP_DEFLATED) as z:
        paths=sorted(list(ROOT.glob("*.py"))+list(ROOT.glob("*.md"))+list((ROOT/"app").glob("*.py"))+[ROOT/"Dockerfile",ROOT/".dockerignore",ROOT/".gitignore"])
        for p in paths:z.write(p,p.relative_to(ROOT))
        sdk=ROOT/"work/sdk"
        for p in sorted((sdk/"contest_harness").glob("*.py")):
            z.write(p,"vendor/contest_harness/"+p.name)
        z.write(sdk/"template/harness.py","vendor/harness.py")
        z.write(ROOT.parent/"Harness_SDK/contest-harness-sdk-1.0.4.txt","vendor/SDK使用说明.txt")
        z.writestr("requirements-runtime.txt","numpy==2.2.6\n# contest_harness==1.0.4 from the contest SDK, prepared by prepare.py\n")
        z.writestr("requirements-development.txt","numpy==2.5.4\nrequests\npytrio==0.2.9\nswanlab\n")
        z.writestr("requirements-materials.txt","PyMuPDF==1.28.2\npython-pptx==1.0.2\npython-docx==1.2.0\n# ffmpeg and LibreOffice for rendering\n")
        z.write(OUT/"01_模型配置.txt","config/model-v1.txt")
        z.writestr("REPRODUCTION.md","Use Python 3.12 for training. Follow README.md for preparation, training, evaluation and image verification. Material authoring may use a separate environment. Contest data, checkpoints and SDK are external prerequisites. Third-party rights are detailed in 来源声明.md. Original entrant code is provided for contest review with rights reserved.\n")
    sizes={"01":50,"02":50,"03":50,"04":10,"05":10,"06":10,"07":50}
    manifest={}
    for p in sorted(OUT.iterdir()):
        if not p.is_file() or p.name[:2] not in sizes:continue
        assert p.stat().st_size<sizes[p.name[:2]]*1000000
        manifest[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    image=ROOT/"work/delivery/harness.tar"
    assert hashlib.sha256(image.read_bytes()).hexdigest()==HASH
    manifest["harness.tar"]=dict(bytes=image.stat().st_size,sha256=HASH)
    (OUT/"交付清单.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    status="02 已填入经核实的 Google Drive 分享链接。" if args.image_share_url else "02 需补评委可下载的 Google Drive 分享链接。"
    (OUT/"提交说明.txt").write_text("本目录对应图示七项材料，按编号上传同名字段。"+status+"镜像原件为 ../delivery/harness.tar。试运行原始报告在 ../platform/trial-report.txt，赛方页面需授权登录。正式成绩以赛方发布为准。\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
