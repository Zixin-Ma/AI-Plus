"""Assemble the enterprise V2 technical report, replay and reproducibility bundles."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import zipfile

import pymupdf as pdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE.parent / "work/v2"
DELIVERY = WORK / "delivery"
VERSIONS = ("v1", "v2")
TASKS = "ABC"
METRICS = {"A":"Macro-F1 ↑", "B":"RMSE 分钟 ↓", "C":"RMSE 循环 ↓"}
TITLE = "漂移流上的受控工业模型续训"
CSS = """
*{font-family:sans-serif;color:#102e42}
h1{font-size:27px;margin:0 0 14px;color:#0c3742}
h2{font-size:17px;margin:13px 0 4px;color:#086758}
p{font-size:15px;line-height:1.47;margin:6px 0 10px}
table{border-collapse:collapse;width:100%;margin:7px 0 11px}
th,td{font-size:13.8px;text-align:left;padding:5px 6px;border-bottom:1px solid #c9d9d9}
th{background:#dceee7;color:#154d40}
a{color:#056e62}.note{font-size:12.4px;color:#3d5665}
"""
def read(path):
    return json.loads(path.read_text())


def table(labels, rows):
    def tr(tag,items):
        cells="".join(f"<{tag}>{html.escape(str(s))}</{tag}>" for s in items)
        return "<tr>"+cells+"</tr>"
    return "<table>"+tr("th",labels)+"".join(tr("td",row) for row in rows)+"</table>"


def para(s, style=""):
    return f'<p class="{style}">{s}</p>'


def page(doc, section, title, body):
    p=doc.new_page(width=842,height=560)
    p.draw_rect(p.rect,fill=(.986,.986,.971),color=None)
    p.draw_rect(pdf.Rect(0,0,11,560),fill=(.025,.35,.31),color=None)
    p.insert_text((36,30),"BJTU   /   INDUSTRIAL AI                         ENTERPRISE V2",fontname="hebo",fontsize=10.5,color=(.09,.38,.34))
    p.insert_text((595,30),section,fontname="hebo",fontsize=10.5,color=(.09,.38,.34))
    p.draw_line((36,42),(805,42),color=(.69,.77,.74))
    remaining,scale=p.insert_htmlbox(pdf.Rect(36,58,806,519),f"<h1>{html.escape(title)}</h1>"+body,css=CSS)
    if remaining<0 or scale<.84:
        raise ValueError(f"page {len(doc)} text exceeds layout; scale {scale:.3f}")
    p.draw_line((36,527),(805,527),color=(.71,.79,.75))
    p.insert_text((36,546),"TECHNICAL EVIDENCE    /    McCarthy",fontsize=9,color=(.27,.38,.42))
    p.insert_text((770,546),f"{len(doc):02}",fontname="hebo",fontsize=10,color=(.12,.30,.34))


def files_for_runs():
    return [WORK/"replay"/f"{s}-{v}-{k}-{n}.json" for s,ns in (("drift",range(5)),("official",range(3)))
            for v in VERSIONS for k in TASKS for n in ns]


def load_records():
    runs={p.stem:read(p) for p in files_for_runs()}
    for name,r in runs.items():
        if [z["round"] for z in r["rounds"]]!=list(range(1,11)):
            raise ValueError(f"incomplete replay {name}")
    live={}
    for k in TASKS:
        p=WORK/"live"/k
        record=read(p/"round.json")
        calls=[read(p/f"call-{i}.json") for i in (1,2)]
        if [c["id"] for c in calls] != record["model_call_ids"]:
            raise ValueError(f"unmatched live receipts for {k}")
        live[k]=(record,calls)
    return runs,live


def subset(runs,version,splits,setting="drift",tasks=TASKS):
    return [runs[f"{setting}-{version}-{k}-{s}"] for k in tasks for s in splits]


def stats(runs):
    flat=[r for stream in runs for r in stream["rounds"]]
    return {**{k:statistics.mean(r[k] for r in runs) for k in ("mean","worst","forget","delay")},
            "min":min(r["worst"] for r in runs),"cpu":sum(r["cpu"] for r in flat),
            "calls":sum(r["calls"] for r in flat),"prompt":sum(r["prompt_bytes"] for r in flat),
            "mb":sum(r["checkpoint_bytes"] for r in flat)/1e6}


def pair(a,b,digits=2,mult=1):
    return f"{a*mult:.{digits}f} → {b*mult:.{digits}f}"


def comparison(name,a,b):
    return [name,pair(a["mean"],b["mean"],1,100),pair(a["worst"],b["worst"],1,100),
            pair(a["forget"],b["forget"],1,100),pair(a["delay"],b["delay"],2)]


COLS=["任务 / 拆分","平均增益 % ↑","最低增益 % ↑","历史回落点 ↓","延迟轮 ↓"]


BASE=(.52,.42,.31)
NEW=(.025,.41,.33)
GRID=(.78,.83,.81)

def chart_header(p,y,title):
    p.insert_htmlbox(pdf.Rect(38,y,552,y+25),f"<b>{title}</b>",css="*{font-family:sans-serif;font-size:13px;color:#10564b}")
    for x,v,color in [(665,"V1",BASE),(735,"V2",NEW)]:
        p.draw_line((x,y+10),(x+19,y+10),color=color,width=2.4)
        p.insert_text((x+24,y+14),v,fontsize=10,color=color,fontname="hebo")


def overview_chart(p,runs):
    chart_header(p,369,"每项任务的最差阶段收益  ·  两条冻结流等权（%）")
    pairs=[[stats(subset(runs,v,(3,4),tasks=k))["worst"]*100 for v in VERSIONS] for k in TASKS]
    lower=min(-5,(int(min(map(min,pairs))//5))*5)
    upper=max(5,(int(max(map(max,pairs))//5)+1)*5)
    def px(x):
        return 125+(x-lower)/(upper-lower)*593
    for val in range(lower,upper+1,5):
        x=px(val)
        p.draw_line((x,398),(x,478),color=GRID if val else (.35,.47,.43),width=.65)
        p.insert_text((x-7,496),f"{val}",fontsize=10,color=(.23,.34,.35))
    for i,(k,pts) in enumerate(zip(TASKS,pairs)):
        y=408+30*i
        p.insert_text((45,y+4),k,fontname="hebo",fontsize=13,color=NEW)
        x1,x2=map(px,pts)
        p.draw_line((x1,y),(x2,y),color=(.5,.64,.58),width=2.3)
        p.draw_circle((x1,y),4,color=BASE,fill=(.986,.986,.971),width=1.5)
        p.draw_circle((x2,y),3,color=NEW,fill=NEW)
        p.insert_text((735,y+4),f"{pts[1]:+.1f}%",fontsize=11,color=NEW)


def trajectories(p,runs,n):
    chart_header(p,388,f"拆分 {n} 逐轮相对收益（%）  ·  三图共用纵轴，每个点代表一轮")
    series=[[[100*z["stage_gain"] for z in runs[f"drift-{v}-{k}-{n}"]["rounds"]] for v in VERSIONS] for k in TASKS]
    low=min(-10,10*int(min(min(points) for sets in series for points in sets)//10))
    high=max(10,10*(int(max(max(points) for sets in series for points in sets)//10)+1))
    def py(val):
        return 495-(val-low)/(high-low)*67
    for j,(k,sets) in enumerate(zip(TASKS,series)):
        left=65+j*249;right=left+206
        p.insert_text((left,422),k,fontname="hebo",fontsize=11,color=NEW)
        for value in (low,0,high):
            y=py(value)
            p.draw_line((left,y),(right,y),color=GRID,width=.6)
            p.insert_text((left-24,y+3),str(value),fontsize=8,color=(.30,.39,.39))
        for points,color in zip(sets,(BASE,NEW)):
            xy=[pdf.Point(left+i*(right-left)/9,py(z)) for i,z in enumerate(points)]
            p.draw_polyline(xy,color=color,width=1.6)
            for point in xy:
                p.draw_circle(point,1.4,color=color,fill=color)
        p.insert_text((left,509),"1",fontsize=8,color=BASE)
        p.insert_text((right-4,509),"10",fontsize=8,color=BASE)


def report(runs,live):
    d=pdf.open()
    six=[stats(subset(runs,v,(3,4))) for v in VERSIONS]
    a,b=six
    worst_shift=min(((q["worst"]-p["worst"],k,i) for k in TASKS for i in (3,4)
                       for p,q in [(runs[f"drift-v1-{k}-{i}"],runs[f"drift-v2-{k}-{i}"])]))
    page(d,"01   /   OVERVIEW",TITLE,
         para("企业赛 V2 以企业 V1 为配对基线。研究目标是提升轮换工况的平均及最低阶段表现，同时控制旧阶段性能回落、恢复时间和运行开销。")+
         table(["冻结后复验","企业 V1","企业 V2"],[
             ["六条流等权平均阶段增益",f'{100*a["mean"]:.2f}%',f'{100*b["mean"]:.2f}%'],
             ["六条流中最低的单个阶段",f'{100*a["min"]:.2f}%',f'{100*b["min"]:.2f}%'],
             ["每条流最大历史回落的均值",f'{100*a["forget"]:.2f} 个百分点',f'{100*b["forget"]:.2f} 个百分点']])+
         para("每项任务采用两个新冻结顺序，各执行十轮。逐轮记录、模型回执与可复验源码均随包提供。公开代理指标采用 Macro-F1 或 RMSE 对静态起点的相对变化，赛事客观分和专家分由平台另行评定。")+
         para("应用场景包括气体识别、旋转设备维护和发动机健康评估。三项任务共用同一个 CPU 检查点续训与候选裁决流程。","note"))
    overview_chart(d[-1],runs)
    page(d,"02   /   SYSTEM","每次更新都要通过当前工况的测量",
         table(["执行点","算法与部署约束"],[
            ["接收父产物","从官方父检查点继续更新 MLP 与 Adam。训练与归一化参数来自已释放样本。"],
            ["提出训练工具","官方 Qwen3.8-27B 读取公开漂移与历史诊断，在重拟合、保留任务头续训、单步 Adam 中作出选择。协议允许在稳定投影上复用既有计划。"],
            ["任务建模","A 比较气体动力学条件曲线与旧分类器。B 以双轴谱阶次、寿命覆盖和信号支持控制修正强度；C 用工况内归一化和有界神经残差预测寿命。"],
            ["决定是否部署","用同一公开验证面板比较父模型、候选、静态参考和允许恢复的历史产物。B 在新增训练行时还限制寿命四分位的预测移动。宿主验证反馈和模型裁决均写入记录。"]])+
         para("预测以单条授权输入工作，训练、选型与审计分开。未知工具、无效输出和保护条件未通过时保留可用父产物。")+
         para('<a href="https://www.mdpi.com/2227-9709/13/1/15">时序气体漂移研究</a> · <a href="https://github.com/scikit-learn/scikit-learn">scikit-learn 判别分析</a> · <a href="https://github.com/McDonnell-Research-Lab/RanPAC">RanPAC 特征更新对照</a> · <a href="https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.DX.2025.9">跨设备寿命诊断</a> · <a href="https://arxiv.org/abs/2611.28759">C-MAPSS 工况归一化研究</a>','note'))
    page(d,"03   /   PROTOCOL","五项评分方向的公开证据",
         table(["方向","代理指标与统计方法","冻结复验 V1 → V2"],[
             ["漂移流综合","每轮相对静态改善率，先逐流平均、再等权",pair(a["mean"],b["mean"],2,100)+" %"],
             ["最差阶段","每条十轮流先取最低，再等权平均",pair(a["worst"],b["worst"],2,100)+" %"],
             ["恢复速度","切换后到相对收益非负的轮数，在阶段末截尾",pair(a["delay"],b["delay"])+" 轮"],
             ["稳定与抗遗忘","同一已见阶段相对其历史最佳增益的最大回落",pair(a["forget"],b["forget"],2,100)+" 个百分点"],
             ["资源效率","60 轮本地训练协议 CPU 合计及控制调用次数",pair(a["cpu"],b["cpu"],1)+" 秒 / "+pair(a["calls"],b["calls"],0)+" 次"]])+
         para("拆分 0、1、2 用于开发，3、4 于配置冻结后执行。训练从每组时序的前 55% 逐轮扩大到 100%，两个版本从相同首批样本重新建起点。每条流内，选型和审计行互斥，C 的审计按整机划分。不同拆分会复用公开样本。")+
         para("CPU 回放执行实际训练、产物序列化和模拟宿主复算。规划与评审使用受控回复，实际 27B 首轮用量单列。","note"))
    rows=[]
    for i in (3,4):
        for k in TASKS:
            rows.append(comparison(f"{k} / {i}",*(runs[f"drift-{v}-{k}-{i}"] for v in VERSIONS)))
    delta,k,i=worst_shift
    b4=[runs[f"drift-{v}-B-4"]["mean"] for v in VERSIONS]
    a4=[runs[f"drift-{v}-A-4"]["forget"] for v in VERSIONS]
    page(d,"04   /   CONFIRMATION","六条冻结流逐项配对",
         table(COLS,rows)+para(f"B / 4 的平均阶段收益为 {pair(*b4,2,100)} %。A / 4 的历史回落为 {pair(*a4,2,100)} 个百分点。最低阶段改善最少的配对为 {k} / {i}，差值为 {100*delta:+.2f} 个百分点。")+
         para("A 的增益按 Macro-F1 上升方向计算。B/C 使用 RMSE 下降方向。表内百分数均相对同一拆分的静态起点。","note"))
    trajectories(d[-1],runs,4)
    dev=[comparison(k,*[stats(subset(runs,v,(0,1,2),tasks=k)) for v in VERSIONS]) for k in TASKS]
    fixed=[stats(subset(runs,v,(0,1,2),setting="official")) for v in VERSIONS]
    official=[comparison(k,*[runs[f"official-{v}-{k}-1"] for v in VERSIONS]) for k in TASKS]
    page(d,"05   /   SENSITIVITY","训练池变化与起点选择影响表现",
         "<h2>三个开发顺序等权，随轮次释放训练行</h2>"+table(COLS,dev)+
         "<h2>官方完整起点，固定训练池，验证顺序 1</h2>"+table(COLS,official)+
         para(f'官方固定池九条流等权，平均增益 {pair(*[r["mean"] for r in fixed],2,100)}%，历史回落 {pair(*[r["forget"] for r in fixed],2,100)} 点，恢复 {pair(*[r["delay"] for r in fixed],3)} 轮。',"note"))
    ga=read(WORK/"gas/public.json")
    be=read(WORK/"bearing/loo.json")["summary"]
    route="narrow-0.7-0.5"
    en=[read(WORK/f"engine/fold{i}-confirmation.json") for i in (1,2)]
    page(d,"06   /   ABLATIONS","三类任务的选择依据",
         table(["任务","实际比较","独立模型证据"],[
             ["A","分别检验曲线、类内子空间、V1 核岭与双头选型。当前验证表现决定使用的头。","完整公开首轮，独立 V1 与曲线头 F1 为 "+pair(ga["official"]["v1"]["macro"],ga["official"]["v2"]["macro"],4)+"。四折中最低值为 "+pair(*[min(f[v]["macro"] for f in ga["folds"]) for v in VERSIONS],4)+"。"],
             ["B","比较原有校正、不同信号支持阈值、方向性寿命修正和参数压缩，逐轴承重新拟合。","整轴承折等权 RMSE 为 "+pair(be["v1"]["mean"],be[route]["mean"],1)+" 分钟，最差设备为 "+pair(be["v1"]["worst"],be[route]["worst"],1)+" 分钟。"],
             ["C","比较冻结读出、周期性重拟合、权重平均及隐藏层宽度和训练轮数。","两个设备留出折最终 RMSE 分别为 "+"、".join(pair(*[z[v][-1]["rmse"] for v in VERSIONS],2) for z in en)+" 循环。"]])+
         para("A 的完整训练池可优先采用旧头。B 的整体与最差设备误差可能不同向。C 同时记录近失效阶段、工况、最差子集及 NASA 非对称代价。每项的具体路线和未采用试验见源码压缩包内实验说明。"))
    g=read(WORK/"gas/cross-day.json");ds=read(WORK/"gas/ds-ps.json")
    ext=read(WORK/"bearing/external.json");ka=read(WORK.parent/"bearing/kaist-shift.json")
    c=read(WORK/"engine/DS02-confirmation.json")
    page(d,"07   /   TRANSFER","六份补充材料在什么条件下有效",
        table(["数据","独立测量","V1 → V2 或漂移量"],[
            ["一年漂移电子鼻",f'{g["train"]} 条训练、{g["test"]} 条后续采集日测试；按前序日选择分支。',pair(g["results"]["v1"]["macro"],g["results"]["heldout_choice"]["macro"],4)+" Macro-F1"],
            ["DS-PS",f'{ds["train"]} 条训练、{ds["test"]} 条未见重复；各非零浓度整体划分。',pair(ds["results"]["v1"]["macro"],ds["results"]["heldout_choice"]["macro"],4)+" Macro-F1"],
            ["FEMTO / PRONOSTIA",f'{len(ext["folds"])} 根轴承分别作为留出设备。',pair(ext["summary"]["v1"]["mean"],ext["summary"][route]["mean"],2)+" 分钟"],
            ["KAIST",f'{ka["records"]} 个设备日志抽样，比较早期与晚期的幅度分组。',f'最低幅度区间 {ka["early_counts"][0]}/{sum(ka["early_counts"])} → {ka["late_counts"][0]}/{sum(ka["late_counts"])}'],
            ["N-CMAPSS 与 CruiseBench",f'DS02 原生传感器独立拟合，{c["v2"][-1]["n"]:,} 条留出记录；巡航掩码与原始轨迹同源。',"整体 "+pair(*[c[v][-1]["rmse"] for v in VERSIONS],3)+"；巡航 "+pair(*[c[v][-1]["cruise"] for v in VERSIONS],3)]])+
        para("外部数据采用各仪器原生特征与设备划分。跨日电子鼻出现选型失准，KAIST 仅支持信号漂移观察。比赛镜像只收录运行源码、SDK 和依赖。","note")+
        para('<a href="https://zenodo.org/records/15681119">一年电子鼻</a> · <a href="https://zenodo.org/records/19557093">DS-PS</a> · <a href="https://data.phmsociety.org/nasa/">PHM 数据仓库</a> · <a href="https://data.mendeley.com/datasets/5hcdd3tdvb/6">KAIST</a> · <a href="https://github.com/NostalgiaJohn/CruiseBench">CruiseBench</a>','note'))
    rows=[]
    for k,(rec,calls) in live.items():
        cfg=rec["configuration"];sc=cfg["scores"]
        rows.append([k,METRICS[k],pair(sc["parent"]["primary"],sc[cfg["deployed"]]["primary"],4),
                     cfg["deployed"],sum(c["usage"]["total_tokens"] for c in calls)])
    booked={key:r for k in TASKS for key,r in receipts(k).items()}
    billed=sum(r["usage"]["total_tokens"] for r in booked.values())
    page(d,"08   /   GATEWAY","真实 27B 公开首轮与计算成本",
        table(["任务","目标","父模型 → 实际采用","裁决","实际 Token"],rows)+
        para(f"三项均从赛方公开起点运行生产训练入口，保存候选后独立读取权重复算。每项规划、裁决各对应一份发布版回执。A、C 在规划格式修正前各产生一次额外调用，全部请求按回执 ID 去重。整个公开验证实际共 {len(booked)} 次调用、{billed:,} Token。相同请求复用评审回执。")+
        table(["六条冻结流，60 轮合计","企业 V1","企业 V2"],[
            ["CPU 进程时间（秒）",f'{a["cpu"]:.2f}',f'{b["cpu"]:.2f}'],
            ["候选检查点上传量（十进制 MB）",f'{a["mb"]:.2f}',f'{b["mb"]:.2f}'],
            ["输入提示文本字节数",f'{a["prompt"]:,}',f'{b["prompt"]:,}']])+
        para("CPU 时间包含本地宿主复算及序列化。线上排队、实际网络和平台预算扣分由主办方的计量系统决定。","note"))
    controls=[]
    for tag in ("unlimited","class04","class09","class16","always_plan"):
        series=[read(f) for f in sorted((WORK/"controller"/tag).glob("*.json"))]
        t=stats(series)
        controls.append([tag,pair(t["mean"],t["worst"],1,100),f'{t["forget"]*100:.1f}',t["calls"]])
    page(d,"09   /   COMPUTE","单 CPU 交替复测与控制器消融",
        table(["任务","十轮 CPU 秒","单样本预测 µs","十轮检查点 MB","调用次数"],resource_rows())+
        para("每任务固定一个逻辑核，顺序在 V1/V2 间交替，三次重复取中位数。字节与调用次数取首次执行。表内为企业 V1 → V2。","note")+
        table(["开发路线","均值 / 最低均值 %","平均历史回落点","三条流总调用"],controls)+
        para("前四项在 A 改变单类下降阈值。部署保留 unlimited 和公共主指标与组别守门。always_plan 在 C 每轮重新规划，部署方案可复用既有计划。","note"))
    page(d,"10   /   OPERATIONS","上线、异常恢复与工业使用",
        "<h2>运行条件与复现</h2>"+para("Linux/amd64，NumPy 2.2.6，SDK 1.0.4。以 SHA256SUMS.txt 核对镜像与材料。镜像默认普通用户，启动入口 python3 /app/harness.py，支持 describe、train、predict。训练读取宿主投影，预测流逐条响应。")+
        "<h2>部署检查</h2>"+para("输入必须符合任务特征和有限数值条件。SDK 闭环七项测试、训练集替换和轴承快速路径等三项专项测试均通过。正式镜像以 SDK 模拟宿主逐任务运行两轮，覆盖训练、候选上传、回执和流式预测。规划结果经过工具白名单解析，历史恢复仅使用宿主允许访问且已复核的产物。异常输入由宿主留痕并触发处置。")+
        "<h2>现场边界</h2>"+para("电子鼻需重新核查同型仪器的弱类召回和浓度范围。轴承寿命保护依赖转速、载荷及振动分布。发动机需同时监控晚期误差和高估寿命的非对称损失。维护、停机和告警阈值由设备负责人结合失效后果确认。")+
        para("各轮证据在公开实验证据压缩包中，研究说明、依赖及运行脚本在源码包中。","note"))
    return d
def receipts(k):
    p=WORK/"live"/k
    ordered=[read(z) for z in sorted(p.glob("call-*.json"))+sorted(p.glob("prior-*.json"))]
    return {r["id"]:r for r in ordered}


def resource_rows():
    rows=[]
    for k in TASKS:
        versions=[[read(WORK/"resources"/str(i)/f"drift-{v}-{k}-0.json") for i in range(3)] for v in VERSIONS]
        T=[[stats([r])["cpu"] for r in series] for series in versions]
        U=[[r["prediction_us"] for r in series] for series in versions]
        rows.append([k,pair(*map(statistics.median,T)),pair(*map(statistics.median,U),1),
                     pair(*[stats([series[0]])["mb"] for series in versions],2),
                     pair(*[stats([series[0]])["calls"] for series in versions],0)])
    return rows


def replay_video(runs,live,path,stage):
    d=pdf.open()
    six=[stats(subset(runs,v,(3,4))) for v in VERSIONS]
    page(d,"REPLAY   /   START","公开运行回放  ·  十轮漂移与模型裁决",
        para("逐帧重建企业 V2 公开训练过程。真实 27B 首轮、冻结拆分 3 的逐轮审计和拆分 4 的独立复验分别取自原始记录。")+
        table(["复验范围","V1 → V2"],[
            ["六条冻结流的平均阶段增益",pair(*[r["mean"] for r in six],2,100)+" %"],
            ["六条流各自最低增益的均值",pair(*[r["worst"] for r in six],2,100)+" %"],
            ["每条流最大回落的均值",pair(*[r["forget"] for r in six],2,100)+" 个百分点"]])+
        para("每轮的采纳对象、CPU 和当前时段指标可在随附 JSON 复查。完整回放包含全部任务，共 48 条公开十轮记录。"))
    labels={"candidate":"本轮候选","parent":"沿用父模型","history":"恢复历史"}
    for k in TASKS:
        rec,calls=live[k]
        cfg=rec["configuration"]
        page(d,"REPLAY   /   LIVE 27B",f"{k}  ·  真实模型与公开宿主完成第一轮",
             table(["执行步骤","记录"],[
                 ["训练计划",cfg["method"]+" / "+cfg["plan"]["strength"]],
                 ["检查点",f'Adam 步数 {cfg["old_steps"]} → {cfg["steps"]}'],
                 ["公开样本",f'训练 {cfg["train_rows"]} 行，验证 {cfg["validation_rows"]} 行'],
                 ["部署裁决",labels.get(cfg["deployed"],cfg["deployed"])],
                 [METRICS[k],pair(cfg["scores"]["parent"]["primary"],cfg["scores"][cfg["deployed"]]["primary"],4)],
                 ["发布版规划 / 裁决用量"," / ".join(str(x["usage"]["total_tokens"]) for x in calls)+" Token"]])+
             para("真实回执对应公开首轮。接下来的十轮使用受控大模型回复和独立审计行，实际执行两版模型训练与宿主复算。","note"))
        for start in (0,5):
            r=runs[f"drift-v2-{k}-3"]; v1=runs[f"drift-v1-{k}-3"]
            rows=[]
            for z,old in zip(r["rounds"][start:start+5],v1["rounds"][start:start+5]):
                idx=z["stage"]
                rows.append([z["round"],idx,labels.get(z["deployed"],z["deployed"]),
                             f'{z["static"][idx]:.3f} / {z["metrics"][idx]:.3f}',
                             pair(old["stage_gain"],z["stage_gain"],1,100),f'{z["forget"]*100:.1f}'])
            page(d,"REPLAY   /   FROZEN STREAM",f"{k}  ·  冻结拆分 3 的第 {start+1}–{start+5} 轮",
                 table(["轮次","阶段","部署","静态 / V2 审计值","相对收益 %  V1 → V2","回落点"],rows)+
                 para(f'原指标为 {METRICS[k]}。阶段编号 0–3，每一行只比较同轮同阶段。回落点取已见阶段现状相对其历史最高相对收益的最大差。')+
                 para("逐轮标签来自审计集，训练、参数选择和当轮裁决使用各自获准的数据。","note"))
    page(d,"REPLAY   /   SECOND STREAM","冻结拆分 4  ·  另一条顺序的结论",
         table(COLS,[comparison(k,*[runs[f"drift-{v}-{k}-4"] for v in VERSIONS]) for k in TASKS])+
         para("拆分 3 与 4 在生产代码及评价器冻结后运行。训练、配置及审计证据分别位于源码、模型配置与公开实验证据包。平台的隐藏综合分需经正式评测取得。"))
    frames=stage/"frames";frames.mkdir()
    for i,p in enumerate(d):
        p.get_pixmap(matrix=pdf.Matrix(1.6,1.6),alpha=False).save(frames/f"{i:02}.png")
    d.close()
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-framerate","1/7",
                    "-i",str(frames/"%02d.png"),"-vf","scale=1280:-2,fps=12,format=yuv420p","-an",
                    "-c:v","libx264","-threads","2","-preset","veryfast","-crf","21",
                    "-movflags","+faststart",str(path)],check=True)
    duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=nw=1:nk=1",str(path)],text=True).strip())
    if not 60<=duration<300:
        raise ValueError(f"unexpected video duration {duration}")
MODEL_CONFIG = """企业赛 V2 模型配置
镜像标签 bjtu-rsi:enterprise-v2
架构 Linux/amd64，Python 3.12，NumPy 2.2.6，Harness SDK 1.0.4，CPU 单线程，容器用户 65534:65534。
协议 contest.harness.workspace.v2。宿主通过 host_gateway_rpc 提供 Qwen/Qwen3.8-27B。
解码 temperature=0，enable_thinking=false。规划最多 56 个输出 Token，裁决最多 32，均受宿主限额约束。稳定且满足条件时复用既有规划。
训练从宿主给出的父产物加载 MLP、归一化及 Adam 状态，按获准输入执行重拟合、沿用任务校正头续训或骨干 Adam 更新。候选与父模型在当前可见验证投影上测量。
A 六类气体分数。候选包括 V1 分类头及气体动力学曲线与类内子空间。
B 以分钟为单位输出非负轴承剩余寿命。用振动阶次和寿命支持控制校正，新增数据时限制寿命分位预测移动。
C 以循环数输出非负发动机剩余寿命。按工况归一化，同一训练池与新增训练池使用不同校正头调度。
镜像入口 python3 /app/harness.py，命令 describe、train、predict。训练和推理的输入、回执、历史和工作目录由赛方 SDK 提供。
"""

EVIDENCE_GUIDE = """公开实验证据索引
replay 目录有 48 条配对十轮记录。drift 按每组公开训练集的 55% 到 100% 逐轮释放，测试拆分为 0 至 4。official 用原始公开预训练检查点和固定完整训练池，拆分为 0、1、2。每条流中训练、当前验证选型和审计用途分开。拆分 3、4 在生产代码与评价器冻结后运行。
freeze.json 为冻结代码文件 SHA-256 和复验顺序。可与源码压缩包中 企业赛/V2 的同名文件逐字节核对。
每轮 stage 指阶段，static/metrics/gains 为各个审计阶段的静态值、部署值和方向统一的相对收益，stage_gain 是当前阶段的值。mean/worst 是流内十轮均值/最低点，forget 是已见阶段较各自历史峰值的最大收益回撤，delay 是切换后重新获得非负收益的平均等待轮数，按阶段末截尾。
resources 有 3 次重复 × 3 项任务 × 两版单 CPU 交替复测。prediction_us 包含单条预测与 JSON 编码。cpu、calls、prompt_bytes、checkpoint_bytes 在 rounds 内逐轮报告。模型协议模拟次数与真实网关 Token 单独计量。
live 是三项公开起点的真实 27B 首轮，每个任务有 round.json、规划 call-1.json、评审 call-2.json。A 和 C 的 prior-*.json 保留格式调整前的真实请求；付费计量按 id 去重。first-attempt-round.json 是执行记录。回执将提示正文映射为 digest。
gas、bearing、engine 包含开发消融与外部评估。legacy/kaist-shift.json 是既有 KAIST 信号切分观察。controller 包含五种开发控制策略。命名、单位、样本划分及第三方数据原始出处见源码包各分项实验说明。通过主实验外部结果文件复算报告第七页。
image-sdk-0.log 对正式镜像展开并以SDK模拟宿主调用 describe/train/predict，每项两轮。final-check-3.log 为闭环单测。frozen-cost-4.log 为控制测试（若存在）。公开回放和复验均未使用隐藏测试样本。
"""

def make_bundles(source,evidence):
    # Preserve repository-relative paths needed by paired evaluation and public preparation.
    source_files=set()
    for base in (HERE,HERE.parent/"V1"):
        source_files.update(p for p in base.iterdir() if p.is_file() and p.suffix in (".md",".py",".txt"))
    campus=ROOT/"校赛"
    for name in ("prepare.py","README.md","来源声明.md","V2/bearing.py","V2/gas.py","V2/engine.py",
                 "app/baseline_model.py","app/industrial.py"):
        source_files.add(campus/name)
    with zipfile.ZipFile(source,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(source_files):
            archive.write(p,p.relative_to(ROOT))
        archive.writestr("复现起点.txt",
            "请先阅读 企业赛/V2/README.md 及分项实验说明。上游 SDK 与授权训练起点由赛方提供。"
            "校赛/prepare.py 生成公有训练、验证和官方父检查点。基础镜像和 wheel 按校赛 README 准备。"
            "运行依赖为 Python、NumPy、比赛 SDK。开发迁移试验另需 SciPy、h5py，报告与视频需 PyMuPDF、FFmpeg。"
            "本包保留团队源码、上游适配器与出处。Python、Debian 和 NumPy 的上游署名与分发材料保存在镜像对应软件目录。")
    paths=set(files_for_runs())
    for name in ("resources","controller","gas","bearing","engine"):
        for p in (WORK/name).rglob("*.json"):
            if p.name.startswith(("drift-","official-")) or p.name in {
                    "cross-day.json","ds-ps.json","public.json","deployed.json","external.json","guarded.json",
                    "loo.json","DS02-confirmation.json","fold1-confirmation.json","fold2-confirmation.json"}:
                paths.add(p)
    for name in ("image-sdk-0.log","final-check-3.log","freeze.json","frozen-cost-4.log"):
        paths.add(WORK/name)
    for k in TASKS:
        p=WORK/"live"/k
        paths.update(p/n for n in ("round.json","call-1.json","call-2.json"))
        paths.update(p.glob("prior-*.json"))
        paths.update(p.glob("first-attempt-round.json"))
    with zipfile.ZipFile(evidence,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.write(p,p.relative_to(WORK))
        archive.write(WORK.parent/"bearing/kaist-shift.json","legacy/kaist-shift.json")
        archive.writestr("复核说明.txt",EVIDENCE_GUIDE)
    for path in (source,evidence):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip():
                raise ValueError(f"invalid archive {path}")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    runs,live=load_records()
    freeze=read(WORK/"freeze.json")
    for name,hexdigest in freeze["production_and_evaluator"].items():
        if digest(HERE/name)!=hexdigest:
            raise ValueError(f"changed after freeze: {name}")
    cfg=HERE/"模型配置.txt"
    if cfg.read_text()!=MODEL_CONFIG:
        raise ValueError("model config needs updating")
    DELIVERY.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=WORK) as temp:
        stage=Path(temp)
        report_path=stage/"企业赛V2技术报告.pdf"
        d=report(runs,live)
        if len(d)>10:
            raise ValueError("report exceeds ten pages")
        d.set_metadata({"title":TITLE+" · 企业赛 V2","author":"McCarthy","subject":"五项公开指标与冻结顺序实证"})
        d.subset_fonts()
        d.save(report_path,garbage=4,deflate=True)
        d.close()
        with pdf.open(report_path) as check:
            for p in check:
                if len(p.get_text().strip())<120:
                    raise ValueError(f"empty page {p.number+1}")
        if report_path.stat().st_size>10_000_000:
            raise ValueError("report exceeds size limit")
        movie=stage/"企业赛V2运行回放.mp4"
        replay_video(runs,live,movie,stage)
        source=stage/"企业赛V2源码.zip";evidence=stage/"公开实验证据.zip"
        make_bundles(source,evidence)
        if max(source.stat().st_size,evidence.stat().st_size)>50_000_000:
            raise ValueError("archive exceeds 50 MB")
        config=stage/cfg.name;shutil.copyfile(cfg,config)
        outputs=[report_path,movie,source,evidence,config]
        image=DELIVERY/"harness.tar"
        sums=stage/"SHA256SUMS.txt"
        sums.write_text("".join(f"{digest(f)}  {f.name}\n" for f in [image,*outputs]))
        for f in [*outputs,sums]:
            f.replace(DELIVERY/f.name)
            print(f.name,(DELIVERY/f.name).stat().st_size,flush=True)


if __name__=="__main__":
    main()
