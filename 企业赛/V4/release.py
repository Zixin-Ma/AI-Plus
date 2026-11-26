"""Build and verify enterprise V4 submission materials from the frozen public evidence."""
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
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=HERE.parent/"work/v4"
DELIVERY=WORK/"delivery"
TASKS="ABC"
VERSIONS=("v3","v4")
CASES=("drift","fresh","official","fresh-measured")
NAMES={"drift":"逐步释放","fresh":"回访换选型样本","official":"固定官方池","fresh-measured":"回访换样本＋按测量裁决"}
TASK_NAMES={"A":"气体识别","B":"轴承寿命","C":"发动机寿命"}
METRICS={"A":"Macro-F1","B":"RMSE（分钟）","C":"RMSE（循环）"}
BASE=(.54,.39,.22)
NEW=(.04,.40,.34)
GRID=(.77,.83,.80)
GREEN="#126856"
TITLE="工况切换下的工业模型续训"
CSS="""
*{font-family:sans-serif;color:#193238}
h1{font-size:26px;margin:0 0 12px;color:#103f3c}
h2{font-size:16px;margin:9px 0 2px;color:#126856}
p{font-size:14.5px;line-height:1.4;margin:4px 0 8px}
table{border-collapse:collapse;width:100%;margin:7px 0 9px}
th,td{font-size:13.4px;text-align:left;padding:4px 6px;border-bottom:1px solid #cddbd5}
th{background:#e0ede5;color:#154e40} a{color:#08604e}.note{font-size:12px;color:#425c60}
"""


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def table(cols,rows):
    def tr(tag,values):
        return "<tr>"+"".join("<"+tag+">"+html.escape(str(v))+"</"+tag+">" for v in values)+"</tr>"
    return "<table>"+tr("th",cols)+"".join(tr("td",r) for r in rows)+"</table>"


def para(text,style=""):
    return f'<p class="{style}">{text}</p>'


def page(doc,english,title,body):
    p=doc.new_page(width=842,height=560)
    p.draw_rect(p.rect,fill=(.985,.985,.968),color=None)
    p.draw_rect(pdf.Rect(0,0,10,560),fill=NEW,color=None)
    p.insert_text((34,30),"BJTU INDUSTRIAL AI     /     ENTERPRISE V4",fontname="hebo",fontsize=10,color=NEW)
    p.insert_text((584,30),english,fontname="hebo",fontsize=10,color=NEW)
    p.draw_line((34,42),(806,42),color=GRID)
    space,scale=p.insert_htmlbox(pdf.Rect(34,57,806,514),"<h1>"+html.escape(title)+"</h1>"+body,css=CSS)
    if space<0 or scale<.85:
        raise ValueError(f"Page {len(doc)} does not fit: {scale}")
    p.draw_line((34,523),(806,523),color=GRID)
    p.insert_text((34,544),"McCarthy   /   PUBLIC EVALUATION",fontsize=9,color=NEW)
    p.insert_text((773,544),f"{len(doc):02}",fontname="hebo",fontsize=10,color=NEW)
    return p


def pair(a,b,scale=1,digits=2):
    return f"{a*scale:.{digits}f} → {b*scale:.{digits}f}"


def counts(s):
    return range(8) if s in ("drift","official") else range(3,8)


def load():
    summary=read(WORK/"paired-summary.json")
    freeze=read(WORK/"freeze.json")
    for name,digest in freeze["code"].items():
        if sha(HERE/name)!=digest:
            raise ValueError("Frozen file changed: "+name)
    if sha(WORK/"protocol.json")!=freeze["protocol"]:
        raise ValueError("Protocol mismatch")
    runs={}
    for section in ("dev","confirm"):
        for case in CASES:
            ids=counts(case) if section=="dev" else (8,9,10)
            for version in VERSIONS:
                batch=[]
                for k in TASKS:
                    for n in ids:
                        name=f"{case}-{version}-{k}-{n}"
                        r=read(WORK/"replay"/(name+".json"))
                        assert [z["round"] for z in r["rounds"]]==list(range(1,11)) and all(z["selection"]>0 and z["audit"]>0 for z in r["rounds"])
                        runs[name]=r
                        batch.append(r)
                stored=summary[section+"/"+case][version]["all"]
                for metric in ("mean","worst","forget","delay"):
                    assert abs(statistics.mean(r[metric] for r in batch)-stored[metric])<1e-10
                assert stored["streams"]==len(batch) and abs(min(r["worst"] for r in batch)-stored["lowest"])<1e-10
    live={}
    for k in TASKS:
        folder=WORK/"live"/k
        record=read(folder/"round.json")
        calls=[read(folder/f"call-{i}.json") for i in (1,2)]
        assert [r["id"] for r in calls]==record["model_call_ids"]
        live[k]=(record,calls)
    return summary,runs,live


def pair_stat(s,case,task,metric,scale=100,digits=2,section="confirm"):
    return pair(*[s[section+"/"+case][v][task][metric] for v in VERSIONS],scale,digits)


def comparison(s,case,task="all",section="confirm"):
    return [pair_stat(s,case,task,m,1 if m=="delay" else 100,3 if m=="delay" else 2,section)
            for m in ("mean","worst","forget","delay")]


def traces(p,runs,case,n,top=326):
    for j,(label,c) in enumerate([("V3",BASE),("V4",NEW)]):
        p.draw_line((625+j*68,top),(647+j*68,top),color=c,width=2)
        p.insert_text((652+j*68,top+3),label,fontsize=10,color=c)
    series=[[[100*z["stage_gain"] for z in runs[f"{case}-{v}-{k}-{n}"]["rounds"]] for v in VERSIONS] for k in TASKS]
    import math
    low=min(-10,10*math.floor(min(min(z) for task in series for z in task)/10))
    high=max(10,10*math.ceil(max(max(z) for task in series for z in task)/10))
    def y(val):
        return 480-(val-low)/(high-low)*(480-top-25)
    for j,(k,lines) in enumerate(zip(TASKS,series)):
        left=66+j*254;right=left+208
        p.insert_text((left,top+12),k,fontname="hebo",fontsize=12,color=NEW)
        for val in [low,0,high]:
            yy=y(val)
            p.draw_line((left,yy),(right,yy),color=GRID,width=.6)
            p.insert_text((left-27,yy+3),str(val),fontsize=8,color=NEW)
        for vals,c in zip(lines,[BASE,NEW]):
            xy=[pdf.Point(left+i*(right-left)/9,y(v)) for i,v in enumerate(vals)]
            p.draw_polyline(xy,color=c,width=1.55)
            for pt in xy:
                p.draw_circle(pt,1.35,color=c,fill=c)
        p.insert_text((left,497),"1",fontsize=9,color=NEW)
        p.insert_text((right-10,497),"10",fontsize=9,color=NEW)


def resource_rows():
    result=[]
    for k in TASKS:
        totals=[]
        for v in VERSIONS:
            group=[read(WORK/"resources"/str(i)/f"drift-{v}-{k}-0.json") for i in range(3)]
            totals.append([statistics.median(sum(z["cpu"] for z in r["rounds"]) for r in group),
                statistics.median(r["prediction_us"] for r in group),
                sum(z["checkpoint_bytes"] for z in group[0]["rounds"])/1e6,
                sum(z["calls"] for z in group[0]["rounds"])])
        result.append([k]+[pair(totals[0][i],totals[1][i],digits=0 if i==3 else 1 if i==1 else 2) for i in range(4)])
    return result



def selected_initial(runs, case, ids):
    return [[k, *[pair(*(runs[f"{case}-{v}-{k}-{n}"]["worst"] for v in VERSIONS),100,2) for n in ids]] for k in TASKS]


def end_to_end():
    rows=[]
    for key in TASKS:
        source=[]
        for version,mode in (("v3","legacy"),("v4","compiled")):
            trials=[read(WORK/"serving"/f"{key}-{version}-{version}-{i}.json")[mode] for i in range(5)]
            if not all(0 < r["n"] <= 600 for r in trials):raise ValueError("Empty serving sample")
            source.append([statistics.median(t[m] for t in trials) for m in ("p50_us","p95_us","cold_us","mean_us")])
        a,b=source
        rows.append([key,pair(a[0],b[0],digits=1),pair(a[1],b[1],digits=1),pair(a[2],b[2],.001,2),
                     pair(1e6/a[3],1e6/b[3],digits=0)])
    return rows



def sdk_measurements():
    rows=read(WORK/"sdk-latency.json")["rows"]
    result=[]
    for key in TASKS:
        pair_groups=[]
        for version in VERSIONS:
            subset=[r for r in rows if r["task"]==key and r["source"]==version and r["runtime"]==version]
            if len(subset)!=5:raise ValueError("Expected five paired SDK measurements for "+key)
            pair_groups.append([statistics.median(z[m] for z in subset) for m in ("p50_us","p95_us","cold_process_ms")])
        a,b=pair_groups
        result.append([key,pair(a[0],b[0],digits=1),pair(a[1],b[1],digits=1),pair(a[2],b[2],digits=2)])
    return result


def verify_release_inputs():
    verification=read(WORK/"verification.json")
    if verification["status"]!="passed" or verification["recorded_rounds"]!=2280 or verification["bearing"]["nonidentical_values"]!=0:
        raise ValueError("Independent replay check incomplete")
    smoke=(WORK/"image-proof-0.log").read_text()
    if not all(f"{k} round {i}:" in smoke for k in TASKS for i in (1,2,3)):
        raise ValueError("Three-round SDK image exercise missing")

def final_engine(dataset):
    record=read(WORK/"engine"/f"{dataset}-frozen-probe.json")
    return [max((r for r in record if r["route"]==route),key=lambda r:r["round"]) for route in ("base","frozen")]


def report(s,runs,live):
    d=pdf.open()
    names=["公开场景","平均收益 %","各流最低 %","历史回落点","恢复轮"]
    q=s["confirm/drift"]
    t=read(WORK/"bearing/confirmed.json")
    verified=read(WORK/"verification.json")
    confirmed_b=verified["bearing"]["paired_rounds"]
    page(d,"01   /   FROZEN RESULTS","企业赛 V4 · 冻结复验",
         para("三个工业任务共用一套十轮续训流程。气体识别针对跨日响应变化，发动机寿命侧重低谷阶段，轴承寿命以保持 V3 预测为约束压缩推理开销。")+
         table(["新留出主漂移，9 条流","V3","V4"],[[label,*[f'{q[v]["all"][metric]*factor:.{digits}f}' for v in VERSIONS]] for label,metric,factor,digits in
               (("平均阶段相对静态收益（%）","mean",100,2),("每条流最低轮收益均值（%）","worst",100,2),
                ("九条流的单一最低轮收益（%）","lowest",100,2),("历史峰值最大回落均值（百分点）","forget",100,2),
                ("漂移切换后恢复延迟（轮）","delay",1,3))])+
         table(["主漂移各流最低收益，V3 → V4（%）","顺序 8","顺序 9","顺序 10"],selected_initial(runs,"drift",(8,9,10)))+
         para("新顺序 8、9、10 调整阶段顺序和审计分块，复用既有公开数据行。主漂移的阶段低谷均值略升，最坏单轮持平，平均历史回落增大。赛事正式分数取决于平台隐藏流和裁决。","note"))
    page(d,"02   /   EXPERIMENT DESIGN","漂移评测如何区分平均收益与最差阶段",
         para("代理在每轮只读取已开放训练窗口和当前选型样本，经过诊断、宿主规划、增量训练与门控后部署。隔离审计块仅用于测量。分类收益按 Macro-F1 提升归一化，回归按 RMSE 降幅归一化，分母为同一静态起点。")+
         table(names,[[NAMES[c],*comparison(s,c)] for c in CASES])+
         para("逐步释放覆盖训练标签逐轮增加与验证阶段切换。回访更换样本防止记住重复选型块。固定官方池隔离选型分布变化。按测量裁决则把代理选择放到公开测量规则下复核。")+
         table(["已知顺序的开发回放","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[NAMES[c],*comparison(s,c,section="dev")] for c in ("drift","fresh")])+
         para("表内各组使用同一起点、相同顺序和相同选型/审计分隔。新留出各组均为 9 条流和 90 轮。开发主流使用 0–7；开发回访组使用 3–7。","note"))
    page(d,"03   /   ADAPTATION","控制器把工况诊断和部署决定分开",
         para("A 通过已释放样本诊断类别与信号可靠度，B 以振动工况和剩余寿命分位检查候选，C 以工况和寿命阶段限制过度延寿。三类更新都保留公开静态参考。")+
         table(["环节","使用的信息与动作"],[["诊断和规划","对已释放训练窗口和当前验证子群计分，向宿主 Qwen/Qwen3.8-27B 提交有界动作菜单。"],
          ["训练候选","从父产物恢复权重、归一化和 Adam 状态。可重拟合专家、沿用旧专家或只更新骨干。"],
          ["C 寿命补偿","当父产物当前公开最弱组落后静态起点，且三个可见寿命区间均低估时，允许残差绝对中位数 15% 的候选上限，按寿命 q20–q65 渐进开放。"],
          ["部署保护","比较父产物、候选与允许的历史/静态方案。主指标、子群下界、记忆探针和相对移动限制共同约束更新。"],
          ["离线推理","加载当轮采用的产物，预绑定数值算子，逐条返回六类概率、剩余分钟或剩余循环。"]])+
         table(["新留出主漂移","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[k+"  "+TASK_NAMES[k],*comparison(s,"drift",k)] for k in TASKS])+
         para("整流均值、最差轮和单次最低值分别回答总体收益、典型流的安全余量、以及已观察到的最坏情况。","note"))
    gas=read(WORK/"gas/day-paired.json")
    def dayrow(label,key):
        v=gas[key];a,b=(v["results"][k]["f1"] for k in VERSIONS)
        return [label,v["test"],f"{a:.4f}",f"{b:.4f}",f"{b-a:+.4f}",f'[{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]']
    g3,g4=(read(WORK/f"gas/full-{v}.json") for v in VERSIONS)
    ds=read(WORK/"gas/ds-ps.json")
    page(d,"04   /   GAS IDENTIFICATION","A · 跨日泛化要按日期复核",
         para("V4 保留 V3 独立传感器响应专家，并用同一条样本的稳态幅度归一化瞬态响应。每类最多六个原型，物理分支前两类得分越接近，附加校正越弱。")+
         table(["跨日评测窗","审计数","V3 F1","V4 F1","配对差","按日重采样 95% 区间"],
               [dayrow("V3 历史窗口","v3_original_window"),dayrow("V4 冻结末窗","terminal")])+
         para("同一历史窗口显示可复现的正向差值。新末窗只有数个日期，点估计上升但区间包含零，不能据此声称跨日收益已稳定。区间按日期成组配对重采样 4000 次。")+
         table(["补充检查","V3 → V4"],[["完整公开首轮所选 Macro-F1",pair(g3["selected"],g4["selected"],digits=4)],
               ["完整公开最弱组得分",pair(g3["weakest"],g4["weakest"],digits=4)],
               ["DS-PS 外部投影 Macro-F1（72 行）",pair(ds["independent"]["test"],ds["reliable_amend"]["test"],digits=4)]])+
         para("同一方法在新顺序的回访换样本流中，气体每流最差收益为 "+pair_stat(s,"fresh","A","worst")+"%。日期外推与线上阶段切换的估计量不同。","note"))
    b=read(WORK/"bearing/external.json");bentry=read(WORK/"bearing/entry.json");student=read(WORK/"bearing/student.json")
    page(d,"05   /   BEARING RUL","B · 逐轮相同，实际入口更快",
         para(f"V4 在加载时绑定阶次、频带秩曲线与归一化参数。开发与新留出所有场景共 {confirmed_b} 对 B 轮次，V3/V4 的部署选择一致，算法测量的 {verified['bearing']['numeric_comparisons']} 个可比数值逐项完全相等。")+
         table(["B 端点","V3 → V4"],[["完整公开首轮采用后的 RMSE（分钟）",pair(*(t["complete"][v]["rmse"] for v in VERSIONS))],
             ["XJTU 九台宏均 / 最坏 RMSE（分钟）",f'{b["XJTU"]["mean"][1]:.2f} / {b["XJTU"]["worst"][1]:.2f}，两版相同'],
             ["FEMTO 六台宏均 / 最坏 RMSE（分钟）",f'{b["FEMTO"]["mean"][1]:.2f} / {b["FEMTO"]["worst"][1]:.2f}，两版相同'],
             ["KAIST 跨台架至末次记录 RMSE（分钟）",pair(*(b["KAIST"][v+"_rmse"] for v in VERSIONS))],
             ["含混合和 JSON 编码的热入口（条/CPU 秒）",pair(*(bentry[v]["samples_per_cpu_second"] for v in VERSIONS),digits=0)]])+
         para("FEMTO 旧资料的 87.73 / 189.29 分钟属于覆盖门槛为 2.5% 的研究候选。冻结版使用 5%，六折末端样本仅占 3.13%–4.53%，均走早期分支。上表已按实际部署代码校正。")+
         para(f'低秩学生把头压缩至 {student["student"]["head_bytes"]} 字节，但完整公开裸头 RMSE 从 {student["teacher"]["public_rmse"]:.2f} 升到 {student["student"]["public_rmse"]:.2f} 分钟。冻结选择保留寿命估计的数值行为。',"note"))
    engine_rows=[]
    for label,dataset in (("官方固定池","official"),("整机折 1","fold1"),("整机折 2","fold2"),("N-CMAPSS DS02","DS02")):
        a,bh=final_engine(dataset);engine_rows.append([label,pair(a["rmse"],bh["rmse"],digits=3),pair(a["nasa"],bh["nasa"],digits=0)])
    ext=final_engine("DS02");fixed=final_engine("official")
    page(d,"06   /   ENGINE RUL","C · 用寿命阶段监督连续残差",
         para("按当轮训练寿命的第 20、45、70 百分位建立三个顺序事件，训练时与连续寿命残差共享神经表示。辅助损失在新增训练池取 0.7，稳定且已有头时取 0.2。部署时丢弃三个辅助输出，沿用 V3 的单边延寿保护。")+
         table(["训练域或外部整机对照","RMSE（循环）V3 → V4","NASA 非对称代价 V3 → V4"],engine_rows)+
         para("DS02 同时检查巡航和非巡航条件。在相同审计样本上，巡航 RMSE 为 "+pair(*(z["cruise"] for z in ext),digits=3)+"，非巡航为 "+pair(*(z["noncruise"] for z in ext),digits=3)+"。两个整机折的 RMSE 略升，固定官方池的 NASA 代价也上升。")+
         table(["C 新留出","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[NAMES[k],*comparison(s,k,"C")] for k in ("drift","fresh","official")])+
         para("C 在四个新留出协议的各流低谷均值均上升。主漂移最坏单轮由 "+pair_stat(s,"drift","C","lowest")+"%，主漂移、换样本及固定池等设置出现更大历史回撤。训练与推理只用当轮开放数据。","note"))
    page(d,"07   /   PERFORMANCE","逐条推理的热延迟、长尾与冷装载",
         para("在同一核上交替运行两版，五次重复，每次从公开验证抽取最多 600 行。计时从解析输入 JSON 开始，经过实际 Harness 接口，到编码输出 JSON 结束。每列取五次指标的中位。")+
         table(["任务","p50（微秒）","p95（微秒）","冷首条（毫秒）","热吞吐（条/秒）"],end_to_end())+
         para("冷首条包含重新打开检查点及首次运算，磁盘操作系统缓存可能仍热。热吞吐由单次平均用时换算，p95 描述长尾。三类任务分别按自身已冻结产物测量。")+
         table(["真实 SDK 镜像管道","逐条往返 p50（微秒）","逐条往返 p95（微秒）","新进程至首包（毫秒）"],sdk_measurements())+
         para("第二组使用真实 SDK 与镜像文件系统，每次写入一条 JSON 并等待读回应。新进程冷首包另含 Python 和 SDK 初始化。两个时钟覆盖范围不同。","note"))
    row=page(d,"08   /   STREAM TRACE","同一冻结顺序的十轮轨迹",
             para("主漂移协议顺序 10。三幅图依次显示分类、轴承和发动机在当轮审计阶段相对各自静态起点的收益（%）。纵轴范围一致，水平零线表示与静态起点持平；每个点对应一次实际部署。")+
             table(["任务","平均收益 %","最低阶段 %","最大历史回落点"],[[k,*[pair(*(runs[f"drift-{v}-{k}-10"][m] for v in VERSIONS),100) for m in ("mean","worst","forget")]] for k in TASKS]))
    traces(row,runs,"drift",10,top=280)
    calls=[]
    for k,(rec,items) in live.items():
        cfg=rec["configuration"]
        calls.append([k,cfg["train_rows"],cfg["validation_rows"],cfg["plan"]["tool"],cfg["deployed"],
                      sum(z["usage"]["total_tokens"] for z in items),"/".join(sorted({z.get("provenance","记录") for z in items}))])
    distinct={z["id"]:z for _,items in live.values() for z in items}
    paid=[z for z in distinct.values() if z["provenance"]=="v4-api"]
    ncache=sum(z["provenance"]=="v3-cache" for z in distinct.values())
    page(d,"09   /   HOST & SAFETY","宿主模型回执与工业部署边界",
         para("控制模型由 SDK 宿主提供 Qwen/Qwen3.8-27B。规划通过短动作菜单描述候选，实际训练与预测在单线程离线 CPU 容器内执行。验证与审计分离，保留父检查点供恢复。")+
         table(["公开首轮","训练行","选型行","动作","部署","回执 Token","回执来源"],calls)+
         para(f"按回执 ID 去重后共 {len(distinct)} 次决定，其中 {ncache} 次命中 V3 请求摘要，{len(paid)} 次新请求合计 {sum(z['usage']['total_tokens'] for z in paid)} Token。表内按任务记录 Token。公开流的可控评审在本地运行。")+
         table(["单核十轮训练","CPU 秒","预测均时（微秒）","候选文件合计 MB","决策次数"],resource_rows())+
         table(["运行环节","工程控制"],[["输入与数值","校验维度、任务、非有限值。分类按六类概率输出，回归输出限定为非负寿命。"],
          ["异常与回退","候选达不到可见组表现和记忆条件时保留已有产物。离线预测只访问已加载检查点。"],
          ["使用范围","轴承寿命以分钟、发动机以循环计。跨域排序偏弱或误差增大时，保留人工维护判断。"]]))
    page(d,"10   /   REPRODUCIBILITY","从发布件复核每一个指标",
         table(["材料","核查入口"],[["公开实验证据.zip","paired-summary.json 与 replay/ 提供两版配对的轮次记录，protocol.json、freeze.json 提供预登记和文件指纹。"],
          ["企业赛V4源码.zip","企业赛/V4 含实际训练/推理代码、消融、复验和打包脚本，也收录直接依赖的前版代码。"],
          ["性能与端点","serving/ 提供原始 p50/p95/冷加载；resources/ 提供单核训练。verification.json 独立重算 2280 个版本轮次；sdk-latency.json 提供 SDK 管道往返记录。"],
          ["跨域与安全","gas/day-paired.json、bearing/external.json、engine/*-frozen-probe.json 记录各自外部实验。镜像与源码摘要见 SHA256SUMS.txt。"]])+
         para("复现顺序为准备授权公开训练/验证投影、校验冻结摘要、运行开发与新留出配对、单核成本及镜像验收，再执行 release.py。数据集、模型权重和平台凭据由各自授权渠道提供。")+
         para("参考与原始项目　气体跨日记录及特征取自 <a href='https://doi.org/10.1038/s41597-025-05993-8'>Scientific Data 数据论文</a> 与 <a href='https://zenodo.org/records/15681119'>开源特征提取</a>；轴承寿命参考 <a href='https://doi.org/10.3390/machines14101119'>Yao 等的多域健康指标</a>；发动机阶段监督参考 <a href='https://doi.org/10.3390/s26227305'>退化阶段研究</a>，巡航掩码取自 <a href='https://github.com/NostalgiaJohn/CruiseBench'>CruiseBench</a>。","note"))
    return d

def movie(runs,live,path,stage):
    d=pdf.open()
    page(d,"REPLAY   /   FROZEN V4","企业 V4 公开评测运行回放",
         para("九条冻结主漂移流，三项任务，各十轮。下面以预先登记的拆分 10 为例，展示真实执行的逐轮选型、实际审计指标和资源记录。审计量、延迟及全部顺序可在配套 JSON 逐项复算。")+
         table(["设置","数据与执行"],[["主回放","随轮次释放训练数据与公开选型，控制模型回复固定，工业模型实际更新。"],["交叉检查","回访换样本、官方固定训练池、按公开测量裁决。"],["线上连通性","三项公开首轮由宿主转发或读取完全相同请求的 27B 回执。"]]))
    names={"candidate":"候选","parent":"父产物","history":"历史","anchor":"静态参考"}
    for k in TASKS:
        rec,calls=live[k]
        c=rec["configuration"]
        page(d,"REPLAY   /   QWEN 27B",f"{k} · {TASK_NAMES[k]}的公开首轮",
             table(["真实执行项","记录"],[["模型","Qwen/Qwen3.8-27B"],["训练与验证样本",f'{c["train_rows"]} / {c["validation_rows"]}'],["规划动作",c["method"]+" / "+c["plan"]["strength"]],["Adam 步数",f'{c["old_steps"]} → {c["steps"]}'],["部署决策",names.get(c["deployed"],c["deployed"])],["规划 / 裁决 Token"," / ".join(str(a["usage"]["total_tokens"]) for a in calls)]]))
        old,new=[runs[f"drift-{v}-{k}-10"]["rounds"] for v in VERSIONS]
        for start in (0,5):
            rows=[]
            for a,b in zip(old[start:start+5],new[start:start+5]):
                i=b["stage"]
                rows.append([b["round"],i,names.get(b["deployed"],b["deployed"]),b["audit"],pair(a["metrics"][i],b["metrics"][i],digits=3),pair(a["stage_gain"],b["stage_gain"],100)])
            page(d,"REPLAY   /   TEN-ROUND STREAM",f"{k} · 顺序 10 的第 {start+1} 至 {start+5} 轮",
                 table(["轮","阶段","部署","审计数","V3 → V4 "+METRICS[k],"相对增益 %"],rows)+
                 para("每轮结束后以相同的隔离审计样本重新预测。相对增益为正表示优于本流静态起点。阶段编号为 0–3。"))
    page(d,"REPLAY   /   EVIDENCE","从一次回放追溯到原始记录",
         para("打开公开实验证据，按 replay/场景-版本-任务-拆分.json 检查当轮选型与部署。freeze.json、protocol.json 记录冻结代码和输入摘要。resources 保存单核交替测量，live 保存按请求摘要验证过的回执。"))
    frames=stage/"frames";frames.mkdir()
    for i,p in enumerate(d):
        p.get_pixmap(matrix=pdf.Matrix(1.6,1.6),alpha=False).save(frames/f"{i:02}.png")
    d.close()
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y","-framerate","1/7","-i",str(frames/"%02d.png"),
                    "-vf","scale=1280:-2,fps=12,format=yuv420p","-an","-c:v","libx264","-threads","2","-preset","veryfast",
                    "-crf","21","-movflags","+faststart",str(path)],check=True)
    probe=read_ffprobe(path)
    if not (60<=float(probe["format"]["duration"])<300 and probe["streams"][0]["codec_name"]=="h264" and path.stat().st_size<50_000_000):
        raise ValueError("Video check failed")


def read_ffprobe(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]))

def audit_pdf(path):
    output=WORK/"visual-qa";output.mkdir(exist_ok=True)
    images=[]
    with pdf.open(path) as d:
        if len(d)!=10 or path.stat().st_size>10_000_000:
            raise ValueError("Report exceeds specification")
        for i,page in enumerate(d,1):
            if len(page.get_text().strip())<150:
                raise ValueError("Incomplete PDF text")
            pix=page.get_pixmap(matrix=pdf.Matrix(1.45,1.45),alpha=False)
            pix.save(output/f"page-{i:02}.png")
            thumb=Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
            thumb.thumbnail((575,383));images.append(thumb)
    for j in range(0,len(images),2):
        sheet=Image.new("RGB",(575,2*383+8),"#e4e9e5")
        sheet.paste(images[j],(0,0));sheet.paste(images[j+1],(0,391))
        sheet.save(output/f"review-{j//2+1}.jpg",quality=89)


GUIDE="""成组配对复核
paired-summary.json 按开发或确认、场景、版本及任务给出汇总。replay 内每行 round 为轮次，stage_gain 为本阶段部署相对静态收益，metrics 为全部阶段审计端点，forget 为曾见阶段的历史回落。每条公开主流十轮，确认集按任务和顺序配对。冻结摘要与原件哈希见 protocol.json、freeze.json。
gas/day-paired.json 区分 V3 历史窗口和 V4 冻结末窗，含配对日期重采样区间。bearing/confirmed.json 是逐轮选择的等价检查；bearing/external.json 直接重算 V3/V4 真实预测并复核旧 FEMTO 研究候选。engine 目录的多路线日志记录实际工作点。
serving 含原始 JSON 入口延迟与差分，resources 含单核计时。controller 为不同控制器的开发消融。live 保留脱敏请求指纹、生成文字和Token回执，provenance 记录相同请求的缓存或实际接口。模型参数、逐条原始训练观测、完整裁决请求及账户信息不在本包。
"""


def bundles(source,evidence):
    files=set()
    for folder in (HERE,HERE.parent/"V3",HERE.parent/"V2",HERE.parent/"V1"):
        files.update(path for path in folder.iterdir() if path.is_file() and path.name!="AGENTS.md"
                     and path.suffix in (".py",".md",".txt"))
    campus=ROOT/"校赛"
    files.update(campus/name for name in ("prepare.py","build_image.py","README.md","来源声明.md","Dockerfile"))
    files.update((campus/"app").glob("*.py"))
    files.update(campus/"V2"/name for name in ("gas.py","bearing.py","engine.py","bench_external.py",
                                               "policy.py","control.py","replay.py","README.md"))
    # The official starting scripts and source data are supplied by the contest owner.
    with zipfile.ZipFile(source,"w",zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for f in sorted(files):
            z.write(f,f.relative_to(ROOT))
        z.writestr("复现入口.txt","使用本包原有的仓库相对目录。企业赛/V4/README.md 与三份实验说明给出镜像、授权投影和回放命令；校赛/prepare.py 处理授权数据。Python/NumPy 为生产计算依赖，比较与材料阶段另需 h5py、SciPy、Pillow、PyMuPDF、FFmpeg。评测 SDK、官方权重、比赛数据及补充数据由各授权渠道获取。")
    paths={}
    for folder in ("replay","resources","controller","serving","gas","bearing","engine","live"):
        for f in (WORK/folder).rglob("*.json"):
            if f.stat().st_size>3_000_000:raise ValueError("Unexpected evidence size "+str(f))
            paths[f.relative_to(WORK).as_posix()]=f
    for f in (WORK/"freeze.json",WORK/"protocol.json",WORK/"paired-summary.json",WORK/"verification.json",WORK/"sdk-latency.json",WORK/"image-proof-0.log"):
        paths[f.name]=f
    for mask in ("sdk-*.log","qualification-*.log","quiet-*.log","*smoke*.log","*acceptance*.json"):
        for f in WORK.glob(mask):
            if f.is_file() and f.stat().st_size and f.stat().st_size<150_000:
                paths[f.name]=f
    with zipfile.ZipFile(evidence,"w",zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for name,f in sorted(paths.items()):
            if name.startswith("live/") and f.name.startswith("call-"):
                receipt=read(f)
                allowed={"id","text","usage","provenance","digest"}
                if set(receipt)-allowed:raise ValueError("Review additional receipt fields before packaging")
            z.write(f,name)
        z.writestr("公开证据索引.txt",GUIDE)
    for path in (source,evidence):
        if path.stat().st_size>50_000_000:raise ValueError("ZIP exceeds 50 MB")
        with zipfile.ZipFile(path) as z:
            if z.testzip() or any(Path(name).suffix.lower() in (".npz",".env",".toml",".key",".h5",".pem")
                                  or "AGENTS.md" in name for name in z.namelist()):
                raise ValueError("Invalid archive "+str(path))


def main():
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument("--preview",action="store_true",help="Inspect layout without releasing artifacts")
    args=parser.parse_args()
    verify_release_inputs()
    summary,runs,live=load()
    DELIVERY.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        stage=Path(tmp); path=stage/"企业赛V4技术报告.pdf"
        d=report(summary,runs,live)
        d.set_metadata({"title":TITLE+" · 企业赛V4","author":"McCarthy","subject":"公开冻结复验、跨域及单条延迟"})
        d.subset_fonts();d.save(path,garbage=4,deflate=True);d.close()
        audit_pdf(path)
        if args.preview:
            print("Report preview rendered to",WORK/"visual-qa");return
        video=stage/"企业赛V4运行回放.mp4"; movie(runs,live,video,stage)
        src=stage/"企业赛V4源码.zip";ev=stage/"公开实验证据.zip"; bundles(src,ev)
        cfg=stage/"模型配置.txt";shutil.copyfile(HERE/cfg.name,cfg)
        outputs=[path,video,src,ev,cfg]
        sums=stage/"SHA256SUMS.txt"
        sums.write_text("".join(sha(f)+"  "+f.name+"\n" for f in [DELIVERY/"harness.tar",*outputs]))
        for f in [*outputs,sums]:
            f.replace(DELIVERY/f.name)
            print(f.name,(DELIVERY/f.name).stat().st_size,flush=True)


if __name__=="__main__":
    main()
