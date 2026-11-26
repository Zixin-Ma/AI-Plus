"""Build and verify enterprise V3 submission materials from the frozen public evidence."""
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
WORK=HERE.parent/"work/v3"
DELIVERY=WORK/"delivery"
TASKS="ABC"
VERSIONS=("v2","v3")
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
    p.insert_text((34,30),"BJTU INDUSTRIAL AI     /     ENTERPRISE V3",fontname="hebo",fontsize=10,color=NEW)
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
    return range(5) if s in ("drift","official") else (3,4)


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
            ids=counts(case) if section=="dev" else (5,6,7)
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
    for j,(label,c) in enumerate([("V2",BASE),("V3",NEW)]):
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


def comparison_controls():
    """Intersect route/control filenames to keep splits, settings, and tasks identical."""
    folder=WORK/"controller"
    rows=[]
    for series,control in [("v2-experts","control"),("integrated","control")]:
        base=folder/series
        for route in ("anchor","recall","both","switch05","block75"):
            selected=sorted((base/route).glob("*.json"))
            pairs=[]
            for path in selected:
                ref=base/control/path.name
                if ref.is_file():
                    pairs.append((read(ref),read(path)))
                elif series=="v2-experts":
                    # Historical reference for V2 experts has the actual V2 harness.
                    ref=WORK/"replay"/path.name.replace("-v3-","-v2-")
                    if ref.is_file():
                        pairs.append((read(ref),read(path)))
            if not pairs:
                continue
            delta=lambda key:100*statistics.mean(b[key]-a[key] for a,b in pairs)
            rows.append([("V2" if series=="v2-experts" else "V3"),route,len(pairs),
                         round(delta("mean"),2),round(delta("worst"),2),round(delta("forget"),2)])
    return rows


def live_facts(live):
    active={r["id"]:r for _,calls in live.values() for r in calls}
    paid=dict(active)
    for path in (WORK/"live").glob("*/prior-*.json"):
        r=read(path)
        paid[r["id"]]=r
    charges=[v for v in paid.values() if v.get("provenance")=="v3-api"]
    reused=[r for r in active.values() if r.get("provenance")=="v2-cache"]
    return len(active),sum(r["usage"]["total_tokens"] for r in active.values()),len(reused),len(charges),sum(r["usage"]["total_tokens"] for r in charges)


def report(s,runs,live):
    d=pdf.open()
    columns=["公开场景","均值收益 %","各流低谷均值 %","历史回落点","恢复轮"]
    score=lambda key:pair_stat(s,"fresh","all",key,1 if key=="delay" else 100,3 if key=="delay" else 2)
    primary=s["confirm/drift"]
    nbetter=sum(runs[f"drift-v3-{k}-{n}"]["worst"]>runs[f"drift-v2-{k}-{n}"]["worst"] for k in TASKS for n in (5,6,7))
    movements=[(runs[f"drift-v2-{k}-{n}"],runs[f"drift-v3-{k}-{n}"]) for k in TASKS for n in (5,6,7)]
    mean_wins=sum(b["mean"]>a["mean"]+1e-10 for a,b in movements)
    less_forget=sum(b["forget"]<a["forget"]-1e-10 for a,b in movements)
    same_forget=sum(abs(b["forget"]-a["forget"])<=1e-10 for a,b in movements)
    more_forget=len(movements)-less_forget-same_forget
    page(d,"01   /   PUBLIC RESULTS","企业 V3 在九条冻结漂移流上的表现",
         para("每当气体响应、轴承负荷或发动机退化阶段变化，系统需要在少量已释放样本上更新模型，并保留此前工况的预测能力。企业 V3 从企业 V2 出发，以新样本回访、整机及跨设备实验检验五项客观评分方向。")+
         table(["九条冻结复验流，各十轮","企业 V2","企业 V3"],[
           [label,*[f'{primary[v]["all"][m]*scale:.{digits}f}' for v in VERSIONS]] for label,m,scale,digits in [
             ("平均阶段相对收益（%）","mean",100,2),("每流最低收益的均值（%）","worst",100,2),
             ("全部阶段中的最低收益（%）","lowest",100,2),("已见阶段最大回落均值（百分点）","forget",100,2),
             ("切换后恢复延迟（轮）","delay",1,3)]])+
         para(f"九条流中，{nbetter} 条的最低阶段收益上升。{mean_wins} 条流平均收益上升。历史回落 {less_forget} 条改善、{more_forget} 条增大、{same_forget} 条持平；宏平均改善主要来自轴承大回落受控。两版在同一轨迹、相同起点下配对，资源与真实模型回执另页计量。")+
         table(["各流低谷 %（V2 → V3）","冻结顺序 5","冻结顺序 6","冻结顺序 7"],
               [[k+"  "+TASK_NAMES[k],*[pair(*[runs[f"drift-{v}-{k}-{n}"]["worst"] for v in VERSIONS],100,1) for n in (5,6,7)]] for k in TASKS])+
         para("这里的收益来自公开数据和本地可复现协议。赛事平台按自身隐藏流、模型回复和计费规则给出正式分数。","note"))
    orders=read(WORK/"protocol.json")["new_orders"]
    page(d,"02   /   STUDY DESIGN","冻结的是程序、输入和未来阶段顺序",
         table(["协议环节","实施口径"],[
           ["开发与留出","拆分 0–4 用于设计。冻结生产模块、评价器、公开训练与验证输入的 SHA-256 后运行拆分 5、6、7。每个版本每种设置运行 A、B、C 各三条流，共 90 轮。"],
           ["训练的可见范围","逐步释放用每条轨迹前 55% 建立静态起点，此后分轮放至 100%。固定池使用官方检查点和完整训练池，验证仍按同样的十阶段顺序轮换。"],
           ["选型与审计","A、B 按任务域和阶段留出连续审计块，C 按整机隔离。fresh 的同一阶段回访逐次分摊选择块；drift 重用该阶段原有选择块。不同拆分共享公开来源。"],
           ["裁决对照","受控回复给出相同训练动作和候选请求，所有方案均执行真实工业训练、宿主复算及安全门。fresh-measured 改由可用候选中主指标最好的产物获选。"],
           ["计分方向","A 取 Macro-F1 上升比例，B/C 取 RMSE 下降比例。先在流内取均值、最低值，再对九条流等权。历史回落相对已见阶段历史峰值，恢复按相对收益非负计，阶段结束时截尾。"]])+
         table(["冻结拆分","第 1 至第 10 轮使用的阶段序号"],[[i,"   ·   ".join(map(str,stages))] for i,stages in zip((5,6,7),orders)])+
         para("四种设置共用冻结的代码及数据。逐轮记录附有训练行数、选择样本、审计量、各阶段静态值、部署值、门槛结果与检查点大小。","note"))
    page(d,"03   /   TASK COMPARISON","三项任务在新选型样本上的表现",
         table(["三个冻结顺序的均值",*columns[1:]],[[k+"  "+TASK_NAMES[k],*comparison(s,"fresh",k)] for k in TASKS])+
         para("表中每项任务等权汇总拆分 5、6、7。下图单独展示拆分 6，横轴为第 1 至 10 轮，纵轴为当轮阶段相对收益（%），各图共用刻度。验证样本用完一个阶段后切换，短时下降会同时影响低谷和恢复。"))
    traces(d[-1],runs,"fresh",6,289)
    page(d,"04   /   SCENARIO SENSITIVITY","更换训练池和裁决规则后的四组结果",
         table(columns,[[NAMES[c],*comparison(s,c)] for c in CASES])+
         table(["冻结训练场景","V2 → V3 的全局最低阶段 %","九条流中低谷上升 / 持平 / 下降"],[
           [NAMES[c],pair_stat(s,c,"all","lowest")," / ".join(str(sum((d>1e-10 if z==0 else abs(d)<=1e-10 if z==1 else d< -1e-10) for d in [runs[f"{c}-v3-{k}-{n}"]["worst"]-runs[f"{c}-v2-{k}-{n}"]["worst"] for k in TASKS for n in (5,6,7)])) for z in range(3))] for c in CASES])+
         para("表中箭头均为企业 V2 到企业 V3。每流低谷均值反映典型顺序中的弱阶段，全局最低值对应一次最不利组合。四种设置使用相关的公开流，重复设定主要用于评价敏感性。","note"))
    controls=comparison_controls()
    shortlist=[r for r in controls if r[1] in ("anchor","recall","both")]
    page(d,"05   /   POLICY ABLATIONS","控制器能否在阶段重现时找到合适的父模型",
         para("每轮从宿主给出的父检查点恢复，保留 Adam、归一化参数与原始工业模型。27B 根据漂移诊断提出动作；验证面板测量当前候选、父产物与获准历史。历史产物须经过当前投影复核，回归任务还检查训练寿命分位的性能回落。")+
         table(["任务头","开发路线","匹配流数","均值差（点）","低谷差（点）","回落差（点）"],shortlist)+
         para("anchor 允许把静态骨干列为选择对象，recall 按相近的可见工况找回历史，both 同时启用。逐一匹配相同任务、拆分与裁决设置比较；V2 头缺省对照为原 V2，V3 头使用同组 control。均值及低谷差为正有利，回落差为负有利。其余候选与原始 JSON 在证据包。","note"))
    ga,gb=[read(WORK/f"gas/full-{v}.json") for v in VERSIONS]
    bearing=read(WORK/"bearing/final.json")
    crows=read(WORK/"engine/official-probe-frozen.json")
    ca,cb=[r for r in crows if r["round"]==max(x["round"] for x in crows)]
    page(d,"06   /   TASK MODELS","三个任务头分别限制什么误差",
         table(["部署机制","同一公开起点或模型的开发对照"],[
           ["A · 约束幅度外推，合并独立传感器证据；压缩头部数组",
            f'完整公开首轮 Macro-F1 {pair(ga["macro"],gb["macro"],digits=4)}，最弱类召回 {pair(ga["worst"],gb["worst"],digits=4)}。受控重拟合，两个版本按同一协议选权。'],
           ["B · 以低频振动秩估计退化进程；高频冲击和骨干寿命控制修正",
            f'完整公开数据直接用头的 RMSE {pair(*[bearing[v]["public"]["all"] for v in VERSIONS])} 分钟。首轮经宿主选权的 RMSE {pair(*[bearing[v]["selected"]["rmse"] for v in VERSIONS])} 分钟。'],
           ["C · 高于骨干的延寿量保留一半，低于骨干的预警全部保留",
            f'固定起点完整释放模型 RMSE {pair(ca["rmse"],cb["rmse"])} 循环，NASA 总代价 {pair(ca["nasa"]/1e9,cb["nasa"]/1e9)} 十亿。过早预警代价增加，延迟检修的高估代价下降。']])+
         para("训练标签只来自获准的公开训练分区；供审计的类别、设备和服役相位仅用于计分。校正与上传均保留归一化、单位及回退接口，单条样本完成推理。")+
         para('<a href="https://doi.org/10.1002/cem.70148">气体传感器校准迁移</a>　<a href="https://doi.org/10.3390/machines14101119">轴承变化阶段识别研究</a>　<a href="https://arxiv.org/abs/2611.28759">工况归一化与 NASA 代价</a>　<a href="https://doi.org/10.36001/phme.2026.v9i1.4903">健康及退化状态的不确定性</a>','note'))
    external=[]
    for name,key in [("DS-PS 设备迁移","ds-ps"),("跨日气体传感","cross-day")]:
        z=read(WORK/f"gas/{key}.json")
        external.append(["A · "+name,f'{z["train"]} / {z["selection"]} / {z["test"]} 行', "Macro-F1",
                         pair(*[z["candidates"][z["selected"][v]]["raw"]["f1"] for v in VERSIONS],digits=4)])
    bearing_panel=read(WORK/"bearing/impact/panel.json")
    def bearing_pair(name):
        return pair(*[bearing_panel[name][v]["mean"] for v in ("v2","impact-full-0.75")])
    external.extend([["B · XJTU 整轴承","9 个设备折","宏平均 RMSE（分钟）",bearing_pair("xjtu")],
                     ["B · FEMTO 整轴承","6 个设备折","宏平均 RMSE（分钟）",bearing_pair("femto")],
                     ["B · KAIST 单轨",f'{bearing["kaist_rows"]} 个观测文件',"端点 RMSE（分钟）",pair(*[bearing[v]["kaist"]["rmse_minutes"] for v in VERSIONS])]])
    for source,label,split in [("fold1","C-MAPSS 整机折 1","23,777 行"),("fold2","整机折 2","22,691 行"),("DS02","N-CMAPSS DS02","2 台外部测试机")]:
        rr=read(WORK/f"engine/{source}-probe-frozen.json")
        vals=[r for r in rr if r["round"]==max(x["round"] for x in rr)]
        external.append(["C · "+label,split,"RMSE（循环）",pair(*[r["rmse"] for r in vals])])
    page(d,"07   /   DOMAIN TRANSFER","跨仪器、跨轴承与跨发动机验证",
         table(["数据来源","样本范围","度量","V2 → V3"],external)+
         para("A 跨日的精度下降，阈值需结合本厂仪器复标。B 的 FEMTO 六个设备中一个单机误差下降，均值改善由该设备主导；KAIST 以末次观测为目标终点。C 的 DS02 巡航与非巡航均独立评估，公开固定池有更大的早期寿命低估代价。源数据由使用者按原许可获取。","note"))
    a,b=[primary[v]["all"] for v in VERSIONS]
    active,tokens,reused,new,newtokens=live_facts(live)
    page(d,"08   /   RESOURCE ACCOUNTING","将训练成本、在线调用和推理分开计量",
         table(["单核中位数","十轮 CPU 秒","单样本含 JSON（µs）","十轮上传 MB","控制调用"],resource_rows())+
         para("每项任务在可用 CPU 核上固定亲和性，三个重复交替两个版本的执行顺序；秒/微秒取中位数，上传量和调用次数取首次。")+
         table(["逐步释放，冻结九条流合计","V2","V3"],[
           ["受控模型调用次数",a["calls"],b["calls"]],["提示正文 UTF-8 字节",f'{a["prompt_bytes"]:,}',f'{b["prompt_bytes"]:,}'],
           ["候选检查点 MB",f'{a["checkpoint_bytes"]/1e6:.2f}',f'{b["checkpoint_bytes"]/1e6:.2f}']])+
         para(f"真实 27B 公开首轮共链接 {active} 个规划与裁决回执，合计 {tokens:,} Token，其中 {reused} 个复用同摘要历史调用。计入本次 API 新调用及留存的早期尝试后，按唯一 ID 统计为 {new} 次、{newtokens:,} Token。离线回放请求字节与在线 Token 分别记录。","note"))
    labels={"candidate":"候选","parent":"父产物","history":"历史产物","anchor":"静态参考"}
    rows=[]
    for k,(rec,calls) in live.items():
        c=rec["configuration"]
        selected=c["deployed"]
        rows.append([k,c["method"],labels.get(selected,selected),f'{c["old_steps"]} → {c["steps"]}',pair(c["scores"]["parent"]["primary"],c["scores"][selected]["primary"],digits=4),sum(r["usage"]["total_tokens"] for r in calls)])
    page(d,"09   /   DEPLOYMENT","从离线容器启动到人工检修决策",
         table(["27B 真实首轮","动作","部署","Adam 步数","验证主指标","Token"],rows)+
         para("Linux/amd64、Python 3.12、NumPy 2.2.6 和 SDK 1.0.4。先核对 SHA256SUMS.txt，再用 docker load -i harness.tar 导入。以镜像执行 describe 检查协议和模型；train、predict 通过赛方宿主装载授权数据与工作目录。流式输出为气体分类分数或非负寿命。")+
         para("未识别的训练动作或不满足守门条件的候选按代码回退。异常预测须保留设备及输入记录，暂停自动维护决策，重新装载已通过验收的父检查点或上版镜像。上线前使用本厂工况分层验证气体弱类、轴承冲击分布和发动机寿命高估。检修、换件或停机由设备负责人核准。")+
         para("源码包内有运行说明、研究脚本和相关许可说明；完整配对记录、冻结哈希、镜像与 SDK 验收、外部数据汇总和去正文模型回执位于公开实验证据包。","note"))
    def event(n,index):
        old,new=[runs[f"drift-{v}-B-{n}"]["rounds"][index-1] for v in VERSIONS]
        phase=old["stage"]
        return [f"B / {n} / {index}",old["audit"],f'{old["static"][phase]:.2f}',f'{old["metrics"][phase]:.2f}',f'{new["metrics"][phase]:.2f}',pair(old["stage_gain"],new["stage_gain"],100)]
    base_live=HERE.parent/"work/v2/live"
    rows=[]
    for task in TASKS:
        c=live[task][0]["configuration"];b=read(base_live/task/"round.json")["configuration"]
        rows.append([task,METRICS[task],pair(*[q["scores"][q["deployed"]]["primary"] for q in (b,c)],digits=4)])
    page(d,"10   /   FAILURE ANALYSIS","哪些阶段和使用成本仍需承担",
         table(["任务 / 顺序 / 轮","审计数","静态 RMSE","V2 RMSE","V3 RMSE","收益 %"],[event(7,10),event(5,2)])+
         para("轴承顺序 7 的原始低谷发生在选定阶段末次回访，允许采用静态参考后回到起点水平。V3 仍有寿命估计偏离起点的轮次。冻结漂移复验中，A 的每流最低收益均值下降但全局最低改善；C 的每流最低收益均值由正转负。")+
         table(["真实公开首轮","指标","V2 → V3"],rows)+
         para("完整公开首轮中，B 和 C 的 RMSE 均升高。单核复测显示 A、B 单条服务变慢，C 大致持平；减少规划调用和检查点传输并未同步缩短所有任务的计算时间。设备方应同时验收吞吐、每条流的最差阶段及维护成本。","note"))
    return d

def validate_checks():
    for name,count in [("qualification-3.log",7),("qualification-4.log",3),("sdk-1.log",4)]:
        text=(WORK/name).read_text()
        if f"Ran {count} tests" not in text or not text.rstrip().endswith("OK"):
            raise ValueError("Missing successful check "+name)
    text=(WORK/"sdk-0.log").read_text()
    for k in TASKS:
        for i in (1,2,3):
            assert f"{k} round {i}:" in text
    for i in range(3):
        q=WORK/f"quiet-{i}.log"
        if not q.exists() or len([s for s in q.read_text().splitlines() if s.startswith(("v2 ","v3 "))])!=6:
            raise ValueError("Missing final resource measurements: "+str(q))


def movie(runs,live,path,stage):
    d=pdf.open()
    page(d,"REPLAY   /   FROZEN V3","企业 V3 公开评测运行回放",
         para("九条冻结主漂移流，三项任务，各十轮。下面以预先登记的拆分 7 为例，展示真实执行的逐轮选型、实际审计指标和资源记录。审计量、延迟及全部顺序可在配套 JSON 逐项复算。")+
         table(["设置","数据与执行"],[["主回放","随轮次释放训练数据与公开选型，控制模型回复固定，工业模型实际更新。"],["交叉检查","回访换样本、官方固定训练池、按公开测量裁决。"],["线上连通性","三项公开首轮由宿主转发或读取完全相同请求的 27B 回执。"]]))
    names={"candidate":"候选","parent":"父产物","history":"历史","anchor":"静态参考"}
    for k in TASKS:
        rec,calls=live[k]
        c=rec["configuration"]
        page(d,"REPLAY   /   QWEN 27B",f"{k} · {TASK_NAMES[k]}的公开首轮",
             table(["真实执行项","记录"],[["模型","Qwen/Qwen3.8-27B"],["训练与验证样本",f'{c["train_rows"]} / {c["validation_rows"]}'],["规划动作",c["method"]+" / "+c["plan"]["strength"]],["Adam 步数",f'{c["old_steps"]} → {c["steps"]}'],["部署决策",names.get(c["deployed"],c["deployed"])],["规划 / 裁决 Token"," / ".join(str(a["usage"]["total_tokens"]) for a in calls)]]))
        old,new=[runs[f"drift-{v}-{k}-7"]["rounds"] for v in VERSIONS]
        for start in (0,5):
            rows=[]
            for a,b in zip(old[start:start+5],new[start:start+5]):
                i=b["stage"]
                rows.append([b["round"],i,names.get(b["deployed"],b["deployed"]),b["audit"],pair(a["metrics"][i],b["metrics"][i],digits=3),pair(a["stage_gain"],b["stage_gain"],100)])
            page(d,"REPLAY   /   TEN-ROUND STREAM",f"{k} · 顺序 7 的第 {start+1} 至 {start+5} 轮",
                 table(["轮","阶段","部署","审计数","V2 → V3 "+METRICS[k],"相对增益 %"],rows)+
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


GUIDE="""九条冻结流每版各十轮。同一套 A、B、C，各用顺序 5、6、7。replay 有完整开发及冻结配对；paired-summary.json 可由企业赛/V3/qualify.py summary 从源文件重算。
每轮 stage 为当前阶段；static/metrics 为分阶段起点和部署值，gains 是方向一致的相对收益。mean 是十轮平均，worst 是流内最低谷，lowest 为同组单一最低值。forget 表示已见阶段相对峰值的最大回落，delay 表示切换后恢复到非负收益的平均等待轮数。
controller 分 integrated 和 v2-experts，记录控制器与两代任务头的组合。gas、bearing、engine 记录分项消融和外部验证；各外部原件和私有设备文件不在证据包中。
resources 是安静时段各任务三次单核交替对照。checkpoint-io.json 包含检查点读写与数值一致性审计。sdk-0.log 记录三项任务每项三轮的镜像宿主检查，qualification-3.log、qualification-4.log 和 sdk-1.log 为14个自动测试的验收。
live 的回执以摘要取代提示正文。provenance 为 v3-api 表示新请求，v2-cache 表示命中相同请求的既有回执；请按调用 ID 去重。baseline-live 是 V2 三个实际首轮的对照记录。
"""


def bundles(source,evidence):
    files=set()
    for folder in (HERE,HERE.parent/"V2",HERE.parent/"V1"):
        files.update(f for f in folder.iterdir() if f.is_file() and f.suffix in (".py",".md",".txt") and f.name!="AGENTS.md")
    for name in ("prepare.py","README.md","来源声明.md","V2/gas.py","V2/bearing.py","V2/engine.py","app/baseline_model.py","app/industrial.py"):
        files.add(ROOT/"校赛"/name)
    with zipfile.ZipFile(source,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f,f.relative_to(ROOT))
        z.writestr("复现入口.txt","先阅读 企业赛/V3/README.md 与三个分项实验说明。代码按服务器仓库相对目录组织。授权原始数据、SDK 与官方起点按赛方渠道获取。校赛/prepare.py 构造公开输入。依赖、版本、训练与复核命令在说明文档。比赛生产依赖 Python、NumPy 和赛方 SDK；实验与材料制作另使用 SciPy、h5py、PyMuPDF、Pillow 及 FFmpeg。")
    entries={}
    for folder in ("replay","resources","controller","gas","bearing","engine","live"):
        for f in (WORK/folder).rglob("*.json"):
            if f.stat().st_size>3_000_000:
                raise ValueError("Unexpectedly large evidence: "+str(f))
            entries[str(f.relative_to(WORK))]=f
    for fname in ("freeze.json","protocol.json","paired-summary.json","checkpoint-io.json","sdk-0.log","sdk-1.log","qualification-3.log","qualification-4.log","quiet-0.log","quiet-1.log","quiet-2.log"):
        entries[fname]=WORK/fname
    for k in TASKS:
        entries[f"baseline-live/{k}/round.json"]=WORK.parent/f"v2/live/{k}/round.json"
    with zipfile.ZipFile(evidence,"w",zipfile.ZIP_DEFLATED) as z:
        for name,f in sorted(entries.items()):
            z.write(f,name)
        z.writestr("公开证据索引.txt",GUIDE)
    for path in [source,evidence]:
        if path.stat().st_size>50_000_000:
            raise ValueError("ZIP exceeds 50 MB")
        with zipfile.ZipFile(path) as z:
            if z.testzip() or any(name.endswith(("AGENTS.md",".npz",".env",".toml",".key",".h5")) for name in z.namelist()):
                raise ValueError("Unexpected archive entry or CRC failure")


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


def main():
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument("--preview",action="store_true",help="Render the report for layout review")
    args=parser.parse_args()
    summary,runs,live=load()
    validate_checks()
    DELIVERY.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        stage=Path(tmp); path=stage/"企业赛V3技术报告.pdf"
        d=report(summary,runs,live)
        d.set_metadata({"title":TITLE+" · 企业赛 V3","author":"McCarthy","subject":"公开冻结复验及部署证据"})
        d.subset_fonts();d.save(path,garbage=4,deflate=True);d.close()
        audit_pdf(path)
        if args.preview:
            print("Report rendered and checked");return
        video=stage/"企业赛V3运行回放.mp4"; movie(runs,live,video,stage)
        src=stage/"企业赛V3源码.zip";ev=stage/"公开实验证据.zip";bundles(src,ev)
        cfg=stage/"模型配置.txt";shutil.copyfile(HERE/cfg.name,cfg)
        outputs=[path,video,src,ev,cfg]
        sums=stage/"SHA256SUMS.txt"
        sums.write_text("".join(sha(f)+"  "+f.name+"\n" for f in [DELIVERY/"harness.tar",*outputs]))
        for f in [*outputs,sums]:
            f.replace(DELIVERY/f.name);print(f.name,(DELIVERY/f.name).stat().st_size,flush=True)


if __name__=="__main__":
    main()
