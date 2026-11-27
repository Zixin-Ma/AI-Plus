"""Build and verify enterprise V5 submission materials from the frozen public evidence."""
import hashlib
import html
import json
import re
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
WORK=HERE.parent/"work/v5"
DELIVERY=WORK/"delivery"
TASKS="ABC"
VERSIONS=("v4","v5")
CASES=("drift","fresh","official","fresh-measured","trace")
NAMES={"trace":"按 V4 平台轨迹","drift":"逐步释放","fresh":"回访换选型样本","official":"固定官方池","fresh-measured":"回访换样本＋按测量裁决"}
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
    p.insert_text((34,30),"BJTU INDUSTRIAL AI     /     ENTERPRISE V5",fontname="hebo",fontsize=10,color=NEW)
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
    return range(11) if s in ("drift","official","trace") else range(3,11)


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
            ids=counts(case) if section=="dev" else (11,12,13)
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
    for j,(label,c) in enumerate([("V4",BASE),("V5",NEW)]):
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



def sdk_rows(same=False):
    data=read(WORK/"sdk-latency.json")["rows"]
    out=[]
    for task in TASKS:
        cols=[]
        for version in VERSIONS:
            source="v4" if same else version
            rows=[v for v in data if v["source"]==source and v["runtime"]==version and v["task"]==task]
            assert len(rows)==5 and all(r["n"]>0 for r in rows)
            cols.append([statistics.median(r[k] for r in rows) for k in ("p50_us","p95_us","cold_process_ms")])
        out.append([task,*[pair(a,b,digits=2) for a,b in zip(*cols)]])
    return out


def expert_streams():
    frozen=read(WORK/"bearing/frozen.json")
    folds=read(WORK/"bearing/folds.json")
    return frozen,folds


def task_rows(summary,case,keys=("A","B","C"),section="confirm"):
    return [[k,*comparison(summary,case,k,section=section)] for k in keys]


def measurement_case(s,key,case):
    return [[field,*[f'{s["confirm/"+case][v][key][metric]*mult:.{digits}f}' for v in VERSIONS]]
            for field,metric,mult,digits in (("平均阶段收益（%）","mean",100,2),("各流最低轮收益均值（%）","worst",100,2),
                                             ("所有轮次的单一最低收益（%）","lowest",100,2),("历史回落（百分点）","forget",100,2),
                                             ("恢复延迟（轮）","delay",1,3))]


def pair_text(a,b,field,scale=1,digits=2):
    return pair(a[field],b[field],scale,digits)


def shape_report(doc,summary,runs):
    page(doc,"01   /   FROZEN COMPARISON","企业赛 V5 · 漂移流与寿命尾部",
         para("三项离线工业任务共用十轮诊断、更新与部署流程。V5 重点检验发动机临近失效时的过晚预测，同时保留轴承最差阶段，并在气体识别上分别考察日期外推和在线回访。")+
         table(["新留出主漂移，九条流各十轮","V4","V5"],measurement_case(summary,"all","drift"))+
         table(["各流最低轮收益，V4 → V5（%）","顺序 11","顺序 12","顺序 13"],
               selected_initial(runs,"drift",(11,12,13)))+
         para("三条冻结后开启的新顺序复用公开数据行，改变时间与审计分块。主漂移的平均收益及各流低谷均改善，历史回落增大。表中收益为公开审计模拟值。","note"))
    page(doc,"02   /   PROTOCOL","五种回放分开检验适应、遗忘与恢复",
         para("每轮只使用当时已开放训练标签与选型行。审计集的分类 Macro-F1 或寿命 RMSE 与同阶段静态起点比较，归一化后统一以正值表示改善。先取每条流最低轮，再跨流平均，得到表中的“各流最低”。")+
         table(["新留出，各 9 条流","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[NAMES[case],*comparison(summary,case)] for case in CASES])+
         para("逐步释放从可见子集开始续训。fresh 在同一阶段重访时更换选型样本。固定池从完整官方起点开始。按测量裁决使用可见指标审核候选。trace 将 V4 平台可见的样本数比例和规划动作投影到公开数据，各版本接受同一动作。")+
         table(["已知开发流","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[NAMES[case],*comparison(summary,case,section="dev")] for case in ("drift","trace")]))
    page(doc,"03   /   TASK TRADEOFFS","从各流最低轮定位性能变化",
         table(["主漂移","平均收益 %","各流最低 %","历史回落点","恢复轮"],task_rows(summary,"drift"))+
         table(["回访时更换选型样本","平均收益 %","各流最低 %","历史回落点","恢复轮"],task_rows(summary,"fresh"))+
         table(["按实际发布节奏与动作回放","平均收益 %","各流最低 %","历史回落点","恢复轮"],task_rows(summary,"trace"))+
         para("主漂移中 A 的低谷改善，但其历史回落扩大；C 的低谷和回落同步改善。按平台发布节奏回放时，三任务平均低谷从 5.07% 降至 4.09%，历史回落从 5.77 缩至 4.59 个百分点。","note"))


def bearing_page(doc):
    freeze,folds=expert_streams()
    ref,cur="baseline","endpoints"
    b=folds["FEMTO"]["summary"];interval=b[cur]["difference_interval"]
    protocols=freeze["streams"]
    pairs=sum(it["equal_choices"] for it in protocols.values())
    page(doc,"05   /   BEARING RUL","B · 稀疏观测中识别已有的寿命终点",
         para("V4 以近终点比例 5% 判断寿命覆盖。V5 在比例达到 2.5%、末端至少 12 行时，还检查是否确实观察到至少三种不同的零寿命双轴频带指纹。重复记录只算一次。改动发生在拟合时，沿用当前工况、频带秩和强冲击保护。")+
         table(["实际部署或整机测量（分钟）","V4","V5"],[
             ["XJTU 九台宏均 / 最坏",*[f'{folds["XJTU"]["summary"][v]["mean"]:.2f} / {folds["XJTU"]["summary"][v]["worst"]:.2f}' for v in (ref,cur)]],
             ["FEMTO 六台宏均 / 最坏",*[f'{b[v]["mean"]:.2f} / {b[v]["worst"]:.2f}' for v in (ref,cur)]],
             ["KAIST 距末次记录 RMSE",*[f'{folds["kaist"][v]["rmse"]:.2f}' for v in (ref,cur)]],
             ["完整公开首轮守门后 RMSE",*[f'{freeze["complete"][v]["rmse"]:.2f}' for v in VERSIONS]]])+
         para(f'在已知 0–10 的四种十轮回放中，{pairs} 对轮次的部署选择一致，阶段 RMSE 最大差为零。FEMTO 六台中 {b[cur]["wins"]} 台改善，设备配对差区间为 [{interval[0]:+.2f}, {interval[1]:+.2f}] 分钟。该资料参与方案开发，区间跨零。')+
         para("按历史超额误差或局部残差包络回缩，能改善部分外部设备，但降低 XJTU 最坏设备和已知逐批流最低轮的表现。门槛使用当轮训练标签和观测指纹，不使用设备身份或未来时刻。","note"))


def cold_report():
    data=read(WORK/"cold-start.json")
    pairs=[]
    for key in TASKS:
        both=[]
        for var in ("plain","cache"):
            times=[v["cold_process_ms"] for v in data["runs"] if v["task"]==key and v["variant"]==var]
            if len(times)!=7:raise ValueError("Seven cold-start repeats required")
            both.append(statistics.median(times))
        pairs.append(key+" "+pair(*both,digits=1))
    return "同一 V5 仅改变字节码缓存、七次时序交替的冷首包中位（毫秒），无缓存 → 有缓存："+"，".join(pairs)+"。"
    

def resource_page(doc):
    page(doc,"08   /   SDK LATENCY","真实镜像逐条往返与冷进程",
         para("同核交替五遍，每条写入后等待一条完整 SDK 回复。表内为五遍的中位。换用各版自身训练产物时，A 与 C 热 p50 上升，B 略降；三项冷首包均缩短。")+
         table(["各版自身训练产物","热 p50（微秒）","热 p95（微秒）","冷首包（毫秒）"],sdk_rows())+
         table(["相同 V4 检查点","热 p50（微秒）","热 p95（微秒）","冷首包（毫秒）"],sdk_rows(same=True))+
         para(cold_report())+
         table(["单核十轮资源","训练 CPU 秒","单行预测微秒","候选合计 MB","决策调用数"],resource_rows()))


def historical_trial():
    reference=WORK.parent/"v4/platform/企业赛V4试运行结果.md"
    text=reference.read_text()
    calls=int(re.search(r"模型调用 ([0-9,]+) 次",text).group(1).replace(",",""))
    tokens=int(re.search(r"合计 ([0-9,]+)\S*。三任务",text).group(1).replace(",",""))
    nasa=float(re.search(r"nasa_score ([0-9.]+)",text).group(1))
    rmse=float(re.search(r"nasa_score [0-9.]+，rmse ([0-9.]+)",text).group(1))
    return reference,calls,tokens,nasa,rmse


def receipt_page(doc,summary,live):
    _,base_calls,base_tokens,nasa_c,rmse_c=historical_trial()
    calls=[]
    all_receipts={}
    for k,(record,items) in live.items():
        cfg=record["configuration"]
        for obj in items:all_receipts[obj["id"]]=obj
        calls.append([k,cfg["train_rows"],cfg["validation_rows"],cfg["plan"]["tool"],cfg["deployed"],
                      sum(c["usage"]["total_tokens"] for c in items),"/".join(sorted({i["provenance"] for i in items}))])
    paid=[r for r in all_receipts.values() if r.get("provenance")=="v5-api"]
    page(doc,"09   /   TRACE & DECISIONS","平台回执基准与离线复演",
         para(f"V4 平台试运行共 {base_calls} 次调用、{base_tokens:,} Token；其 C 隐藏端点为 RMSE {rmse_c:.3f} 循环、NASA {nasa_c/1e8:.2f} 亿。V5 的公开 trace 仅复用可见的数量比例和动作；V5 各轮仍按独立公开审计行计分。")+
         table(["V5 本轮宿主首轮验证","训练","选型","规划动作","部署","回执 Token","来源"],calls)+
         para(f'本次三任务首轮记录含 {len(all_receipts)} 个独立调用 ID；其中本次新请求 {len(paid)} 次、共 {sum(r["usage"]["total_tokens"] for r in paid)} Token。相同请求摘要可复用旧回执。十轮公开矩阵的确定性裁决不产生线上模型费。')+
         table(["trace 新留出","均值 %","各流最低 %","历史回落点","恢复轮"],task_rows(summary,"trace"))+
         para("工业模型更新、预测和回滚在 CPU 容器内进行。宿主控制模型通过白名单动作和限额参与计划与裁决，回执记录不携带账户凭据。","note"))


def evidence_page(doc):
    verified=read(WORK/"verification.json")
    econ=read(WORK/"resource-controls.json")
    page(doc,"10   /   REPRODUCIBILITY","重建与核对",
         table(["发布件","核查入口"],[["公开实验证据.zip","paired-summary.json 汇总各场景，replay/ 提供每轮候选与审计，freeze.json、protocol.json 提供运行代码及输入摘要。"],
          ["企业赛V5源码.zip","包含冻结版代理、各任务实验、控制器评估、构建与验收入口，以及复现直接依赖的前版源码。"],
          ["延迟与资源","sdk-latency.json 记录真实镜像和管道逐条往返，serving/ 区分角色，resources/ 为重复的单核资源回放。"],
          ["外部与消融","gas/、bearing/、engine/ 分任务记录，controller/ 保留控制器与专家对照，live/ 记录模型回执。"]])+
         para(f'独立核验覆盖 {2*verified["pairs"]} 个版本记录和 {verified["recorded_rounds"]} 个版本轮次。对应训练、选型和审计数逐轮配对。数据、授权起点和 SDK 由相应授权渠道取得。')+
         para("模型配置说明 CPU 环境、平台模型、决策限额、输入输出和执行协议。核对 SHA256SUMS.txt 后加载镜像与授权模型目录；异常时使用最近可用父产物。")+
         para(f'控制器的分组标签在三个样本规模上成员归属一致。独立微基准分组耗时 {econ["group_cpu_ms"][0]:.1f} → {econ["group_cpu_ms"][1]:.1f} 毫秒，历史上下文编码 {econ["context_bytes"][0]} → {econ["context_bytes"][1]} 字节。')+
         para("方法与数据参考　<a href='https://ieeexplore.ieee.org/abstract/document/11763722'>电子鼻置信判断研究</a>，<a href='https://arxiv.org/abs/2607.08273'>轴承工况迁移下的可靠性评估</a>，<a href='https://doi.org/10.36001/ijphm.2026.v17i1.4724'>寿命非对称区间研究</a>，<a href='https://doi.org/10.3390/data6010005'>N-CMAPSS 数据论文</a>，<a href='https://github.com/NostalgiaJohn/CruiseBench'>CruiseBench 开源项目</a>。三份专项说明列出更多文献以及工程改动的验证边界。","note"))


def gas_page(doc,s):
    base=WORK/"gas"
    cross=read(base/"cross-day-angular.json")
    def row(name,r,show_ci=False):
        old,new=[r["families"][v]["f1"] for v in VERSIONS]
        interval=r.get("paired_date_ci95")
        effect=f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if show_ci and interval else "—"
        return [name,r["test_n"],f"{old:.4f}",f"{new:.4f}",effect]
    page(doc,"04   /   GAS IDENTIFICATION","A · 证据不一致时收缩动力学票",
         para("尾部专家分别由传感器动力学和原始响应子空间给出类对数证据。V5 对两个类别向量中心化，按余弦一致性决定限幅，最多削弱 30% 动力学证据，保留原独立响应专家。改动不新增训练参数。")+
         table(["独立跨域设置","审计数","V4 Macro-F1","V5 Macro-F1","按日配对差区间"],
               [row("一年电子鼻历史窗",cross["historical"],True),row("一年电子鼻末窗",cross["terminal"],True),
                row("DS-PS",read(base/"ds-ps-angular.json")),row("高浓度并跨重复",read(base/"cross-concentration-angular.json")),
                row("UCI 362 家用设备末期",read(base/"home-angular.json"))])+
         para("日期窗口与外部平台上两版表现相同，高浓度跨重复的识别能力仍较弱。UCI 362 以完整传感器记录为样本，按日期留后，原始记录缺失的编号 95 未使用。")+
         table(["新留出 A","平均收益 %","各流最低 %","历史回落点","恢复轮"],
               [[NAMES[case],*comparison(s,case,"A")] for case in ("drift","fresh","trace")])+
         para("开发回放和新顺序逐轮使用不同的选型与审计分块，外部域端点单列，避免把同一试验重复当作独立证据。","note"))


ENGINE_ROUTE="frozen"
ENGINE_EVIDENCE="frozen"
ENGINE_METHOD="V5 在训练池扩张时，让训练梯度与部署限幅和双尾非对称代价相匹配；在固定池重拟合时，三项寿命顺序概率送入使用保留行标定的线性读出。标定候选只有在高估和低估两侧的截尾代价都不增加时才可采用。"
ENGINE_LIMITS="固定池的总 NASA 代价下降，但末寿命组高估超过 30 循环的概率仍上升。第一整机折的整体 RMSE 下降，高估侧代价却增加。不同端点分开报告，保留异常工况复核与父产物回退。"


def engine_result(name,route=None):
    # Production route and evidence filename are pinned explicitly when the engine head is frozen.
    selected=ENGINE_ROUTE if route is None else route
    path=WORK/"engine"/f"{name}-{ENGINE_EVIDENCE if route is None else route}.json"
    data=read(path)
    return [max((z for z in data if z["route"]==r),key=lambda z:z["round"]) for r in ("base",selected)]


def controller_effect():
    folder=WORK/"controller/v5"
    names=sorted(p.name for p in (folder/"base").glob("trace-*.json"))
    if len(names)!=5:raise ValueError("Five development controller trace pairs required")
    summaries=[]
    for mode in ("base","origin"):
        flows=[read(folder/mode/name) for name in names]
        summaries.append([100*statistics.mean(r[k] for r in flows) for k in ("mean","worst","forget")]
                         +[statistics.mean(row["nasa_mean"] for r in flows for row in r["rounds"])])
    stats=[pair(a,b,digits=0 if i==3 else 2) for i,(a,b) in enumerate(zip(*summaries))]
    return ("C 的 fit_release 保存实际部署头的拟合数据指纹。固定同一专家的五条开发 trace 中，"
            f"启用来源校验后，平均收益 {stats[0]}%，各流最低 {stats[1]}%，回落 {stats[2]} 点；每轮 NASA 样本均值 {stats[3]}。")


def deployed_nasa(case):
    checks=read(WORK/"verification.json")
    paired=[row["mean_per_round_nasa"] for row in checks["streams"]
            if row["part"]=="confirm" and row["setting"]==case and row["task"]=="C"]
    if len(paired)!=3:raise ValueError("Three C confirmation trajectories required")
    lhs,rhs=(statistics.mean(series) for series in zip(*paired))
    return f"{lhs:.3g} → {rhs:.3g}"


def engine_pages(doc,s):
    timing=read(WORK/"engine/speed.json")
    stable_cpu=[statistics.median(z["single_us"] for z in timing if z["route"]==route and z["round"]==2) for route in ("base","frozen")]
    items=[(label,*engine_result(dataset)) for label,dataset in
           (("完整官方固定池","official"),("逐批发布末轮","drift"),("整机留出折 1","fold1"),
            ("整机留出折 2","fold2"),("N-CMAPSS DS02","DS02"))]
    page(doc,"06   /   ENGINE LATE-RISK","C · 同时核对整体误差与过晚预测",
         para(ENGINE_METHOD)+
         table(["最后一次释放","RMSE（循环）","NASA 总代价","高估部分","低估部分"],
               [[name,pair_text(a,b,"rmse",digits=3),f'{a["nasa"]:.3g} → {b["nasa"]:.3g}',
                 f'{a["nasa_late"]:.3g} → {b["nasa_late"]:.3g}',f'{a["nasa_early"]:.3g} → {b["nasa_early"]:.3g}'] for name,a,b in items])+
         para("NASA 对延寿高估和保守低估分别施加指数代价。各行来自相同域、相同审计行与同一释放轮次，适宜逐行配对。不同样本规模和寿命分布的绝对代价不宜横向相加。")+
         para("DS02 按整台发动机训练与留出，巡航掩码来自 CruiseBench，外部权重不进入参赛产物。稳定专家的独立热单行耗时为 "+pair(*stable_cpu,digits=2)+" 微秒。","note"))
    page(doc,"07   /   ENGINE TAIL & DRIFT","C · 最弱阶段和尾部风险分开计量",
         table(["直接模型端点","误差绝对值第90百分位（循环）","末寿命组高估超30循环"],
               [[name,pair_text(a,b,"q90_abs",digits=2),pair_text(a,b,"terminal_over30",100,2)+" %"] for name,a,b in items[:4]])+
         table(["新留出 C","平均收益 %","各流最低 %","历史回落点","恢复轮","每轮样本均值 NASA"],
                [[NAMES[case],*comparison(s,case,"C"),deployed_nasa(case)] for case in CASES])+
         para(ENGINE_LIMITS)+
         para(controller_effect(),"note"))


def report(summary,runs,live):
    doc=pdf.open()
    shape_report(doc,summary,runs)
    gas_page(doc,summary)
    bearing_page(doc)
    engine_pages(doc,summary)
    resource_page(doc)
    receipt_page(doc,summary,live)
    evidence_page(doc)
    return doc


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




GUIDE="""公开复核证据
paired-summary.json 按开发或确认、场景、版本及任务汇总。replay/ 保存指标、动作、部署、阶段、可见样本数和审计记录。mean 为十轮平均收益；worst 为单流最差阶段；lowest 为跨流单一最低阶段；forget 为曾见阶段最大回落；delay 为切换后恢复所需轮数。公开回放不产生平台正式成绩。
V4-trial/企业赛V4试运行结果.md 是历史基线试运行摘要，其隐藏测试指标只作改进起点。trace 从既有 V4 平台记录抽取三项任务十轮的样本数和控制动作；认证指标仍来自本次公开审计。gas/、bearing/、engine/ 和 controller/ 分别包含专项及系统消融。Resources 和 SDK 计时分别提供训练资源与单条完整往返的冷、热分布。
训练数据、权重、原始轨迹、密钥及付款信息不包含在证据 ZIP 中。固定规划回放、相同请求摘要的旧回执及真实新请求分开计量。
"""


def verify_release_inputs():
    verified=read(WORK/"verification.json")
    if verified.get("status")!="passed":raise ValueError("Independent matrix verification incomplete")
    smoke=(WORK/"image-proof-0.log").read_text()
    if not all(f"{k} round {n}:" in smoke for k in TASKS for n in (1,2,3)):
        raise ValueError("Incomplete SDK image exercise")
    if not all((HERE / name).is_file() for name in ("README.md","模型配置.txt","轴承实验.md","气体实验.md","发动机实验.md")):
        raise ValueError("Missing project configuration or a task-level reproducibility report")

def bundles(source,evidence):
    files=set()
    for folder in (HERE,HERE.parent/"V4",HERE.parent/"V3",HERE.parent/"V2",HERE.parent/"V1"):
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
        z.writestr("复现入口.txt","使用本包原有的仓库相对目录。企业赛/V5/README.md 与三份实验说明给出镜像、授权投影和回放命令；校赛/prepare.py 处理授权数据。Python/NumPy 为生产计算依赖，比较与材料阶段另需 h5py、SciPy、Pillow 与 PyMuPDF。评测 SDK、官方权重、比赛数据及补充数据由各授权渠道获取。")
    paths={}
    for folder in ("replay","resources","controller","serving","gas","bearing","engine","live"):
        for f in (WORK/folder).rglob("*.json"):
            if f.stat().st_size>3_000_000:raise ValueError("Unexpected evidence size "+str(f))
            paths[f.relative_to(WORK).as_posix()]=f
    for f in (WORK/"freeze.json",WORK/"protocol.json",WORK/"paired-summary.json",WORK/"verification.json",WORK/"sdk-latency.json",WORK/"resource-controls.json",WORK/"cold-start.json",WORK/"image-proof-0.log"):
        paths[f.name]=f
    historical,*_=historical_trial()
    paths["V4-trial/"+historical.name]=historical
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
        stage=Path(tmp); path=stage/"企业赛V5技术报告.pdf"
        d=report(summary,runs,live)
        d.set_metadata({"title":TITLE+" · 企业赛V5","author":"McCarthy","subject":"公开冻结复验、跨域及单条延迟"})
        d.subset_fonts();d.save(path,garbage=4,deflate=True);d.close()
        audit_pdf(path)
        if args.preview:
            print("Report preview rendered to",WORK/"visual-qa");return
        src=stage/"企业赛V5源码.zip";ev=stage/"公开实验证据.zip"; bundles(src,ev)
        cfg=stage/"模型配置.txt";shutil.copyfile(HERE/cfg.name,cfg)
        outputs=[path,src,ev,cfg]
        sums=stage/"SHA256SUMS.txt"
        sums.write_text("".join(sha(f)+"  "+f.name+"\n" for f in [DELIVERY/"harness.tar",*outputs]))
        for f in [*outputs,sums]:
            f.replace(DELIVERY/f.name)
            print(f.name,(DELIVERY/f.name).stat().st_size,flush=True)


if __name__=="__main__":
    main()
