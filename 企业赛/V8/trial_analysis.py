"""Generate the settled three-version platform comparison from official run receipts."""
from pathlib import Path
from statistics import mean
import json
from release import WORK,ROOT,DEST,VERS,read,platform,write_document,pack,pct,triple

def main():
    info=platform()
    base=WORK.parent
    paths=(base/"v5/formal/formal-report.json",base/"v7/platform-trial/report.json",WORK/"platform-trial/report.json")
    reports=[read(p)["report"] for p in paths]
    sets=[{a["key"]:a for a in r["workspace_report"]["datasets"]} for r in reports]
    label={"accuracy":"准确率 ↑","macro_f1":"宏 F1 ↑","rmse":"RMSE ↓","mae":"MAE ↓","nasa_score":"NASA 代价 ↓"}
    units={"A":"","B":" 分钟","C":" 周期"}
    rows=[];wins={}
    for metric,val in info["metrics"].items():
        k,m=metric.split(".",1)
        higher=k=="A"
        more=(val[-1]>val[0]) if higher else (val[-1]<val[0])
        wins.setdefault(k,[]).append(more)
        fmt=lambda x:f"{x:,.0f}" if abs(x)>1.e7 else f"{x:.6f}" if higher else f"{x:.3f}"
        rows.append([k,label.get(m,"NASA 代价 ↓" if "nasa" in m else m)+ (units[k] if m in ("rmse","mae") else ""),
                     *(fmt(x) for x in val),f"{100*(val[-1]/val[0]-1):+.3f}%"])
    phrases=["、".join(k for k in "ABC" if all(wins[k])) or "无"]
    wins_n=sum(sum(vals) for vals in wins.values())
    total_n=sum(len(vals) for vals in wins.values())
    pages=[("平台隐藏集 · 最终部署工件",f"本次 V8 试运行三项各完成 10 轮。以 V5 正式结果为参照，{total_n} 个终态指标中 {wins_n} 项改善，各自全部指标改善的任务为 {phrases[0]}。",["任务","指标","V5 正式","V7 试运行","V8 试运行","V8 较 V5"],rows,
    ["变化率为数值的相对变化。A 的两个指标越高越好，B、C 的误差和 NASA 代价越低越好。NASA 对高估剩余寿命的较大误差惩罚更重。",
     "最终部署工件的指标覆盖 4,886 个气体样本、200 个轴承样本和 9,687 个发动机样本。同一隐藏集上只有聚合值，无法计算逐样本配对置信区间。"])]
    rr=[]
    for k in "ABC":
        counts=[d[k]["prediction_count"] for d in sets]
        assert len(set(counts))==1
        idx=[]
        for ds in sets:
            rs=ds[k]["rounds"]; art=rs[-1]["selected_artifact_id"]
            matches=[r["round_index"] for r in rs if any(c.get("artifact_id")==art for c in r.get("candidates",[]) if isinstance(c,dict))]
            idx.append(str(matches[-1]) if matches else "见回执")
        rr.append([k,f"{counts[0]:,}",*idx,triple([d[k]["cpu_seconds"] for d in sets],lambda x:f"{x:,.1f}")])
    cpu=info["cpu_seconds"];tok=[c["total_tokens"] for c in info["costs"]]
    rt=[c["training_seconds"] for c in info["costs"]]
    pages.append(("平台资源与模型选择","平台运行相同评测配置。每行最后一列按 V5 / V7 / V8 顺序排列，CPU 秒数由平台结算。",["任务","隐藏样本","V5 最终轮","V7 最终轮","V8 最终轮","CPU 秒"],rr,
     ["合计 CPU 秒为 "+triple(cpu,lambda x:f"{x:,.1f}")+"。控制模型总 Token 为 "+triple(tok,lambda x:f"{x:,}")+"。平台训练计时为 "+triple(rt,lambda x:f"{x:,.1f}")+" 秒。",
      "V8 使用 1 次试运行；平台保留 1 次正式机会。V8 运行编号 "+info["runs"][-1]["id"]+"。"]))
    obj=read(base/"v5/formal/analysis.json")["objective"]["objective_components"]
    stats=read(WORK/"paired-summary.json")["confirm/anchored"]
    r=[]
    for k in "ABC":
        def pair(f,fmt):return " → ".join(fmt(stats[v][k][f]) for v in ("v5","v8"))
        r.append([k,pair("stage_mean",pct),pair("stage_floor",pct),pair("forget",pct),pair("delay",lambda x:f"{x:.2f}")])
    pages.append(("五项客观分的证据边界","平台仅对正式评测发布五项客观分。下表是官方公开起点、V5 发布时序、八个冻结划分的十轮代理结果，箭头左侧为 V5，右侧为 V8。",["任务","平均阶段收益","最低阶段收益","历史最大回撤 ↓","恢复延迟（轮）↓"],r,
      [f"V5 正式成绩 23.034385/35。漂移流综合 {obj['mean_gain']:.3f}/15，最低阶段 {obj['worst_stage']:.3f}/5，稳定与抗遗忘 {obj['retention']:.3f}/5，恢复 {obj['recovery']:.3f}/5，资源效率 {obj['resource_efficiency']:.3f}/5。V8 仅获得试运行结果。",
       "八个划分共享公开数据。A 在分批发布及跨域选型的最低阶段出现回撤；B 在多种协议中最低阶段仍为负；C 在交替漂移中的平均收益和恢复速度略有回撤。"]))
    rows=[]
    for split in (201,203,207):
        recs=[read(WORK/f"cloud/streams/anchored-{v}-B-{split}.json") for v in VERS]
        rows.append([str(split),*[pct(d["stage_floor"]) for d in recs],triple([d["stage_mean"] for d in recs],pct)])
    pages.append(("轴承 · 真实部署复核","固定任务头和公开数据，用真实 PyTRiO 规划、复核每轮部署。201 是原确认划分；观察到其回撤后，额外复核 203、207 两个划分。",["划分","V5 最低","V7 最低","V8 最低","V5 / V7 / V8 平均阶段收益"],rows,
       ["该实验测量同一工业模型在不同可见验证样本与模型决策下的稳定性，不能替代平台隐藏阶段打分。201 的主要损失出现在第一工况，模型获得更好可见指标时，该工况的隔离审计指标可能下降。",
        "可复核文件包含全部配对划分、逐轮调用回执摘要和冻结清单。技术依据、补充数据留出及消融写入气体、轴承、发动机三份分项实验。"]))
    write_document("企业赛V8平台试运行分析",pages)
    pack()
if __name__=="__main__":
    main()
