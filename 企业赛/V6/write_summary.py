"""Write the readable frozen-release comparison from the independent checks."""
import json
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v6"
def read(f):
    return json.loads((OUT/f).read_text())

def main():
    score=read("scorecard.json")
    props=read("resource-summary.json")
    notes=read("release-notes.json")
    official=read("official-reference.json")
    buf=["# 企业赛 V6 冻结验证","",notes["decision"],"",
         "V5 官网正式总分 23.034385 / 35。漂移综合 9.079221，最差阶段 0，恢复 3.968254，稳定与抗遗忘 4.986910，资源效率 5。A、B、C 的最低阶段收益分别为 −0.375977、−0.211668、−0.358173。原始网站报告与 V5 本地正式评测分析已交叉核对。","",
         "实验从 V5 冻结镜像和源码出发，三条路线分别做开发和消融，随后锁定生产源码、评测器、九个输入文件及划分。六种协议、七个开发顺序和六个新确认顺序共有 234 对十轮流，独立复算 4,680 个逐轮记录。下文所有 V6 数据来自服务器本地。","",
         "## 冻结后按官网日程回放","",
         "|任务|阶段等权平均收益 V5 → V6|最低阶段收益 V5 → V6|最低阶段上升／持平／下降的划分数|",
         "|---|---:|---:|---:|"]
    def fmt(d):
        return f"{100*d['v5']:.2f}% → {100*d['v6']:.2f}%"
    for k in "ABC":
        r=score["schedule/"+k]; low=r["stage_floor"]
        buf.append(f"|{k}|{fmt(r['stage_mean'])}|{fmt(low)}|{low['increased']}／{low['tied']}／{low['decreased']}|")
    buf+=["","A 模拟八个类内采集顺序片段，B 按三个载荷/转速，C 按四个 FD 源子集。十轮访问次数沿用 V5 正式报告。每个阶段内部先对访问求均值，每条流再求平均与最低，各划分等权。A/B/C 的本地阶段收益是相对公开静态 Macro-F1 或 RMSE 的百分比，与官网内部阶段收益量纲不同。阶段身份、隐藏标签及正式计分映射未公开。","",
          "## 逐协议确认","","|任务／协议|阶段平均 V5 → V6|最低阶段 V5 → V6|最大历史回撤 V5 → V6（百分点）|恢复延迟 V5 → V6（轮）|","|---|---:|---:|---:|---:|"]
    for k in "ABC":
        for m in ("schedule","trace","official","drift","fresh","fresh-measured"):
            v=score[m+"/"+k]
            buf.append(f"|{k} · {m}|{fmt(v['stage_mean'])}|{fmt(v['stage_floor'])}|{100*v['forget']['v5']:.2f} → {100*v['forget']['v6']:.2f}|{v['delay']['v5']:.3f} → {v['delay']['v6']:.3f}|")
    buf+=["","schedule 按官网阶段数及访问顺序回放。trace 沿用平台公开样本释放比例与动作。official 从完整官方公开训练池起步。drift 逐步释放训练样本。fresh 在回访时更换选型样本，fresh-measured 另外根据当轮实测验证择优。调参可见 0–13 的旧顺序，整版本开发矩阵只采用其中七个；六个确认顺序为 14–19。共享公开样本的不同划分反映顺序敏感性。","",
          "## 任务机制与完整首批验证","",notes["A"],"",notes["B"],"",notes["C"],"",notes["external"],"",
          "|公开全首批|主要指标 V5 → V6|SDK p50 µs|SDK p95 µs|代表性十轮进程 CPU 秒|","|---|---:|---:|---:|---:|"]
    for k in "ABC":
        v=props[k]; a,b=v["v5"],v["v6"]
        buf.append(f"|{k}|{a['metric']:.6f} → {b['metric']:.6f}|{a['p50_us']:.1f} → {b['p50_us']:.1f}|{a['p95_us']:.1f} → {b['p95_us']:.1f}|{a['cpu']:.2f} → {b['cpu']:.2f}|")
    buf+=["",notes["resources"],"","发动机当前阶段每行 NASA 代价（更低为优），高估和低估分别按 10、13 循环计指数惩罚。","",
          "|协议|V5|V6|变化|","|---|---:|---:|---:|"]
    for m in ("schedule","trace","official","drift","fresh","fresh-measured"):
        v=score[m+"/C"]["nasa"];buf.append(f"|{m}|{v['v5']:,.1f}|{v['v6']:,.1f}|{100*(v['v6']/v['v5']-1):+.1f}%|")
    buf+=["","## 版本结论","",notes["recommendation"],"","企业赛 V5 官网正式运行 ID 为 "+official["run"]+"。官网 V4/V5 试运行、V5 正式客观分和逐阶段指标的可读摘录见证据包 official-reference.json；全量原始对照仍保存在企业赛/work/v5/formal 及 platform。","","完整配对结果见 scorecard.json、audit.json 和 replay/。公开决策回放采用确定性宿主。镜像走 SDK 真实的 train、predict 和工件回滚路径；这次本地研究没有调用付费大模型，也没有提交平台试运行或正式评测。",""]
    dest=OUT/"delivery/企业赛V6验证汇总.md"
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text("\n".join(buf))

if __name__=="__main__":
    main()
