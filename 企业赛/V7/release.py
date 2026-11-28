"""Build a human-readable V7 report and signed source, image and experiment packages."""
import hashlib
import html
import json
import re
import statistics as st
import subprocess
import zipfile
from pathlib import Path

import pymupdf as pdf

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORK=HERE.parent/"work/v7"
OUT=WORK/"delivery"
NAMES={"A":"气体识别","B":"轴承寿命","C":"发动机寿命"}
CASES={"schedule":"官网日程代理","transfer":"当前工况退出选型","official":"完整官方起点","trace":"按发布比例分阶段","drift":"训练池逐批增长","fresh":"回访更换选型行","fresh-measured":"回访按实测选择"}
def read(f):return json.loads((WORK/f).read_text())
def sha(f):
    h=hashlib.sha256()
    with f.open("rb") as s:
        for block in iter(lambda:s.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def xy(a,b,s=100,n=2):
    return f"{a*s:.{n}f} → {b*s:.{n}f}"
def measure(d,s=100,n=2):return xy(d["v6"],d["v7"],s,n)

def sections():
    card=read("scorecard.json")
    data=read("evidence.json")
    hw=read("resource-summary.json")
    n=read("release-notes.json")
    sections=[]
    def add(title,intro,header,rows,notes):
        sections.append((title,intro,header,rows,notes))
    rows=[]
    for k in "ABC":
        d=card["schedule/"+k];f=d["stage_floor"]
        rows.append([NAMES[k],measure(d["stage_mean"]),measure(f),f"{f['increased']}/{f['tied']}/{f['decreased']}",f"{d['positive_floor'][0]}/8 → {d['positive_floor'][1]}/8"])
    add("企业赛 V7 · 冻结验证",n["decision"],["任务","阶段平均 %","最低阶段 %","提高/平/降","低谷为正的划分"],rows,[
        "每格从 V6 到 V7。三任务各有八个冻结后执行的预留编号。阶段内按访问轮次取均值，最低阶段是每条流各阶段均值的最小值。这里的百分比以同一公开静态起点为参照。",
        "McCarthy 的平台回执列出 V5 试运行和 V5 正式。V5 正式 23.034385/35，最低阶段 0/5，A1、B3、C3 在官网最低。本次按队伍选择，直接对比冻结 V6 的公开回放；真实控制模型验证使用 PyTRiO。"])
    official=read("platform-reference.json")["reports"]["80900572e91d7d33d6637088915dc5e6"]["objective"]
    primary=[card["schedule/"+k] for k in "ABC"]
    def avg(m,ver):return st.mean(t[m][ver] for t in primary)
    components=[("漂移流综合性能","mean_gain",15,"三任务阶段平均收益 %","stage_mean",100),
                ("最差阶段表现","worst_stage",5,"三任务最低收益等权 %","stage_floor",100),
                ("恢复速度","recovery",5,"转换后恢复所需轮数","delay",1),
                ("稳定性与抗遗忘","retention",5,"历史最大回撤 pp","forget",100)]
    rows=[[label,f"{official[name]:.4f}/{total}",proxy,xy(avg(m,"v6"),avg(m,"v7"),scale,3)] for label,name,total,proxy,m,scale in components]
    live_tokens=[sum(t[v]["tokens"] for t in data["cloud_runs"].values()) for v in ("v6","v7")]
    rows.append(["资源效率",f"{official['resource_efficiency']:.4f}/5","六条真实流引用 Token",str(live_tokens[0])+" → "+str(live_tokens[1])])
    add("五项客观成绩的参照与代理","官网列为 V5 正式实分。V6 与 V7 列使用相同协议及划分，给出与评分关注点相对应的实测量。",["客观项","V5 官网","本地量纲","V6 → V7"],rows,["前四行对三项任务等权。每个任务的最低收益先在每条流内求最小，再对预留划分取均值。三任务等权值会掩盖仍处于负值的工况，详见分任务页。恢复轮数、回撤及 Token 越低越好。","资源效率还涉及 CPU、工件与推理成本。资源效率页给出服务器训练和 SDK 逐条往返的测量。评测平台在隐藏流上独立结算各项分数。"])
    for k in "ABC":
        rows=[]
        for mode,title in CASES.items():
            d=card[mode+"/"+k];rows.append([title,measure(d["stage_mean"]),measure(d["stage_floor"]),measure(d["forget"]),measure(d["delay"],1,3)])
        add(NAMES[k]+"的低谷与平均收益",n[k],["确认协议","阶段平均 %","最低阶段 %","最大回撤 pp","恢复轮数"],rows,[n[k+"_limit"]])
    rows=[]
    for k in "ABC":
        z=data["opening"][k];a,b=z["v6"],z["v7"]
        rows.append([NAMES[k],f"{a['groups']} × {a['rounds']}",xy(a["mean"],b["mean"]),xy(a["floor"],b["floor"])])
    add("完整首批的跨工况检查","从官方完整训练池和检查点起步，每次把一个完整工况排除在选型之外，连续重放首次发布内的所有轮次。",["任务","组数 × 轮数","组间平均收益 %","最弱组收益 %"],rows,[n["grid"],"A 的八组来自一个公开气体批次。B 的三个载荷各自对应一台验证设备。C 有四个 FD 子集，阶段池审计按整台发动机隔离。公开编号与隐藏工况的对应关系未知。"])
    rows=[]
    for m,title in CASES.items():
        v=card[m+"/C"]["nasa"];rows.append([title,f"{v['v6']:,.1f}",f"{v['v7']:,.1f}",f"{100*v['delta']/v['v6']:+.1f}%"])
    add("发动机的非对称代价","先对当轮审计样本计算每行 NASA，再按轮次与划分等权平均。官网隐藏集的 NASA 总和另有样本范围。",["协议","V6 每行 NASA","V7 每行 NASA","变化"],rows,[n["external"]])
    rows=[]
    for i,case in data["targeted_nasa"].items():
        for method,label in (("fixed","固定动作"),("qwen","真实 Qwen")):
            old,new=case[method+"_v6"],case[method+"_v7"]
            rows.append([i,label,xy(old["nasa"],new["nasa"],1,0),xy(old["fd004"],new["fd004"],1,0),f"{(new['nasa']/old['nasa']-1)*100:+.1f}%"])
    add("发动机高代价工况复核","查看八个日程划分后，定位到 NASA 增量最大的 103 和 108，随即固定这两个案例追加真实模型对照，两版的参数和审计行均保持冻结。",["编号","控制器","全流每行 NASA","FD004 每行 NASA","全流变化"],rows,[n["tail"],"这里按已知高风险结果选样，用于诊断，不能用两条流的均值估计所有工况。独立预留的三个真实模型日程结果另页列出。"])
    rows=[]
    for k in "ABC":
        for m,l in [("schedule","日程"),("official","完整起点")]:
            z=data["cloud_runs"][m+"/"+k];a,b=z["v6"],z["v7"]
            rows.append([k+" · "+l,xy(a["stage_floor"],b["stage_floor"]),xy(a["stage_mean"],b["stage_mean"]),str(a["calls"])+" → "+str(b["calls"]),str(a["tokens"])+" → "+str(b["tokens"])])
    add("真实 Qwen3.8-27B 控制的十轮流","两种公开回放各选一个已知划分，两版均运行十轮。PyTRiO 接收公开指标与诊断，为规划和部署复核返回实测决定；工业模型在服务器执行。",["流","最低阶段 %","阶段平均 %","调用次数","按流引用 Token"],rows,[n["cloud"],"相同请求按摘要复用真实回执。表内 Token 随每条流分别累计；整项研究实际请求去重后单列。"])
    rows=[]
    for k in "ABC":
        a,b=[data["cloud_reserved"][k][v] for v in ("v6","v7")]
        rows.append([NAMES[k],xy(a["stage_mean"],b["stage_mean"]),xy(a["stage_floor"],b["stage_floor"]),xy(a["forget"],b["forget"]),xy(a["delay"],b["delay"],1,2),xy(a["tokens"],b["tokens"],1,0)])
    add("冻结后的真实控制器复核","从八个预留日程划分中事前等间隔取 101、104、107，两个版本分别运行全部十轮，直接调用实际 Qwen。共 9 对流、180 个轮次。",["任务","平均收益 %","最低收益 %","回撤 pp","恢复轮数","每流 Token"],rows,[n["reserved"],"训练、选型及审计样本摘要逐轮与相应固定控制回放核对。这三个划分共享公开样本，结果用于估计真实规划及复核对升级方向的影响。"])
    rows=[]
    for k in "ABC":
        a,b=hw[k]["v6"],hw[k]["v7"]
        rows.append([k,xy(a["metric"],b["metric"],1,5),xy(a["p50_us"],b["p50_us"],1,1),xy(a["p95_us"],b["p95_us"],1,1),xy(a["cpu"],b["cpu"],1,2),xy(a["checkpoint_bytes"]/1024,b["checkpoint_bytes"]/1024,1,0)])
    add("资源效率与交付判断","完整首批上报告实际选中模型的 Macro-F1 或 RMSE。SDK 单条延迟为两场交替计时的中位结果；训练 CPU 来自固定单核、同一十轮流的三次配对复测。",["任务","全首批指标","p50 µs","p95 µs","训练 CPU 秒","工件 KiB"],rows,[n["resources"],n["recommendation"]])
    add("方法依据与复算入口","每项机制均在相同输入、父工件及评审规则下和 V6 配对。补充设备在各自传感器空间拟合，按设备、日期或重复分别审计。",["任务","研究问题","项目内的对照"],[
        ["A 气体","浓度及日期变化时，传感器共同幅值是否压过动力学形状","分开替换原型和距离，再合用；按整日期检查全年电子鼻"],
        ["B 轴承","按窗口拟合是否使长寿命训练设备过度影响先验","无条件替换、硬切换、渐进切换；留台架与 27 种窗口"],
        ["C 发动机","早段弱化个体寿命偏差是否改善末端，代价如何变化","固定寿命上限、变化斜率、首批成熟度约束；整机分折"],
        ["控制器","可行动作的表达及分组风险惩罚是否有独立收益","固定工业头做提示析因，再将排序惩罚迁移到新头"]
    ],["气体方法参考 [MDFE-Net](https://doi.org/10.1021/acssensors.6c00538) 与[跨时段复现代码](https://github.com/SuperAIfuture/E-nose)。轴承留整机对照参考 [Dual-Stream Vibration](https://github.com/Rifatsarkar3/Dual-Stream-Vibration)。",
       "发动机误差及非对称代价对照参考 [Adeel，2026 年 11 月预印本](https://arxiv.org/abs/2611.28759)。可复算的各候选、补充数据、来源与逐轮回执收录于公开实验证据包。"])
    return sections

def reports(blocks):
    md=["# 企业赛 V7 验证汇总",""]
    doc=pdf.open()
    for i,(title,intro,header,rows,notes) in enumerate(blocks,1):
        md+=["## "+title,"",intro,"","|"+"|".join(header)+"|","|"+"|".join(["---"]*len(header))+"|"]
        md+=["|"+"|".join(map(str,r))+"|" for r in rows]
        for note in notes:md+=["",note]
        md+=[""]
        def para(t):
            escaped=html.escape(t)
            linked=re.sub(r"\[([^\]]+)\]\((https://[^)]+)\)",r'<a href="\2">\1</a>',escaped)
            return "<p>"+linked+"</p>"
        def row(tag,vals):return "<tr>"+"".join(f"<{tag}>{html.escape(str(v))}</{tag}>"for v in vals)+"</tr>"
        body="<h1>"+html.escape(title)+"</h1>"+para(intro)+"<table>"+row("th",header)+"".join(row("td",r)for r in rows)+"</table>"+"".join(para(t)for t in notes)
        page=doc.new_page(width=842,height=595);page.draw_rect(pdf.Rect(0,0,842,9),color=None,fill=(.09,.30,.42))
        page.insert_text((36,40),"BJTU  /  INDUSTRIAL AI     |     ENTERPRISE V7",fontsize=11,color=(.1,.35,.46))
        free,scale=page.insert_htmlbox(pdf.Rect(36,56,806,546),body,css="*{font-family:sans-serif;color:#202c36}h1{font-size:27px;color:#173e52;margin:0 0 12px}p{font-size:15px;line-height:1.42;margin:9px 0}table{border-collapse:collapse;width:100%;margin:9px 0}td,th{text-align:left;font-size:14px;padding:6px 4px;border-bottom:1px solid #d1dfe2}th{background:#e5eef2;color:#173e52}")
        if free<0 or scale<.80:raise ValueError(title)
        page.draw_line((36,557),(806,557),color=(.77,.84,.87));page.insert_text((36,578),"McCarthy   /   PUBLIC VALIDATION",fontsize=9);page.insert_text((780,578),str(i),fontsize=9)
    md+=["## 复现与资料","","294 对十轮流，5,880 个逐轮记录。其中固定后确认 168 对。两版配对核对训练、选型及审计行索引的摘要。PyTRiO 真实模型流、完整首批整组留出、轴承窗口穷举和补充数据另计。","",
         "冻结源代码与输入摘要见 freeze.json，逐流复算见 audit.json、scorecard.json，真实 Qwen 调用与对照见 cloud/ 和 evidence.json。技术依据、失效条件及补充设备结果详见各分项报告；V5 官网回执摘录见 platform-reference.json。",""]
    (OUT/"企业赛V7验证汇总.md").write_text("\n".join(md))
    doc.set_metadata({"title":"企业赛 V7 冻结验证","author":"McCarthy"})
    doc.subset_fonts();dest=OUT/"企业赛V7技术报告.pdf";doc.save(dest,garbage=4,deflate=True);doc.close()
    with pdf.open(dest) as d:
        assert len(d)==len(blocks) and all(len(pg.get_text())>160 for pg in d)
        for pg in d:pg.get_pixmap(matrix=pdf.Matrix(.6,.6))

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=read("freeze.json")
    assert all(sha(HERE/n)==s for n,s in lock["code"].items())
    assert all(sha(HERE.parent/"V6"/n)==s for n,s in lock["baseline"].items())
    assert read("audit.json")["pairs"]==294
    assert all(f"{k} round {i}:" in (WORK/"image-proof.log").read_text() for k in "ABC" for i in (1,2,3))
    assert "Ran 24 tests" in (WORK/"tests.log").read_text() and (WORK/"tests.log").read_text().strip().endswith("OK")
    actual=sha(OUT/"harness.tar")
    assert read("artifact-audit.json")["archive_sha256"]==actual
    assert actual in (HERE/"模型配置.txt").read_text()
    reports(sections())
    (OUT/"模型配置.txt").write_bytes((HERE/"模型配置.txt").read_bytes())
    src=OUT/"企业赛V7源码.zip"; evidence=OUT/"公开实验证据.zip"
    with zipfile.ZipFile(src,"w",zipfile.ZIP_DEFLATED) as z:
        names=subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard","--","企业赛","校赛"],cwd=REPO)
        for name in sorted(set(names.split(b"\0"))):
            if not name:continue
            f=REPO/name.decode()
            current=f.is_relative_to(HERE)
            historical=f.suffix==".py" and f.is_relative_to(HERE.parent)
            provenance=f==REPO/"校赛/来源声明.md"
            include=current and (f.suffix in (".py",".txt",".md") or f.name in ("evaluation_protocol.json","cloud_protocol.json"))
            if f.is_file() and f.name!="AGENTS.md" and (include or historical or provenance):
                z.write(f,f.relative_to(REPO))
    with zipfile.ZipFile(evidence,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(WORK.rglob("*.json")):
            if {"delivery","serving-assets"}.isdisjoint(f.relative_to(WORK).parts):z.write(f,f.relative_to(WORK))
        for f in sorted(HERE.glob("*.md")):z.write(f,"说明/"+f.name)
        for f in (WORK/"image-proof.log",WORK/"tests.log"):z.write(f,f.name)
    for path in (src,evidence):
        with zipfile.ZipFile(path)as z:assert z.testzip() is None and path.stat().st_size<=50000000
    files=sorted(x for x in OUT.iterdir() if x.is_file() and x.name!="SHA256SUMS.txt")
    (OUT/"SHA256SUMS.txt").write_text("".join(sha(x)+"  "+x.name+chr(10) for x in files))
if __name__=="__main__":main()
