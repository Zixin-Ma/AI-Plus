"""Compile the evidence for the submitted release and the subsequent trial separately."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile, ZIP_DEFLATED

import fitz

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent/"work/v8"
DEST=WORK/"delivery"
VERS=("v5","v7","v8")

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as f:
        h=hashlib.sha256()
        for part in iter(lambda:f.read(1048576),b""):
            h.update(part)
        return h.hexdigest()

def write_document(title,sections):
    pdf=fitz.open()
    markdown=["# "+title,""]
    for i,(heading,intro,cols,rows,notes) in enumerate(sections,1):
        page=pdf.new_page(width=842,height=595)
        page.draw_rect(fitz.Rect(0,0,842,10),color=None,fill=(.08,.29,.34))
        page.insert_text((34,39),"McCARTHY   /   INDUSTRIAL AI     |     ENTERPRISE V8",fontsize=11,color=(.13,.39,.43))
        def row(tag,values):
            return "<tr>"+"".join(f"<{tag}>{html.escape(str(value))}</{tag}>" for value in values)+"</tr>"
        paragraph=lambda s:"<p>"+html.escape(s)+"</p>"
        body="<h1>"+html.escape(heading)+"</h1>"+paragraph(intro)+"<table>"+row("th",cols)+"".join(row("td",r) for r in rows)+"</table>"+"".join(paragraph(n) for n in notes)
        css="*{color:#24333a;font-family:sans-serif}h1{font-size:29px;margin:0 0 12px;color:#123e4a}p{font-size:15px;line-height:1.4;margin:10px 0}table{border-collapse:collapse;width:100%}th,td{text-align:left;font-size:14px;padding:6px;border-bottom:1px solid #cadcdb}th{background:#deeeec;color:#08505e}"
        free,scale=page.insert_htmlbox(fitz.Rect(34,52,808,546),body,css=css)
        assert free>=0 and scale>=.82, (heading,free,scale)
        page.draw_line((34,555),(808,555),color=(.74,.81,.81))
        page.insert_text((35,578),"BJTU   /   McCarthy",fontsize=9)
        page.insert_text((746,578),f"{i}  /  {len(sections)}",fontsize=9)
        markdown += ["## "+heading,"",intro,"","|"+"|".join(cols)+"|","|"+"|".join(["---"]*len(cols))+"|"]
        markdown += ["|"+"|".join(map(str,r))+"|" for r in rows]
        for n in notes: markdown.extend(["",n])
        markdown.append("")
    (DEST/(title+".md")).write_text("\n".join(markdown))
    pdf.set_metadata({"title":title,"author":"McCarthy"})
    pdf.subset_fonts()
    target=DEST/(title+".pdf")
    pdf.save(target,garbage=4,deflate=True)
    pdf.close()
    with fitz.open(target) as check:
        assert len(check)==len(sections) and all(len(page.get_text())>150 for page in check)
        for page in check: page.get_pixmap(matrix=fitz.Matrix(.7,.7))
    assert target.stat().st_size<2_000_000

def pct(val):return f"{100*val:.2f}%"
def triple(values,fmt):return " / ".join(fmt(x) for x in values)

def local_sections():
    s=read(WORK/"paired-summary.json")
    audit=read(WORK/"audit.json")
    assert audit["observations"]==3600 and audit["pairs"]==120
    digest=read(WORK/"artifact-audit.json")["archive_sha256"]
    def stats(mode,task,field):return [s["confirm/"+mode][v][task][field] for v in VERS]
    pages=[]
    pages.append(("V8 如何改变三项工业任务","正式参照为企业赛 V5，最近一次试运行为 V7。V8 从 V5 的三个任务头重新做消融，复用 V7 的部署检查和 SDK 接口。",["任务","生产改动","冻结前公开对照"],
    [["A 气体","类内动力学聚类、组均值漂移惩罚和尾部温度","首批宏 F1 从 0.88905 升至 0.89471"],
     ["B 轴承","完整轨迹提供终点证据；低能量侧按设备校正","首批实际部署 RMSE 从 507.83 降至 460.38 分钟"],
     ["C 发动机","按轨迹覆盖决定参数平均，双侧 NASA 保护稳定轮校准","106 台公开验证发动机 RMSE 从 47.287 降至 46.920"]],
    ["A 按日期整体留出，末 8 日宏 F1 从 0.646 升至 0.660，较早窗口有回撤。B 留整台 XJTU 平均改善，FEMTO 和 KAIST 有小幅退步。C 的整机配对重采样支持 RMSE 改善，MAE 与 NASA 的区间仍跨零。",
     "有界候选先通过可见静态参照和历史工件检查，再由真实 Qwen3.8-27B 复核；赛场仅使用宿主当前释放的数据。"]))
    for mode,desc in (("anchored","固定官方检查点及完整公开训练池，按 V5 平台十轮发布时序改变可见验证范围。"),("schedule","按 V5 的轮次比例逐步释放公开训练数据，重建首批静态模型。"),("transfer","逐步释放培训数据，同时从选型集合排除当前审计阶段，考察跨域部署。"),("fresh-measured","阶段交替，每次提供不同的可见选型块，复核器在合格选项中选择可见主指标最佳者。")):
        rows=[]
        for task in ("A","B","C"):
            v=stats(mode,task,"stage_floor")
            rows.append([task,triple(stats(mode,task,"stage_mean"),pct),triple(v,pct),triple(stats(mode,task,"forget"),pct),
                         triple(stats(mode,task,"delay"),lambda x:f"{x:.2f}")])
        pages.append(("十轮对照 · "+{"anchored":"官方起点","schedule":"分批发布","transfer":"跨域选型","fresh-measured":"交替漂移"}[mode],desc+" 表中每格依次为 V5 / V7 / V8。",["任务","平均阶段收益","最低阶段收益","最大历史回撤","恢复延迟（轮）"],rows,
        ["每个值先按流计算，再对预留划分 201–208 取算术平均。收益相对该流静态模型，分类看宏 F1、回归看 RMSE。最低阶段收益指各阶段平均收益中的最小值。历史回撤与恢复延迟越低越好。",
         "这些划分共用一套公开数据。开发和冻结确认合计 360 条十轮流，逐轮核对静态分数与训练、选型、审计样本索引。平台隐藏阶段和官方评分权重由赛方另行计算。"]))
    real=[]
    for k in "ABC":
        result=[];tokens=[]
        for v in VERS:
            base=WORK/"cloud/streams"; name=f"anchored-{v}-{k}-201"
            result.append(read(base/(name+".json"))["stage_floor"])
            tokens.append(sum(r["tokens"] for r in read(base/("receipts-"+name+".json"))))
        real.append([k,triple(result,pct),triple(tokens,lambda x:f"{x:,}"),triple(stats("anchored",k,"prediction_us"),lambda x:f"{x:.1f}")])
    pages.append(("真实控制器与运行成本","固定同一预留划分，真实调用 PyTRiO Qwen3.8-27B，再走十轮训练和部署接口。每格依次为 V5 / V7 / V8。",["任务","最低阶段收益","十轮消耗 Token","公开单条预测 µs"],real,
    ["最右列单条预测为官方起点协议八个划分的 CPU 时间中位数。真实模型的回复决定每轮候选和部署，Token 取网关回执，完全相同的请求使用同一已记录回复。",
     "三组不同的真实首轮诊断中，简短提示均保留 refit 决定，单次节省 53–97 个 Token。参考预测路径在 30 个配对条件下与原计算逐值一致。归档包含镜像、参考检查点及代码指纹，使用者可复核选型。",
     "镜像 SHA-256 "+digest]))
    return pages

def platform():
    files=[WORK.parent/"v5/formal/formal-report.json",WORK.parent/"v7/platform-trial/report.json",WORK/"platform-trial/report.json"]
    results=[read(f) for f in files]
    runs=[x["run"] for x in results]
    assert runs[-1]["status"]=="completed" and runs[-1]["mode"]=="trial"
    reps=[x["report"] for x in results]
    sets=[{d["key"]:d for d in r["workspace_report"]["datasets"]} for r in reps]
    metrics={}
    lines=[]
    for k in "ABC":
        assert all(d[k]["record_complete"] and len(d[k]["rounds"])==10 for d in sets)
        for name in sets[0][k]["metrics"]:
            if not isinstance(sets[0][k]["metrics"][name],(int,float)):continue
            val=[d[k]["metrics"][name] for d in sets]
            metrics[k+"."+name]=val
            fmt=lambda x:f"{x:,.6f}" if abs(x)<10000 else f"{x:,.0f}"
            lines.append([k,name,*[fmt(x) for x in val],f"{100*(val[-1]/val[0]-1):+.3f}%"])
    output=dict(runs=[{f:r.get(f) for f in ("id","mode","status","score")} for r in runs],
                metrics=metrics,cpu_seconds=[sum(d[k]["cpu_seconds"] for k in "ABC") for d in sets],
                costs=[r["costs"] for r in reps],quota=read(WORK/"platform-trial/rules-latest.json").get("formal_usage"))
    (WORK/"platform-summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2))
    with (DEST/"平台隐藏集对照.md").open("w") as doc:
        doc.write("# V8 平台隐藏集对照\n\n|任务|指标|V5 正式|V7 试运行|V8 试运行|较 V5|\n|---|---|---:|---:|---:|---:|\n")
        for r in lines:doc.write("|"+"|".join(r)+"|\n")
        doc.write("\nV8 试运行记录 "+runs[-1]["id"]+"。本次三项均完成十轮。试运行报告公布最终隐藏指标，五项正式分和最低阶段收益仍需正式评测。\n")
    return output

def pack():
    path=DEST/"公开实验证据.zip"
    with ZipFile(path,"w",ZIP_DEFLATED) as out:
        for directory in ("gas","bearing","engine","replay","opening-history","cloud/streams"):
            for f in sorted((WORK/directory).glob("*.json")):
                out.write(f,f.relative_to(WORK))
        for f in ["protocol.json","audit.json","freeze.json","paired-summary.json","cloud/cold-plan-comparison.json",
                  "reference-prediction-study.json","artifact-audit.json","platform-summary.json","tests.log","image-check-0.log"]:
            src=WORK/f
            if src.is_file():out.write(src,f)
        for f in ROOT.glob("*.md"):out.write(f,"方法/"+f.name)
    with ZipFile(path) as z:assert z.testzip() is None
    assert path.stat().st_size<50_000_000
    names=[p for p in sorted(DEST.iterdir()) if p.is_file() and p.name!="SHA256SUMS.txt"]
    (DEST/"SHA256SUMS.txt").write_text("".join(sha(f)+"  "+f.name+"\n" for f in names))

def main():
    p=argparse.ArgumentParser();p.add_argument("step",choices=("technical","platform"));step=p.parse_args().step
    if step=="technical":
        assert (WORK/"tests.log").read_text().rstrip().endswith("OK")
        assert all(f"{k} round {i}:" in (WORK/"image-check-0.log").read_text() for k in "ABC" for i in (1,2,3))
        write_document("企业赛V8技术报告",local_sections())
        digest=read(WORK/"artifact-audit.json")["archive_sha256"]
        config=("企业赛 V8 / McCarthy\nQwen/Qwen3.8-27B，PyTRiO；temperature 0，规划 56 Token，复核 32 Token。\n"
                "镜像 bjtu-rsi:enterprise-v8；Linux/amd64；CPython 3.12；NumPy 2.2.6；Harness SDK 1.0.4。\n"
                "入口 python3 /app/harness.py ，支持 describe、train、predict。用户 65534:65534。\n"
                "模型由任务头、赛方静态检查点、历史工件和宿主 Qwen 网关共同构成。\n"
                "SHA-256 "+digest+"\n")
        (ROOT/"模型配置.txt").write_text(config)
        (DEST/"模型配置.txt").write_text(config)
    else: platform()
    pack()
if __name__=="__main__":
    main()
