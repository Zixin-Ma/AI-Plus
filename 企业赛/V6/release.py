"""Render the frozen V6 public comparison and package reproducible sources and evidence."""
import hashlib
import html
import json
import statistics as st
import subprocess
import zipfile
from pathlib import Path
import pymupdf as pdf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = HERE.parent / "work/v6"
OUT = WORK / "delivery"
CASES = {"drift":"训练池逐批增长","fresh":"回访更新选型样本","fresh-measured":"回访实测择优",
         "official":"完整官方起点","trace":"历史释放比例","schedule":"按官网阶段日程"}
NAMES={"A":"气体识别","B":"轴承寿命","C":"发动机寿命"}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b""):
            digest.update(block)
        return digest.hexdigest()

def table(header,rows):
    def cells(tag,items):
        return "<tr>"+"".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in items)+"</tr>"
    return "<table>"+cells("th",header)+"".join(cells("td",row) for row in rows)+"</table>"

def para(s):
    return "<p>"+html.escape(s)+"</p>"

def pair(x,y,scale=100,digits=2):
    return f"{x*scale:.{digits}f} → {y*scale:.{digits}f}"

def report():
    s=read(WORK/"paired-summary.json")
    audit=read(WORK/"audit.json")
    official=read(HERE.parent/"work/v5/formal/formal-report.json")["objective_result"]
    overview=read(WORK/"release-notes.json")
    doc=pdf.open()
    def page(title,body):
        p=doc.new_page(width=842,height=595)
        p.draw_rect(pdf.Rect(0,0,842,9),color=None,fill=(.06,.39,.37))
        p.insert_text((36,39),"BJTU  /  INDUSTRIAL AI     •     ENTERPRISE V6",fontsize=11,color=(.10,.36,.34))
        free,scale=p.insert_htmlbox(pdf.Rect(36,55,806,548),"<h1>"+html.escape(title)+"</h1>"+body,
                 css="*{font-family:sans-serif;color:#233139} h1{font-size:27px;color:#104942;margin:0 0 12px} p{font-size:15px;line-height:1.38;margin:8px 0} table{border-collapse:collapse;width:100%;margin:7px 0} th,td{text-align:left;font-size:14px;padding:6px 5px;border-bottom:1px solid #ccdbd6} th{background:#e0eee9;color:#174e46}")
        if scale<.8 or free<0: raise ValueError(f"page overflow {title}")
        p.draw_line((36,558),(806,558),color=(.76,.84,.82))
        p.insert_text((36,578),"McCarthy   /   PUBLIC VALIDATION",fontsize=9)
        p.insert_text((778,578),str(len(doc)),fontsize=9)
    page("V6 的起点与本地验证范围",para(f"V5 官网正式成绩 {official['objective_score']:.6f} / 35，最差阶段 0 / 5。V6 的试验以提升三项低谷、提高 A/B 平均收益、保住 C 平均收益为目标。")+
         table(["任务","V5 官网平均阶段收益","V5 官网最低阶段收益","对应阶段"],[[NAMES[k],f"{v['mean_gain']:.6f}",f"{v['worst_gain']:.6f}",v["stage_gains"].index(v["worst_gain"])+1]for k,v in official["datasets"].items()])+
         para(overview["decision"])+para(f"全矩阵核验 {audit['pairs']} 对十轮流、{audit['observations']} 个逐轮记录。六个预留编号在冻结后执行，共享官方公开样本。V6 仅在服务器本地评估。"))
    score=read(WORK/"scorecard.json")
    rows=[]
    for k in "ABC":
        v=score["schedule/"+k]
        a,b=v["stage_mean"],v["stage_floor"]
        sign=v["positive_floor"]
        rows.append([NAMES[k],pair(a["v5"],a["v6"]),pair(b["v5"],b["v6"]),f"{b['increased']} / {b['tied']} / {b['decreased']}",f"{sign[1]} / 6"])
    page("按官网日程回放的最低阶段",table(["任务","阶段平均收益 %","最低阶段收益 %","提高 / 持平 / 下降","V6 低谷为正"],rows)+
         para("每列先对同一顺序和同一任务的 V5、V6 配对，再汇总六个冻结后预留划分。平均列与最低列都对阶段等权。各划分共享公开数据。")+
         para(overview["schedule_interpretation"])+para("A 的公开审计仅含一个气体批次。B 每个载荷仅对应一台公开验证设备。C 按四个 FD 子集分组。数字相同的阶段编号不保证对应相同的隐藏工况。"))
    for k in "ABC":
        rows=[]
        for case,name in CASES.items():
            data=s["confirm/"+case]
            rows.append([name,*[pair(data["v5"][k][m],data["v6"][k][m]) for m in ("stage_mean","stage_floor","worst")]])
        page(NAMES[k]+"的预留回放",para(overview[k])+table(["场景","阶段等权平均 %","最低阶段 %","最低单轮 %"],rows)+
             para("每格从 V5 到 V6。先按每阶段被访问的轮次求均值，再求阶段等权平均及最小值。最后一列单独列出流中最低轮，六个顺序均等加权。"))
    rows=[]
    for case,name in CASES.items():
        data=s["confirm/"+case]
        rows.append([name,*[pair(data["v5"]["all"][m],data["v6"]["all"][m],1 if m=="delay" else 100,3 if m=="delay" else 2) for m in ("mean","worst","forget","delay")]])
    page("恢复速度、历史回撤与单轮收益",table(["协议","逐轮均值 %","最低轮 %","最大回撤 pp","恢复轮数"],rows)+para(overview["resources"]))
    rows=[]
    for k in "ABC":
        ds=[read(WORK/"opening"/f"{v}-{k}.json") for v in ("v5","v6")]
        rows.append([NAMES[k],len(ds[0]),pair(*[st.mean(z["gain"]for z in dd) for dd in ds]),pair(*[min(z["gain"]for z in dd) for dd in ds])])
    page("首轮对完整留出工况的迁移",table(["任务","审计组数","平均相对收益 %","最弱组相对收益 %"],rows)+
         para("两版从官方检查点和完整首批训练集起跑，每次拿出一组公开验证数据完全排除在选型之外。A 按类内采集顺序八等分，B 按三种载荷/转速，C 按四个 FD 源子集。分组仅供审计，预测端只接收输入特征。")+
         para(overview["opening_history"])+para("官网 A 有八个隐藏阶段，B 有三个，C 有四个。公开气体验证行仅来自同一批次，跨设备、跨日表现见分项补充数据实验；公开编号与隐藏工况没有已知一一对应。真实 V6 五项分数仍取决于平台评测。"))
    card=read(WORK/"scorecard.json")
    rows=[]
    for case,name in CASES.items():
        d=card[case+"/C"]["nasa"]
        rows.append([name,f"{d['v5']:,.1f}",f"{d['v6']:,.1f}",f"{100*(d['v6']/d['v5']-1):+.1f}%"])
    page("发动机的非对称误差成本",table(["确认协议","V5 每行 NASA","V6 每行 NASA","相对变化"],rows)+
         para("对每个样本按高估和低估分别使用 10 和 13 循环的指数尺度，先在当前阶段审计样本内取平均，再对十轮和六个划分等权。官网最终 NASA 使用隐藏测试样本求和。"))
    grid=read(WORK/"bearing-grid/summary.json")
    rows=[]
    for label,data in [("全部不同组合",grid["all"]),("此前未见组合",grid["new_masks"])]:
        lo=data["stage_floor"];avg=data["stage_mean"]
        rows.append([label,data["masks"],pair(avg["v5"],avg["v6"]),pair(lo["v5"],lo["v6"]),lo["increased"]])
    page("轴承低谷的全部窗口组合",table(["覆盖范围","组合数","阶段平均 %","阶段最低 %","低谷改善数"],rows)+
         para("六个预留编号中，有三个碰巧复用了开发期的审计窗口。本项在模型冻结后，枚举三台公开验证轴承按既定取窗规则能形成的全部 27 个组合。每个组合等权；其中 18 个在 0–19 号划分中从未出现。")+
         para("全 27 组中，最低阶段为正的组合从 4 个增至 10 个。三个工况的收益依次从 −4.14%、−9.42%、20.19% 变为 −0.47%、−3.13%、16.88%。总体低谷的改善集中在前两个公开工况，第三个工况有所回落。")+
         para("不同窗口共享同三台设备。这一穷举检验回答取窗方式会不会改变结论；外部整机证据与官网隐藏第三阶段的推断范围另见验证汇总。"))
    refs=read(WORK/"official-reference.json")
    rows=[]
    for name,src in [("V4 试运行",refs["trials"]["V4"]["tasks"]),("V5 试运行",refs["trials"]["V5"]["tasks"]),("V5 正式",refs["final_metrics"])]:
        rows.append([name,f"{src['A']['macro_f1']:.6f}",f"{src['B']['rmse']:.3f}",f"{src['C']['rmse']:.3f}",f"{src['C']['nasa_score']/1e8:.3f}"])
    page("官网记录与补充数据",table(["比赛记录","A Macro-F1","B RMSE 分钟","C RMSE 循环","C NASA 亿"],rows)+
         para(overview["external"])+para(overview["recommendation"]))
    doc.set_metadata({"title":"企业赛 V6 冻结本地验证","author":"McCarthy"})
    path=OUT/"企业赛V6技术报告.pdf"
    doc.subset_fonts();doc.save(path,garbage=4,deflate=True);doc.close()
    with pdf.open(path) as d:
        for p in d:
            assert len(p.get_text().strip())>150
            p.get_pixmap(matrix=pdf.Matrix(.7,.7),alpha=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=read(WORK/"freeze.json")
    assert all(sha(HERE/n)==h for n,h in frozen["code"].items())
    assert all(sha(HERE.parent/"V5"/n)==h for n,h in frozen["baseline"].items())
    assert read(WORK/"audit.json")["pairs"]==234
    log=(WORK/"image-proof-0.log").read_text()
    assert all(f"{k} round {n}:" in log for k in "ABC" for n in (1,2,3))
    report()
    # Include earlier project code used by ablation, data preparation and verification.
    source=OUT/"企业赛V6源码.zip"
    with zipfile.ZipFile(source,"w",zipfile.ZIP_DEFLATED) as z:
        listing=subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard","--","企业赛","校赛"],cwd=REPO)
        paths=set()
        for entry in listing.split(b"\x00"):
            if not entry:
                continue
            f=REPO/entry.decode("utf-8")
            if f.is_file() and f.name!="AGENTS.md" and (f.suffix in (".py",".md",".txt") or f.name=="Dockerfile"):
                paths.add(f)
        paths.add(HERE/"evaluation_protocol.json")
        for f in sorted(paths):
            z.write(f,f.relative_to(REPO))
    evidence=OUT/"公开实验证据.zip"
    with zipfile.ZipFile(evidence,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(WORK.rglob("*.json")):
            if any(t in f.relative_to(WORK).parts for t in ("delivery","serving-assets")):continue
            z.write(f,f.relative_to(WORK))
        for name in ("integrated-4.log","integrated-5.log","integrated-6.log","integrated-7.log"):
            z.write(WORK/name,"contracts/"+name)
        z.write(WORK/"image-proof-0.log","image-proof-0.log")
        for f in sorted(HERE.glob("*.md")):z.write(f,"说明/"+f.name)
    for path in [source,evidence]:
        with zipfile.ZipFile(path) as z:
            assert path.stat().st_size<=50_000_000 and z.testzip() is None
    (OUT/"模型配置.txt").write_bytes((HERE/"模型配置.txt").read_bytes())
    files=[OUT/"harness.tar",source,evidence,OUT/"企业赛V6技术报告.pdf",OUT/"模型配置.txt",OUT/"企业赛V6验证汇总.md"]
    (OUT/"SHA256SUMS.txt").write_text("".join(sha(f)+"  "+f.name+chr(10) for f in files))
if __name__=="__main__":
    main()
