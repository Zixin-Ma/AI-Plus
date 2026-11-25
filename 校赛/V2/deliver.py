"""Render and verify the V2 competition report, evidence replay and source bundle."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import hashlib
import html
import json
import statistics
import subprocess
import sys
import zipfile

import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from make_materials import page

V2 = ROOT / "V2"
WORK = ROOT / "work/v2"
REPO = "https://github.com/Zixin-Ma/AI-Plus/tree/main/%E6%A0%A1%E8%B5%9B/V2"
SWAN = "https://swanlab.cn/@Zixin-Ma/ai-plus-campus-rsi"


def read(rel):
    return json.loads((WORK / rel).read_text())


def table(headers, rows):
    def row(values, tag):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values) + "</tr>"
    return "<table>" + row(headers, "th") + "".join(row(r, "td") for r in rows) + "</table>"


def note(s):
    return '<p class="note">' + s + '</p>'


def link(url, title):
    return f'<a href="{html.escape(url,quote=True)}">{html.escape(title)}</a>'


def engine_and_bearing(doc):
    b = read("bearing/benchmark.json")
    def summary(route):
        p = b["public"][route]
        folds = [f[route]["pooled"]["rmse"] for f in b["folds"].values()]
        return [f"{p['pooled']['rmse']:.1f}", f"{p['pooled']['mae']:.1f}",
                f"{max(g['rmse'] for g in p['groups'].values()):.1f}", f"{statistics.mean(folds):.1f}"]
    page(doc,"04   /   BEARING LIFE","B　振动增强时，让传感器承担更大权重",
         "<p>公开验证的 1,679 条记录中，1,515 条来自同一根轴承。为避免用这根轴承主导选型，同时考察整根轴承隔离的九折结果与最差设备误差。</p>"+
         table(["方案","公开 RMSE","公开 MAE","最差公开轴承","九折等权 RMSE"],
               [[label,*summary(method)] for method,label in [("v1","V1"),("kernel_spectral_0.75","频谱邻域"),("deployed","V2 强度加权")]])+
         "<p>在继续训练的 MLP 上拟合振动特征残差。振动强度低于训练集第 75 百分位时，使用较强的训练寿命先验；到第 99 百分位时，预测完全依赖传感器与残差模型。部署所需统计量全部来自本轮训练轴承。</p>"+
         note("表内误差单位为分钟。九折中每次留出一整根轴承，重新训练基础 MLP。"))
    e = read("engine/confirm.json")
    def erow(key, name):
        row=e[key]
        return [name,*(f"{row[version][k]:.2f}" for k in ("rmse","worst_rmse") for version in ("v1","v2"))]
    page(doc,"05   /   ENGINE LIFE","C　条件归一化与有界寿命修正",
         "<p>按当前运行工况校正 21 路传感器。48 单元小网络读取传感器、工况与父 MLP 估计，学习寿命残差。单次修正最多 65 个循环，降低极端误差对 NASA 非对称评分的影响。</p>"+
         table(["评估集","V1 RMSE","V2 RMSE","V1 最差子集","V2 最差子集"],
               [erow(k,s) for k,s in [("public","公开 106 台"),("reserved_fold_3","留出整机折 3"),("reserved_fold_4","留出整机折 4"),("unseen_altitude_0","移除高度 0")]])+
         "<p>五个整机折的 RMSE 均下降。将占比最大的高度 0 工况完全移出训练后，V2 的误差上升。跨工况外推仍受训练覆盖范围制约。</p>"+
         note("两版本均用首轮候选对照。公开验证 MAE 从 35.16 降至 33.15 个循环；NASA 总分从 3.88×10⁹ 降至 1.43×10⁹。"))


def offline_report(doc):
    page(doc,"01   /   V2 DESIGN BRIEF","智训引擎 V2　漂移下的工业模型续训",
         "<p>面向六类气体识别、轴承剩余分钟数和发动机剩余循环数，智能体根据数据漂移选择续训工具，再用验证结果决定部署。</p>"+
         table(["V1 正式客观分","平均提升","最差阶段","恢复速度","稳定性","资源效率"],
         [["15.054075 / 35","5.978591 / 15","0.107231 / 5","3.968254 / 5","5 / 5","0 / 5"]])+
         "<h2>从最差阶段和资源开销入手</h2><p>V1 在三个任务上逐轮计算多个候选。V2 每轮执行一种训练工具，保留父模型的优化器状态，并对最差类别召回或最差设备误差设置部署条件。平均性能、阶段恢复和成本使用各自可复核的实验观测。</p>"+
         note("V1 平台试运行另有 14,773 条隐藏预测，三项各完成 10 轮，共 22,849 个 token。本报告的 V2 数值来自 BJTU 公开数据实验。"))
    page(doc,"02   /   AGENT AND DEPLOYMENT","两次 27B 决策各回答一个问题",
         table(["时机","Qwen3.8-27B 获得的证据","实际动作"],
               [["训练前","输入偏移、最差组、近期历史，以及训练内分组实验证据与计算成本","选择重拟合任务分支、沿用分支或纯 Adam，并决定全量或半量学习率。"],
                ["训练后","候选与父模型的主指标、当前最差组；必要时补测历史产物","选择新候选、保留父模型或恢复可用历史产物。每轮记录调用与选择依据。"]])+
         "<p>候选须改善主指标。A 的最差类别召回至多回落 0.01，B/C 的最差组 RMSE 至多上升 2%。当前候选未达条件时，从宿主已声明的历史产物中至多恢复一个，在同轮验证集上重新测量。B 另测关闭校正头后的父 MLP，帮助定位最差轴承的损失来源。</p>"+
         note("SDK 1.0.4 · CPU 单线程 · 每轮一次候选上传 · 单样本预测 · 训练工具白名单。历史产物的可用性由赛方宿主控制。"))
    g=read("gas/comparison.json")
    def grow(name,label):
        st = g["official"][name]
        return [label,*(f"{st[k]['macro_f1']:.4f}" for k in ("all","late")),f"{st['all']['min_class_f1']:.4f}"]
    page(doc,"03   /   GAS IDENTIFICATION","A　绝对响应与相对形状共同识别气体",
         "<p>保留 MLP 输出的同时，从传感器响应幅度和动力学构造两组特征。共享协方差的判别模型处理绝对强度，类别协方差经收缩后描述跨传感器的相对形状。</p>"+
         table(["公开批次 2","全批 Macro-F1","预留后 40%","最低类别 F1"],
               [grow(n,s) for n,s in [("v1_10","V1 十轮"),("absolute_lda","仅绝对特征 + MLP"),("relative_qda","仅相对特征 + MLP"),("combined","V2 双分支 + MLP")]])+
         "<p>445 条批次 1 记录用于拟合，批次 2 共 1,244 条。后 40% 按类别内采集顺序留出。乙醛 F1 从 0 升至 0.681，乙醇从 0.695 降至 0.598；甲苯只有 5 条。</p>"+
         note("批次 1 内按类别顺序分三块重训复验，V2 在两块改善，一块由 0.972 降至 0.937。"))
    engine_and_bearing(doc)
    rows=[]
    for key,label in zip("ABC",["A　Macro-F1 ↑","B　RMSE 分钟 ↓","C　RMSE 循环 ↓"]):
        fixed=[read(f"replay/{v}-{key}-fixed.json") for v in ("v1","v2")]
        rotating=[read(f"replay/{v}-{key}-rotating.json") for v in ("v1","v2")]
        rows.append([label,*(f"{r[-1]['audit']['primary']:.4f}" for r in fixed),
                     " → ".join(f"{100*min(p['relative_stage_gain'] for p in r):.1f}%" for r in rotating),
                     " → ".join(f"{sum(p['cpu_s'] for p in r):.2f}" for r in fixed)])
    page(doc,"06   /   OFFLINE ABLATION","同一公开投影下的十轮算法对照",
        table(["固定投影","V1 末轮","V2 末轮","轮换最低相对增益","固定十轮 CPU 秒"],rows)+
        "<p>此处用确定性规则从可选算法中择优，测量工业模型候选本身的能力。固定回放连续使用同一公开投影；轮换回放每组每五条留一条审计，剩余行按原顺序分四段，逐轮切换选择集。</p>"+
        "<p>最低相对增益为十轮中相对同段静态起点的最小改善率。训练数据在回放中固定。CPU 计时覆盖续训、候选选择和保存，另行计量审计集预测。</p>"+
        note("另对两版在相同公开首轮协议环境中各重复三次。CPU 中位秒数（V1 → 最终 V2）为 "+"、".join(f"{k} {r['v1']['cpu_seconds']:.3f} → {r['v2']['cpu_seconds']:.3f}" for k,r in read("paired-resources.json").items())+"；三项 V2 上传字节数均有上升。"))
def verified_agent():
    records = {}
    for key in "ABC":
        for mode, size in (("fixed",1),("rotating",10)):
            rows=read(f"agent/{key}-{mode}.json")
            if [r["round"] for r in rows] != list(range(1,size+1)):
                raise RuntimeError(f"Incomplete measured trial {key}/{mode}")
            for r in rows:
                if not r["planner_response"] or not r["decision_response"] or r["usage"]["completion_tokens"]<=0:
                    raise RuntimeError("Missing two-stage Qwen evidence")
            records[key,mode]=rows
    return records


def live_page(doc, records):
    if records is None:
        page(doc,"07   /   RENDER PREVIEW","双阶段 Qwen 实验 · 排版预览",
             "<p>此页在正式打包时读取完整双阶段运行回执。</p>")
        return
    result=[]
    for key in "ABC":
        series=records[key,"rotating"]
        old=read(f"replay/v1-{key}-rotating.json")[-1]["audit"]
        naive=read(f"agent-naive/{key}-rotating.json")[-1]
        result.append([key,f"{old['primary']:.4f}",f"{naive['primary']:.4f}",f"{series[-1]['primary']:.4f}",
                       f"{old['worst']:.3f} → {series[-1]['worst']:.3f}"])
    page(doc,"07   /   MEASURED QWEN DECISIONS","真实 Qwen 决策下的十轮公开漂移回放",
         table(["任务","V1","初版双阶段","最终双阶段","末轮最差组 V1 → 最终"],result)+
         "<p>V1 使用原算法的确定性选型回放，两版规划器使用真实 PyTRIO 开发回执，三者共用轮换方案和审计行。A 的主指标为 Macro-F1，最差组取类别召回最小值。B、C 使用总体 RMSE 和设备或子集 RMSE 最大值。表内均为第十轮已部署产物。</p>"+
         "<p>初版规划器在 C 的十轮中始终选择纯 Adam，末轮 RMSE 为 51.6687。最终规划器增加训练内分组实验结果与成本信息，工具选择仍由 Qwen 作出。中间版曾因回复附加多行解释，在 B 首轮把 RMSE 已降至 480.20 的候选退回到 681.16。</p>"+
         note("另从官方检查点在固定公开集做一次双阶段首轮验证，各项主指标为 "+"、".join(f"{k} {records[k,'fixed'][0]['primary']:.4f}" for k in "ABC")+"。"))


def decision_cases(doc, records):
    if records is None:
        page(doc,"08   /   DECISION TRACE","决策记录 · 排版预览","<p>正式版将从三个任务各选一条真实记录。</p>")
        return
    rows=[]
    for key in "ABC":
        series=records[key,"rotating"]
        # Pick a later decision where history is available whenever possible.
        event=next((r for r in series[1:] if r["deployed"]=="history"), None)
        if event is None:
            desired="parent" if key=="B" else "candidate"
            event=next((r for r in series[1:] if r["deployed"]==desired),series[-1])
        scores=event["record"]["configuration"]["scores"]
        p=event["plan"]
        rows.append([f"{key} / {event['round']}", f"{p['tool']} / {p['strength']}",
                     f"{scores['parent']['primary']:.4f} → {scores['candidate']['primary']:.4f}",
                     event["requested"],event["deployed"]])
    events=[r for (k,m),items in records.items() if m=="rotating" for r in items]
    inputs=sum(r["usage"]["prompt_tokens"] for r in events)
    outputs=sum(r["usage"]["completion_tokens"] for r in events)
    history_count=sum(r["deployed"]=="history" for r in events)
    adopted=sum(r["deployed"]=="candidate" for r in events)
    by_key="、".join(f"{k} {min(r['stage_relative_gain'] for r in records[k,'rotating']):.1%}" for k in "ABC")
    page(doc,"08   /   DECISION TRACE","三次实测决策与资源回执",
         table(["任务 / 轮","训练工具 / 步长","父模型 → 新候选","模型选择","实际部署"],rows)+
         f"<p>30 轮中部署新候选 {adopted} 次，保留父模型 {30-adopted-history_count} 次，恢复经复测的历史产物 {history_count} 次。60 个决策回执含输入 {inputs:,}、输出 {outputs:,} 个 token。开发侧同一轮次、参数和消息复用既有回执；赛方环境每轮发起两次实际调用。</p>"+
         f"<p>相对同一阶段静态起点，各任务十轮中最低的主指标改善率为 {by_key}。B 加入同轮关闭校正头的对照测量后，最低改善率从 −29.26% 变为 −2.68%，V1 为 −6.81%。</p>"+
         note("表内指标来自该轮可见验证，前页末轮指标来自独立审计子集。完整实验保存工具、假设、优化器步数、分组指标与模型回复。"))


def last_pages(doc):
    d=read("external/external.json")["scores"]
    tab=[[name, f"{d[k]['all']['macro_f1']:.4f}",f"{d[k]['blocks'][-1]['macro_f1']:.4f}"] for k,name in
         [("MLP","重新训练的 MLP"),("MLP_absolute","加绝对响应"),("MLP_relative","加相对形状"),("MLP_combined","二者联合")]]
    page(doc,"09   /   EXTERNAL TRANSFER","换用另一台电子鼻后，形状分支的收益减弱",
         "<p>Wörner 等公开了 62 通道电子鼻对三种物质的测量。作者预提取特征表含 36 个测量日、624 条记录。前 12 天的 215 条用于拟合，后 24 天的 409 条锁定测试，沿用赛题选定的两组判别参数。</p>"+
         table(["独立仪器迁移","24 天 Macro-F1","最后 8 天 Macro-F1"], tab)+
         "<p>绝对响应改善了最后阶段的识别；联合模型在该阶段下降。赛题双分支的选用依据来自官方仪器的跨批次审计。外部试验单独重训，样本和权重仅保存在服务器实验目录。</p>"+
         note(link("https://zenodo.org/records/15681119","原始数据及许可")+"　·　"+link("https://doi.org/10.1038/s41597-025-05993-8","Scientific Data，2025")))
    page(doc,"10   /   REPRODUCIBILITY","复现实验、运行环境与来源",
         "<p><b>复现顺序</b>　以校赛目录为工作路径。prepare.py 读取赛方训练包。V2 内的 bench_gas、bench_bearing、bench_engine_confirm 重做分组对照；bench_external 独立验证电子鼻迁移，replay.py 重做自由选择回放。完整命令与环境见内嵌源码 V2/README.md。</p>"+
         "<p><b>交付与验收</b>　build_image.py --version v2 构建 Linux/amd64 镜像，verify_image.py --version v2 以非特权用户运行 describe、train、predict，检查 A/B/C 连续两轮回执与逐条预测。平台作品文档内嵌源码和演示视频，独立文件的摘要见交付清单。</p>"+
         "<p><b>来源</b>　训练起点与 Harness SDK 1.0.4 由赛方提供；V2 算法和评测由队伍结合 AI 辅助开发。运行时使用 Python、Debian 和 NumPy 并保留发布声明。赛题内训练仅使用平台当轮公开训练投影。学术与开源依据包括 "+
         link("https://arxiv.org/abs/2512.22792","气体响应的判别几何")+"、"+
         link("https://biaowang.tech/xjtu-sy-bearing-datasets/","XJTU-SY 作者数据说明")+"、"+
         link("https://krokotsch.eu/rul-adapt/examples/conditional/","RUL Adapt 工况适配")+"及"+
         link("https://arxiv.org/abs/2502.13138","AIDE 验证反馈")+"。</p>"+
         note(link(REPO,"参赛源码")+"　·　"+link(SWAN,"SwanLab 公共实验项目")+"　·　"+link("https://rsi-index.115.zone/#rules","平台赛规")))


def video(report, out, records):
    with pdf.open(report) as doc, TemporaryDirectory(dir=out) as temp:
        folder=Path(temp)
        doc.select([0,1,2,3,4,5,6])
        if records:
            event=next((r for r in records["C","rotating"] if r["deployed"]=="history"),records["C","rotating"][0])
            p=event["plan"]
            page(doc,"08   /   QWEN TRACE",f"C 第 {event['round']} 轮 · 模型的实际选择",
                 "<p>本轮训练前回复</p><p>"+html.escape(event["planner_response"][:280])+"</p>"+
                 "<p>候选测量后的回复</p><p>"+html.escape(event["decision_response"][:200])+"</p>"+
                 table(["执行工具","学习率","请求部署","实际部署"],
                       [[p["tool"],p["strength"],event["requested"],event["deployed"]]])+
                 note("记录来自 BJTU 的真实 PyTRIO 调用与公开数据回放，协议验收使用导出的提交镜像。"))
        # One observation per scene. An eight-second still retains all source
        # numbers while H.264 inter frames avoid spending the upload on video.
        for i,pg in enumerate(doc):
            pg.get_pixmap(matrix=pdf.Matrix(1.5,1.5),alpha=False).save(folder/f"{i:02}.png")
        duration=len(doc)*8
        subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-framerate","1/8","-i",str(folder/"%02d.png"),
                        "-vf","fps=8,pad=1388:780:(ow-iw)/2:0:color=0xf9f9f3",
                        "-t",str(duration),"-c:v","libx264","-pix_fmt","yuv420p","-threads","2",
                        "-crf","29","-movflags","+faststart",str(out/"流程演示.mp4")],check=True)
        return duration


def sources(out):
    paths = [ROOT/f for f in ("Dockerfile",".dockerignore",".gitignore","prepare.py","build_image.py",
                               "verify_image.py","rehearse.py","publish_records.py","make_materials.py",
                               "README.md","来源声明.md","研究与实验.md")]
    paths += sorted((ROOT/"app").glob("*.py")) + sorted(V2.glob("*.py")) + sorted(V2.glob("*.md"))
    with zipfile.ZipFile(out/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path,path.relative_to(ROOT))
    return paths


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            h.update(chunk)
    return {"bytes":path.stat().st_size,"sha256":h.hexdigest()}


def render(out,records):
    out.mkdir(exist_ok=True,parents=True)
    rpt=out/"技术报告.pdf"
    with pdf.open() as doc:
        offline_report(doc)
        live_page(doc,records)
        decision_cases(doc,records)
        last_pages(doc)
        doc.subset_fonts()
        doc.set_metadata({"title":"智训引擎 V2 · 工业模型续训与漂移评估","author":"McCarthy"})
        doc.save(rpt,garbage=4,deflate=True)
    secs=video(rpt,out,records)
    paths=sources(out)
    attachments={"source.zip":out/"source.zip","workflow.mp4":out/"流程演示.mp4"}
    with pdf.open(rpt) as doc:
        for name,path in attachments.items():
            doc.embfile_add(name,path.read_bytes(),filename=name)
        doc.save(out/"作品文档.pdf",garbage=4,deflate=True)
    with pdf.open(out/"作品文档.pdf") as verified, pdf.open(rpt) as original:
        if len(verified)!=len(original) or not 1<=len(verified)<=10:
            raise ValueError("Report length mismatch")
        for pg,ref in zip(verified,original):
            if pg.get_pixmap().samples!=ref.get_pixmap().samples or len(pg.get_text())<120:
                raise RuntimeError("Report page failed pixel or text verification")
        for name,path in attachments.items():
            if verified.embfile_get(name)!=path.read_bytes():
                raise ValueError(f"Embedded file mismatch: {name}")
    if (out/"作品文档.pdf").stat().st_size>2*1024**2:
        raise ValueError("Single-document upload exceeds 2 MiB")
    with zipfile.ZipFile(out/"source.zip") as z:
        if z.testzip() or any(z.read(str(p.relative_to(ROOT)))!=p.read_bytes() for p in paths):
            raise ValueError("Source bundle changed")
    observed=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
                                            "-of","csv=p=0",str(out/"流程演示.mp4")]).decode().strip())
    if abs(secs-observed)>.2 or observed>300:
        raise ValueError(f"Video duration {observed} differs from expected {secs}")
    print(f"Verified {len(paths)} source files, 10 PDF pages and {observed:.1f}s video; {out}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preview",action="store_true",help="Exercise layouts without publishing a final bundle")
    args=p.parse_args()
    if args.preview:
        with TemporaryDirectory(dir=WORK) as tmp:
            render(Path(tmp),None)
        return
    out=WORK/"delivery"
    records=verified_agent()
    with TemporaryDirectory(dir=WORK) as tmp:
        folder=Path(tmp)
        render(folder,records)
        names=["作品文档.pdf","技术报告.pdf","流程演示.mp4","source.zip"]
        checks={name:digest(folder/name) for name in names}
        checks["harness.tar"]=digest(out/"harness.tar")
        (folder/"SHA256.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
        for name in names+["SHA256.json"]:
            (folder/name).replace(out/name)


if __name__=="__main__":
    main()
