"""Render the judge-facing report, evidence video and portable evidence/source bundles."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import html
import json
import statistics as stats
import subprocess
import zipfile

import pymupdf as pdf

HERE=Path(__file__).resolve().parent
WORK=HERE.parent/"work"
OUT=WORK/"delivery"
OUT.mkdir(exist_ok=True,parents=True)
CSS="""*{font-family:sans-serif;color:#172f42} p{font-size:15px;line-height:1.48;margin:8px 0 12px}
h1{font-size:28px;color:#143a54;margin:0 0 14px} h2{font-size:19px;color:#006e60;margin:12px 0 4px}
table{width:100%;border-collapse:collapse;margin:8px 0} th,td{padding:6px 7px;text-align:left;font-size:14px;border-bottom:1px solid #c7d7dc}
th{background:#dceeea;color:#145148} .small{font-size:12px;color:#486472} a{color:#00675f}"""
def read(path):
    return json.loads((WORK/path).read_text())
def table(cols,rows):
    def row(tag,vals):
        return "<tr>"+"".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in vals)+"</tr>"
    return "<table>"+row("th",cols)+"".join(row("td",r) for r in rows)+"</table>"
def page(doc,section,title,body):
    p=doc.new_page(width=800,height=540)
    p.draw_rect(p.rect,fill=(.98,.986,.987),color=None)
    p.draw_rect(pdf.Rect(0,0,10,540),fill=(.03,.40,.36),color=None)
    p.insert_text((34,31),"ENTERPRISE V1   /   "+section,fontname="hebo",fontsize=12,color=(.12,.33,.39))
    p.draw_line((34,44),(763,44),color=(.65,.78,.78))
    rest,scale=p.insert_htmlbox(pdf.Rect(34,59,764,500),"<h1>"+title+"</h1>"+body,css=CSS)
    if rest<0 or scale<.83:
        raise ValueError((len(doc),scale))
    p.insert_text((34,522),"BJTU   /   AI+      |      2026.11.26",fontsize=10,color=(.31,.43,.45))
    p.insert_text((734,522),f"{len(doc):02}",fontsize=11,color=(.12,.33,.39))
def replay(v,k,n,setting="drift"):
    return read(f"replay/{setting}-{v}-{k}-{n}.json")
def pair(v1,v2,places=2):
    return f"{v1:.{places}f} → {v2:.{places}f}"
def pct(a,b):
    return pair(100*a,100*b)
def metrics(v):
    runs=[replay(v,k,n) for k in "ABC" for n in range(3)]
    return {**{f:stats.mean(r[f] for r in runs) for f in ("mean","worst","forget","delay")},
            "min":min(r["worst"] for r in runs),"calls":sum(z["calls"] for r in runs for z in r["rounds"]),
            "bytes":sum(z["checkpoint_bytes"] for r in runs for z in r["rounds"])}
def resources(v,k):
    runs=[read(f"resources/{i}/drift-{v}-{k}-0.json") for i in range(3)]
    return stats.median(sum(z["cpu"] for z in r["rounds"]) for r in runs),stats.median(r["prediction_us"] for r in runs)
def live(k):
    d=read(f"live/{k}/round.json")["configuration"]
    return [k,{"A":"Macro-F1","B":"RMSE（分钟）","C":"RMSE（循环）"}[k],
            pair(d["scores"]["parent"]["primary"],d["scores"]["candidate"]["primary"],4),
            "采用候选" if d["deployed"]=="candidate" else "保留父模型"]
def paper():
    doc=pdf.open()
    a,b=metrics("campus"),metrics("enterprise")
    page(doc,"OBJECTIVE","漂移下的工业模型续训",'''<p>气体识别、轴承与发动机寿命预测共用一个问题。当前阶段更新带来的收益，能否在工况再次变化时保留下来？</p>
    <h2>校赛 V2 的正式起点</h2>'''+table(["正式客观总分","漂移流","最差阶段","恢复","稳定","资源"],[["16.769565 / 35","7.563216 / 15","0 / 5","4.206349 / 5","5 / 5","0 / 5"]])+'''
    <p>平台报告的 A、B、C 最差阶段分别为 −0.364、−0.471、−1.433。企业 V1 把受影响工况与静态起点逐组配对，再裁决对父模型的更新幅度。</p>
    <h2>公开回放结论</h2><p>三种任务、三种阶段安排，共九条十轮流。等权平均阶段收益由 4.47% 提至 8.77%，九条流中最低收益从 −47.32% 改善到 −22.36%。换序回放仍有退化，企业赛平台隐藏得分待评。</p>''')
    page(doc,"ARCHITECTURE","一次受约束的部署决策",table(["阶段","实际执行与可查证据"],[
        ["恢复与诊断","从宿主父检查点恢复 MLP、Adam 和归一化。A 比较类别召回，B 比较训练 RMS 四分位，C 比较由运行设置辨识的工况。"],
        ["训练与候选","27B 规划重拟合、保头续训或单步 Adam。A/B 在稳定投影复用计划。C 每轮重新规划。每轮向宿主提交一个候选。"],
        ["跨阶段保护","在同一当前公开投影测量父模型、静态起点及候选。回归任务用已释放训练行的寿命四分位检查历史能力损失。"],
        ["裁决与恢复","选择有限的静态混合强度或训练残差偏移。27B 读到有单位的收益及保护结果。候选不合格时可重测最近可见历史产物。"]
    ])+'''<p class="small">验证标签用于候选选择。单条预测只读取授权输入和产物。模型调用次数及产物 ID 随每轮记录保存。</p>
    <p class="small">方法借鉴面向分布变化的验证分割以及最差组风险比较。工程实现保留静态参考、寿命分层测量和宿主授权的回退路径。<a href="https://openreview.net/forum?id=Q692C0WtiD">TMLR 2025 验证分割</a> · <a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/929814ae1e76ae131a473e3f30609fef-Abstract-Conference.html">NeurIPS 2025 多专家鲁棒优化</a></p>''')
    page(doc,"VALIDATION","九条公开漂移流覆盖五项评分方向",table(["公开代理指标","校赛 V2","企业赛 V1","取值口径"],[
        ["漂移流平均收益","4.47%","8.77%","每条流先平均，再对九条流等权"],
        ["各流最差阶段均值","−13.24%","−4.91%","先取每条流的最低相对收益"],
        ["九条流中的最低值","−47.32%","−22.36%","所有当前阶段中的单一最低值"],
        ["恢复延迟（轮）","0.56","0.33","阶段切换后，负收益轮数的截尾均值"],
        ["历史最大回撤均值","13.68 点","12.94 点","同一历史阶段从最佳相对收益起算"],
        ["规划与裁决调用",str(a["calls"]),str(b["calls"]),"九条十轮流，离线协议应答，CPU 见后页"]
    ])+'''<p>训练池从每组的前 55% 增至 100%，两版从同一初始子集重建起点。四个阶段换序出现。每条流内，选型和审计的行互斥，发动机按整机分开。各条流复用同一批公开数据，不能视为九次独立采样。</p>''')
    rows=[]
    for n in range(3):
        for k in "ABC":
            a,b=replay("campus",k,n),replay("enterprise",k,n)
            rows.append([f"{n} / {k}",pct(a["mean"],b["mean"]),pct(a["worst"],b["worst"]),pct(a["forget"],b["forget"]),pair(a["delay"],b["delay"],1)])
    page(doc,"PAIRED RESULTS","逐流看收益与代价",table(["顺序 / 任务","平均收益 % ↑","最差收益 % ↑","回撤点数 ↓","延迟轮数 ↓"],rows)+
        '''<p class="small">每格为校赛 V2 → 企业赛 V1。顺序 0 用于开发；1 交换审计侧并校验，A 后续根据真实首轮调整了弱类保护；2 为修订后预先固定的末期先到压力测试。顺序 1 的 A、顺序 1 和 2 的 C 有最差阶段退化。</p>''')
    fixed=[]
    for k in "ABC":
        left,right=[replay(v,k,1,"official") for v in ("campus","enterprise")]
        fixed.append([k,pct(left["mean"],right["mean"]),pct(left["worst"],right["worst"]),
                      pct(left["forget"],right["forget"]),pair(left["delay"],right["delay"],1)])
    page(doc,"SENSITIVITY","换回官方静态起点，结论如何变化",table(
        ["任务","平均收益 %","最差收益 %","回撤点数","延迟轮数"],fixed)+
        '''<p>本组十轮使用同一官方检查点与完整训练池，轮换验证阶段。B 的平均、最差与恢复均改善，但历史最大回撤增加。A 的最差阶段和回撤变差，C 平均收益接近持平。</p>
        <h2>单独检查代价最大的保护条件</h2>
        <p>B 去掉训练寿命分层保护，开发流平均收益升至 8.54%，历史最大回撤从 27.19 扩大到 58.55 个百分点。去掉训练残差偏移，最差收益降至 −15.58%。C 放宽静态混合权重后，开发流平均与最差收益均下降。</p>
        <p class="small">相同完整公开首轮，校赛历史 27B 记录与企业最终 27B 结果为 A 0.8123 → 0.8171，B 480.20 → 644.09 分钟，C 46.835 → 46.879 循环。B 的单轮代价最大；企业 B 的任务头单独评估为 475.09 分钟，部署保护选择了更保守的混合权重。</p>''')
    page(doc,"EXPERTS","三个任务采用不同的漂移修正",table(["任务","采用路线及主要对照","整组留出或公开证据"],[
        ["A 气体","类内低秩子空间与核岭融合，保留 MLP 和气体相对响应。对照单核、单子空间、校赛双分支。","模块公开 Macro-F1 从 0.8123 到 0.8277。单子空间在此集为 0.8401，融合路线的十轮开发平均收益更好。"],
        ["B 轴承","两轴主频对转速与谱阶次给出软证据，在低振动区修正寿命先验。对照单步 Adam、校赛校正头及频率分箱。","九次整轴承留出的平均 RMSE 从 619.44 到 584.78 分钟，最差从 1109.36 到 1070.08。平均差配对区间跨零。"],
        ["C 发动机","按工况归一化并学习有界神经残差，用训练预测分位轻微扩张寿命两端。比较谱回归、轻量读出和多种混合权重。","两折分别留出 98、99 台发动机，最差 FD 的 RMSE 分别从 69.08 到 68.07、67.88 到 66.92。总体差值区间均跨零。"]
    ])+'''<p class="small">谱阶次先验与 XJTU 已知试验转速和载荷耦合。机型或工况变化时，需要重新检验这个假设。</p>''')
    kaist=read("bearing/kaist-shift.json")
    page(doc,"TRANSFER","六项补充数据如何参与检验",table(["补充数据","独立检验与结果"],[
        ["一年漂移电子鼻","前 12 天训练、后 24 天测试，跨日 Macro-F1 由 0.4960 升至 0.6067。按天配对重采样的差值 95% 区间 [0.0621, 0.1633]。"],
        ["DS-PS","按独立实验重复留出 72 次非零浓度暴露，Macro-F1 从 0.5311 降至 0.4980。中高浓度出现负迁移。"],
        ["FEMTO / PRONOSTIA","六根轴承逐设备重训，平均 RMSE 为 90.356 → 90.363 分钟。XJTU 的收益在这套设备上接近零。"],
        ["KAIST","129 个记录各读前 4096 行双通道加速度。用早期振幅分位作阈值，最低区间占比从 16/64 到 30/65，说明信号工况占比变化。此项只检验漂移覆盖。"],
        ["N-CMAPSS + CruiseBench","在 DS02 本机传感器上另行训练，按两台留出发动机评估。校赛、企业 RMSE 为 8.73、8.76；临近失效为 5.19、5.00。巡航掩码来自同源派生集。"]
    ])+'''<p class="small">外部设备各自使用其特征定义和数据切分。归档、原始样本和外部实验权重保存在开发服务器。</p>''')
    rt=[]
    for k in "ABC":
        a,b=resources("campus",k),resources("enterprise",k)
        rt.append([k,pair(a[0],b[0]),pair(a[1],b[1],1)])
    tokens=sum(read(f"live/{k}/{name}.json")["usage"]["total_tokens"] for k in "ABC" for name in ("call-1","call-2","review-v2"))
    page(doc,"DEPLOYMENT","真实裁决采用了哪些候选",table(["任务","指标","官方起点 → 实际采用","27B 裁决"],[live(k) for k in "ABC"])+
        f'''<p>完整公开首轮，服务器执行生产训练和保存产物后的复算。第一次评审三项均保留父模型。明确收益方向和保护结果后，每项复用原规划再作一次真实裁决。共九份真实调用回执，计 {tokens:,} Token。</p>'''+
        table(["任务","十轮 CPU 秒，校赛 → 企业","单条预测微秒，校赛 → 企业"],rt)+
        '''<p class="small">同机、单线程、两版交替执行三次取中位。回放减少了模型调用，部分任务的 CPU 和预测延迟增加。线上资源分还需宿主结算。</p>''')
    page(doc,"OPERATIONS","发布验收与现场使用边界",'''<h2>镜像与上线</h2><p>Linux/amd64、SDK 1.0.4、NumPy 2.2.6。启动前核对清单摘要，以普通用户运行。已在镜像导出的文件系统内完成三任务各两轮协议测试，检查父产物恢复、上传确认、回执引用和逐条预测。</p>
    <h2>异常与历史恢复</h2><p>模型输出经过工具与产物白名单解析。无效答复保留父模型，非有限输入或异常检查点交由宿主报错。七项闭环测试覆盖差分数阻止部署、未知历史不可选择、有效回滚以及优化器继续推进。</p>
    <h2>工业应用</h2><p>气体报警需重点监控相互混淆的乙醇与乙醛。轴承维护需核查新设备转速、载荷与谱阶次。发动机应分别复核健康期与临近失效的误差，并结合非对称高估成本。由检修团队确认实际检修动作。</p>
    <p class="small">交付内含源码、逐轮实验依据、带字幕的运行回放和镜像摘要。校赛 V2 的正式 16.769565/35 为历史对照，企业赛 V1 尚无平台隐藏成绩或专家评分。</p>''')
    doc.subset_fonts()
    doc.save(OUT/"企业赛V1技术报告.pdf",garbage=4,deflate=True)
    return doc


def video():
    d=pdf.open()
    page(d,"DEMONSTRATION","一次真实首轮如何完成",'''<p>开发服务器使用赛方公开起点和验证行。以下三个任务实际调用 PyTRIO Qwen3.8-27B，数字与裁决来自保存的产物和回执。</p><h2>训练 → 候选上传模拟 → 验证复算 → 大模型裁决</h2><p>末段展示的十轮轨迹使用离线受控裁决，呈现模型在漂移下的变化。</p>''')
    for k in "ABC":
        r=read(f"live/{k}/round.json")["configuration"]
        p,c=r["scores"]["parent"],r["scores"]["candidate"]
        text=html.escape(read(f"live/{k}/review-v2.json")["text"])
        page(d,"PUBLIC ROUND / "+k,k+" 任务的规划、组内测量和裁决",table(["训练动作","公开训练行","公开验证行","优化器步数"],[[r["plan"]["tool"],r["train_rows"],r["validation_rows"],f'{r["old_steps"]} → {r["steps"]}']])+table(["产物","主指标","最弱组指标","最差相对静态组增益"],[[label,f'{a["primary"]:.4f}',f'{a["worst"]:.4f}',f'{a["floor"]:.4f}'] for label,a in [('父模型',p),('候选',c)]])+f"<h2>真实裁决</h2><p>{text}</p>")
    for k in "ABC":
        a,b=replay("campus",k,2),replay("enterprise",k,2)
        rows=[[f'{r["round"]}',f'{r["stage"]}',pct(q["stage_gain"],r["stage_gain"]),r["deployed"]] for q,r in zip(a["rounds"],b["rounds"])]
        page(d,"DRIFT REPLAY / "+k,k+" 末期先到的十轮公开回放",table(["轮","阶段","校赛 → 企业，相对静态增益 %","采用"],rows))
    path=OUT/"企业赛V1运行回放.mp4"
    with TemporaryDirectory(dir=WORK) as t:
        for i,p in enumerate(d):
            p.get_pixmap(matrix=pdf.Matrix(1.6,1.6)).save(str(Path(t)/f"{i:02}.png"))
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","1/9","-i",str(Path(t)/"%02d.png"),
                        "-vf","fps=12,format=yuv420p","-c:v","libx264","-threads","2","-movflags","+faststart",str(path)],check=True)


def bundle():
    with zipfile.ZipFile(OUT/"企业赛V1源码.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.suffix in (".py",".md",".txt"):
                z.write(p,"企业赛/V1/"+p.name)
    with zipfile.ZipFile(OUT/"公开实验证据.zip","w",zipfile.ZIP_DEFLATED) as z:
        for pattern in ("replay/drift-*.json","replay/official-*.json","resources/*/*.json","live/*/*.json",
                        "ablation/*/*.json","gas-study/*.json","bearing/*.json","engine/confirmation.json",
                        "engine/internal.json","engine/external.json","image-smoke.log","loop-tests.log"):
            for p in sorted(WORK.glob(pattern)):
                z.write(p,p.relative_to(WORK))
    (OUT/"模型配置.txt").write_bytes((HERE/"模型配置.txt").read_bytes())
    paths=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    (OUT/"SHA256SUMS.txt").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths))


if __name__=="__main__":
    paper().close()
    video()
    bundle()
