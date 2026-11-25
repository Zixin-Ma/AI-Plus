"""Render the technical report and a narrated-by-captions replay from actual records."""
from pathlib import Path
import hashlib
import html
import json
import os
import subprocess
import tempfile
import zipfile

import pymupdf as pdf

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUT = WORK / "delivery"
SWAN = "https://swanlab.cn/@Zixin-Ma/ai-plus-campus-rsi"
CSS = """* {font-family:sans-serif;color:#183733;} p {font-size:16px;line-height:1.5;margin:6px 0 12px}
h2 {font-size:25px;color:#0a655a;margin:6px 0 6px} .big {font-size:30px;color:#096a5e}
table {width:100%;border-collapse:collapse;margin:12px 0} td, th {text-align:left;font-size:15px;padding:9px 6px;border-bottom:1px solid #bfd1ca}
th {background-color:#dcebe3;color:#144d43} .note {font-size:13px;color:#47625c}
a {color:#116e5b} code {font-family:monospace;font-size:13px}"""


def record(series, key, index):
    return json.loads((WORK/series/key/f"round-{index:02}"/"record.json").read_text())


def page(doc, section, title, body):
    n=len(doc)+1
    pg=doc.new_page(width=760,height=520)
    pg.draw_rect(pg.rect,fill=(.976,.976,.954),color=None)
    pg.draw_rect(pdf.Rect(0,0,12,520),fill=(.055,.36,.31),color=None)
    pg.insert_text((40,36),section,fontsize=11,color=(.10,.35,.30),fontname="hebo")
    pg.insert_text((695,36),f"{n:02d}",fontsize=11,color=(.10,.35,.30))
    pg.draw_line((40,48),(719,48),color=(.70,.78,.72),width=.8)
    rest,scale=pg.insert_htmlbox(pdf.Rect(40,61,718,476),f'<div class="big">{title}</div>'+body,css=CSS)
    if rest<0 or scale<.88:
        raise RuntimeError(f"content exceeded page {n}: scale {scale}")
    pg.insert_text((40,499),"BJTU  /  AI+     |     CAMPUS COMPETITION     |     2026-11-25",
                   fontsize=9,color=(.30,.40,.36))


def main():
    OUT.mkdir(exist_ok=True)
    doc=pdf.open()
    page(doc,"01   /   SYSTEM DESIGN","让工业模型在漂移下持续更新","""
    <p>针对气体传感器、滚动轴承和涡扇发动机，系统在每轮从已部署的 CPU 模型续训，比较新候选，再由 Qwen3.8-27B 判断是否上线。</p>
    <table><tr><th>阶段</th><th>输入与产出</th></tr>
    <tr><td>监测与诊断</td><td>读取公开划分和父检查点，比较训练与验证特征均值，记录偏移最大的三个特征。</td></tr>
    <tr><td>优化与评估</td><td>恢复 Adam 状态，比较纯续训、保留既有校正、收缩、完整和稳健残差更新。</td></tr>
    <tr><td>部署与回退</td><td>提交候选供宿主评测。27B 读取候选、父模型及历史；指标退化时保留上一产物。</td></tr></table>
    <p class="note">SDK 1.0.4 · host_gateway_rpc · Linux/amd64 · 单轮一次模型调用 · 输出上限 160 token</p>""")
    page(doc,"02   /   INDUSTRIAL FEATURES","从当前观测提取可部署的特征","""
    <h2>A　气体分类</h2><p>对 16 个传感器的 8 类响应做有符号对数压缩，减去同次测量中同类响应的中位数，再比较归一化响应形状。验证集只有 5 条甲苯样本，因此以 Macro-F1 衡量六类的整体表现。</p>
    <h2>B　轴承寿命</h2><p>双通道振动快照提供 34 项统计量。用对数变换压缩长尾特征，比较普通残差与降低长寿命样本权重的稳健更新。训练和验证按整根轴承隔离，输出分钟。</p>
    <h2>C　发动机寿命</h2><p>用公开训练行中的运行设置划分工况，学习各工况传感器的中心与尺度。当前循环匹配最近的训练工况，用传感器偏差校正基础模型的剩余循环数。</p>
    <p class="note">拟合只使用训练划分。部署输入只包含当次传感器观测；特征路径取自父检查点。</p>""")
    rowhtml=""
    for key, label in zip("ABC",["气体 · Macro-F1 ↑","轴承 · RMSE（分钟）↓","发动机 · RMSE（循环）↓"]):
        fst=record("validation10",key,1)
        last=record("validation10",key,10)
        live=record("qwen-direction-fix" if key=="C" else "qwen3",key,1 if key=="C" else 3)
        rowhtml+=f'<tr><td>{label}</td><td>{fst["config"]["public_ablation"]["parent"]:.4f}</td><td>{last["selected_score"]:.4f}</td><td>{live["selected_score"]:.4f}</td></tr>'
    page(doc,"03   /   PUBLIC VALIDATION","公开验证集上，三项任务均有改善",f"""
    <table><tr><th>任务与指标</th><th>静态基线</th><th>十轮 CPU 对照</th><th>真实 27B 联调</th></tr>{rowhtml}</table>
    <p>CPU 对照使用验证分数自动选择，每项跑 10 轮。A、B、C 分别接受 5、2、4 次更新。十轮结束时，A 的 Macro-F1 提高 0.01175，B、C 的 RMSE 分别下降 4.62% 和 8.04%。</p>
    <p>真实 27B 列记录 A、B 各 3 轮，以及 C 修正提示后从静态基线重新进行的 1 轮。初版 C 的 3 轮均保留基线。开发数据固定且参与选型，正式成绩依赖赛方隐藏流。</p>
    <p class="note">公开验证量分别为 1,244、1,679、23,049 行。B 的 1,515 行来自同一根轴承，需要结合跨轴承表现解释聚合误差。</p>""")
    rejected=record("qwen3","B",3)
    fix=record("qwen-direction-fix","C",1)
    page(doc,"04   /   DECISIONS AND COST","一次正确回退，一次决策修正",f"""
    <h2>B / 第三轮保留父模型</h2><p>候选 RMSE 为 {rejected["config"]["public_ablation"][rejected["config"]["method"]]:.2f} 分钟，父模型为 {rejected["selected_score"]:.2f} 分钟。27B 选择保留，保护规则也拒绝了本轮更新。</p>
    <h2>C / 修正指标方向表达</h2><p>初版提示并列展示多个数值。27B 把 52.87 降至 48.78 判断为变差，连续拒绝。现将父分数、候选分数和正值代表改善的差值一起交给模型。独立复核的真实回复如下。</p>
    <p><i>{html.escape(fix["response"])}</i></p>
    <p class="note">开发期 10 个推理回执，共 3,176 输入与 410 输出 token。按未缓存公开价估算约 0.0204 元。逐轮假设、用量、候选对比和选择记录见 <a href="{SWAN}">SwanLab 公开项目</a>。</p>""")
    page(doc,"05   /   REPRODUCIBILITY","复现入口、验收与来源","""
    <p><b>运行。</b>prepare.py 读取赛方压缩包。rehearse.py control 可复现十轮 CPU 对照；live 模式用同一训练机制和真实 27B 联调。build_image.py 生成单镜像 Docker-save 归档。</p>
    <p><b>验收。</b>从镜像归档还原文件系统，用非特权进程运行 describe、train 和 predict。A/B/C 各两轮通过父摘要核验、分块上传、回执和逐行预测。SDK 替身回放与官方隐藏评测独立记录。</p>
    <p><b>研究依据。</b>气体跨批漂移参见 <a href="https://www.mdpi.com/2227-9709/13/1/15">Lin 与 Zhan (2026)</a>。跨工况轴承评估参见 <a href="https://www.nature.com/articles/s41598-026-65910-w">Scientific Reports (2026)</a>，发动机工况归一化参见 <a href="https://doi.org/10.31224/8145">2026 年预印本</a>。实验编排参考 <a href="https://github.com/WecoAI/aideml">AIDE</a> 的验证反馈设计。完整比较收录于源码目录《研究与实验》。</p>
    <p><b>使用声明。</b>CPU 基础模型和 Harness SDK 由赛方提供。参赛算法及集成经 AI 辅助开发。镜像保留 Python、Debian 和 NumPy 的分发声明。数据使用赛方公开投影。</p>""")
    rpt=OUT/"技术报告.pdf"
    doc.save(rpt,garbage=4,deflate=True)
    doc.close()
    # Include a sixth scene in the video with the actual exported-image verification transcript.
    with pdf.open(rpt) as doc:
        lines=(WORK/"image-validation.txt").read_text().splitlines()
        proof="".join(f'<p style="font-size:13px">{html.escape(l)}</p>' for l in lines if "streaming predict OK" in l)
        if proof.count("streaming predict OK") != 6:
            raise RuntimeError("missing verified image transcript")
        page(doc,"06   /   RECORDED SDK CHECK","镜像运行回放 · 三项任务各两轮",proof)
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            cmd=["ffmpeg","-loglevel","error","-y"]
            for i,pg in enumerate(doc):
                fn=f"{tmp}/{i}.png"
                pg.get_pixmap(matrix=pdf.Matrix(1.5,1.5),alpha=False).save(fn)
                cmd.extend(["-loop","1","-t","10","-framerate","4","-i",fn])
            cmd.extend(["-filter_complex","".join(f"[{i}:v]" for i in range(len(doc)))+
                        "concat=n=6:v=1:a=0,scale=1140:780,pad=1388:780:(ow-iw)/2:0:color=0xf9f9f3[v]",
                        "-map","[v]","-threads","2","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",
                        str(OUT/"流程演示.mp4")])
            subprocess.run(cmd,check=True)


if __name__=="__main__":
    main()
