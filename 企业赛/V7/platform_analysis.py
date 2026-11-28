"""Compare the archived V7 trial receipt to the V5 trial and formal receipts."""
import json
from pathlib import Path
import re
import zipfile

import qualify

OUT = qualify.OUT / "platform-trial"
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    current = read(OUT / "report.json")
    assert current["run"]["mode"] == "trial" and current["run"]["status"] == "completed"
    baseline = read(qualify.OUT.parent / "v5/formal/formal-report.json")
    trial = read(qualify.OUT / "platform-reference.json")["reports"]["db6e3be38df7fb840f64e7a9faab8b1a"]
    results = {}
    rows = []
    notes = []
    for old, new in zip(baseline["report"]["workspace_report"]["datasets"],
                        current["report"]["workspace_report"]["datasets"], strict=True):
        assert old["key"] == new["key"] and len(new["rounds"]) == 10
        key = new["key"]
        assert new["metrics"] == current["report"]["workspace_report"]["datasets"]["ABC".index(key)]["metrics"]
        # V5 trial and formal exported identical final hidden-set metrics.
        assert old["metrics"] == trial["metrics"][key]
        metrics = {}
        for name, value in new["metrics"].items():
            before = old["metrics"][name]
            delta = value - before
            metrics[name] = dict(v5=before, v7=value, delta=delta, relative_percent=100*delta/before)
            label = name + ("，分钟" if key == "B" else "，周期" if key == "C" and name != "nasa_score" else "")
            rows.append([key, label, f"{before:,.6f}", f"{value:,.6f}", f"{delta:+,.6f}", f"{100*delta/before:+.2f}%"])
        rounds = []
        for old_row, row in zip(old["rounds"], new["rounds"], strict=True):
            cfg = row["configuration"]
            selected = row["selected_artifact_id"]
            match = re.search(r"-round-(\d+)-", selected)
            rounds.append(dict(round=row["round_index"], stage=row["stage_id"],
                               deployed_from_round=int(match.group(1)) if match else None,
                               tool=cfg.get("method"), head=cfg.get("head_family"),
                               validation=row["validation_metrics"], scores=cfg.get("scores"),
                               deployed=cfg.get("deployed"), plan_fallback=cfg.get("plan",{}).get("fallback")))
        results[key] = dict(metrics=metrics, cpu_v5=old["cpu_seconds"], cpu_v7=new["cpu_seconds"],
                            predictions=new["prediction_count"], rounds=rounds)
        notes.append(f"{key} 的完整轮数为 10/10，隐藏预测 {new['prediction_count']:,} 条。各轮部署来源为 " +
                     "、".join(str(r["deployed_from_round"]) for r in rounds) + "。")
    costs = [r["report"]["costs"] for r in (baseline, current)]
    cpus = [sum(r["cpu_"+v] for r in results.values()) for v in ("v5","v7")]
    summary = dict(run=current["run"], baseline=baseline["run"]["id"], tasks=results,
                   cpu_seconds=cpus, costs=costs, objective=current.get("objective_result"),
                   rules=read(OUT / "rules-latest.json"))
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    text = ["# 企业赛 V7 平台试运行对比", "",
            f"试运行 {current['run']['id']} 已完成，平台提交版本号为 5，对应项目企业赛 V7。比较基准为 V5 最近一次正式运行 {baseline['run']['id']}；V5 试运行的最终隐藏指标与该正式结果逐项相同。", "",
            "本次报告返回三任务的最终隐藏集指标、十轮可见验证记录和资源账单。run.score 为 null，响应中没有 objective_result，故五项客观分与逐阶段隐藏收益目前无法核实。", "",
            "|任务|指标|V5|V7|差值|相对变化|", "|---|---|---:|---:|---:|---:|"]
    text += ["|"+"|".join(row)+"|" for row in rows]
    text += ["", "## 资源与运行", "",
             f"三任务 CPU 合计 {cpus[0]:,.3f} → {cpus[1]:,.3f} 秒，相对变化 {100*(cpus[1]/cpus[0]-1):+.2f}%。",
             f"接口记账 Token {costs[0]['total_tokens']:,} → {costs[1]['total_tokens']:,}，相对变化 {100*(costs[1]['total_tokens']/costs[0]['total_tokens']-1):+.2f}%。",
             "平台 costs.training_seconds 是另一口径，不能与三任务 cpu_seconds 合计直接互换。", ""]
    text += [f"- {key} CPU {r['cpu_v5']:.3f} → {r['cpu_v7']:.3f} 秒。" for key,r in results.items()]
    text += ["", "## 部署行为与判断", "", *notes, "",
             "C 在第 1 轮部署后，其余九轮都保留该工件。最后一轮候选在可见验证上的 RMSE 为 47.346，父工件为 47.163，候选 allowed=false。当前证据表明更新未能通过部署约束；头参数改进没有在后续轮次转化成新的部署。",
             "B 最终部署第 9 轮工件。其可见验证 RMSE 从首轮 519.004 降到末轮 515.238，同时隐藏集 RMSE 比 V5 增加约 25.94%。可见验证的缓慢改善与隐藏集退化并存，应优先复查分设备迁移和早期部署的先验。",
             "建议保留现有正式额度，先修复 B 的隐藏误差回退，并解决 C 的有效更新和代价问题后再评估新版本。A 的隐藏 F1 略降，也应作为跨日策略的约束。", "",
             "## 客观评分与额度", "",
             "V5 正式五项为漂移流 9.079221/15、最差阶段 0/5、恢复 3.968254/5、稳定性与抗遗忘 4.986910/5、资源效率 5/5。V7 试运行尚未给出同口径五项分数。最终隐藏集总体误差无法还原每个阶段的收益和恢复评分。",
             f"本次消耗 1 次试运行。平台当前试运行剩余 {summary['rules']['trial_usage']['remaining']} 次，正式评测剩余 {summary['rules']['formal_usage']['remaining']} 次。",
             "", "## 平台入口", "", "https://rsi-index.115.zone/", "",
             "原始响应与各轮候选记录保存在同目录 report.json；comparison.json 提供本报告表格的原始精度数值。"]
    report = OUT / "企业赛V7平台试运行对比.md"
    report.write_text("\n".join(text))
    with zipfile.ZipFile(OUT / "V7平台试运行证据.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.json", "comparison.json", "run.json", "run-status.json", "rules-latest.json",
                     "trial-request.json", "image-status.json", "validation.json", report.name):
            archive.write(OUT / name, name)
    print(json.dumps(dict(cpu_seconds=cpus, tokens=[c["total_tokens"] for c in costs]), indent=2))
    print(report)


if __name__ == "__main__":
    main()
