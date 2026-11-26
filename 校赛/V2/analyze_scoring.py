"""Inspect the team's archived scoring responses without spending evaluation attempts."""
from fractions import Fraction
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1] / "work"
output = {}
for version in ("v1", "v2"):
    report = json.loads((WORK / "platform" / ("official-" + version + "-response.json")).read_text())
    objective = report["objective_result"]
    recovery = objective["objective_components"]["recovery"]
    ratio = Fraction(recovery / 5).limit_denominator(100)
    output[version] = {
        "displayed_run_score": report["run"]["score"],
        "objective_score": objective["objective_score"],
        "components": objective["objective_components"],
        "eligible_for_ranking": objective["eligible_for_ranking"],
        "gates": objective["gates"],
        "recovery_ratio_small_denominator": str(ratio),
        "recovery_reconstruction_error": float(ratio) * 5 - recovery,
        "raw_resource_breakdown_in_formal_report": report.get("report"),
        "all_round_records_complete": all(row["record_complete"] for row in report["training_records"]),
    }
rules_path = WORK / "platform/team-rules-response.json"
if rules_path.exists():
    rules = json.loads(rules_path.read_text())["rules"]
    output["configured_rule"] = {k: rules[k] for k in
        ("formula", "label_budget", "token_budget", "wall_seconds", "training_budget")}
    for version in ("v1", "v2"):
        response = json.loads((WORK / "platform" / ("official-" + version + "-response.json")).read_text())
        growth = {}
        for dataset in response["training_records"]:
            first = dataset["rounds"][0]["configuration"]
            last = dataset["rounds"][-1]["configuration"]
            growth[dataset["key"]] = {
                name: last[name] - first[name] for name in ("train_rows", "validation_rows")}
        output[version]["public_label_growth"] = growth
        output[version]["public_label_growth_total"] = sum(sum(v.values()) for v in growth.values())
output["limits"] = [
    "Two zero-valued resource components cannot identify a unique cost formula or threshold.",
    "The formal report omits the resource-cost breakdown and static hidden-stage metrics.",
    "Recovery's rational representation is a numerical clue and does not identify the event denominator.",
    "Run score and objective score are different fields. Expert component attribution is unavailable.",
    "Public label growth equals the configured label budget. Billing semantics and scoring dependence remain unverified.",
    "A false eligibility gate is separate from the resource component. Causation is unverified.",
]
path = WORK / "v2/scoring-analysis.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps(output, ensure_ascii=False, indent=2))
