"""Development ablation: fixed candidate search and post-experiment LLM only."""
import importlib
from contest_submission import MODULE, METRIC, eligible, gain, values, measures, continue_mlp

def experiment(key, index, model, previous, training, validation):
    x, y, _ = training
    xv, yv, groups = validation
    def assess(head):
        return measures(key, yv, values(key, model, head, xv), groups)
    before = assess(previous)
    old_steps, sampled = continue_mlp(key,index,model,x,y,"full")
    fresh = importlib.import_module(MODULE[key]).fit(model, x, y, index)
    candidates = {"refit": fresh, "adam": {}}
    if previous:
        candidates["continued"] = previous
    scores = {n: assess(h) for n, h in candidates.items()}
    def priority(name):
        s = scores[name]
        return eligible(key, before, s), gain(key, before["primary"], s["primary"])
    name = max(candidates, key=priority)
    return candidates[name], {"method": name, "parent": before, "trials": scores,
                              "old_steps": old_steps, "steps": model.meta["steps"],
                              "train_rows": len(y), "adam_rows": sampled, "validation_rows": len(yv)}


def message(key, index, info, score, history):
    parent = info["parent"]
    change = gain(key, parent["primary"], score["primary"])
    worst = gain(key, parent["worst"], score["worst"])
    previous = str(history[-1].get("reason", ""))[:65] if history else "first round"
    prompt = (f"{key}{index} {METRIC[key]} parent {parent['primary']:.5g}, {info['method']} {score['primary']:.5g}. "
              f"Signed gains (+ better): overall {change:.5g}, worst {worst:.5g}. Last: {previous}. "
              "Positive overall gain and stable worst group favor deployment. Reply DEPLOY or KEEP plus 2-5 words.")
    return [{"role": "user", "content": prompt}]


def choose(key, info, score, text):
    verdict = text.strip().split(maxsplit=1)
    first = verdict[0].strip(".,:*").upper() if verdict else ""
    return first == "DEPLOY" and eligible(key, info["parent"], score)
