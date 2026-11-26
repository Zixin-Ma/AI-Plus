"""Predefined public development contrasts for the deployment guard; keeps confirmation closed."""
import argparse
import benchmark
import contest_submission as submission


def configure(key, variant):
    if variant == "recall-rescue":
        submission.RECALL_RESCUE = True
    elif variant == "fixed-expert":
        submission.WEIGHTS[key] = (1.,)
        submission.OFFSET[key] = 0.
        submission.MEMORY[key] = float("inf")
    elif variant == "no-memory":
        submission.MEMORY[key] = float("inf")
    elif variant == "no-offset":
        submission.OFFSET[key] = 0.
    elif variant == "restrict-mix":
        submission.WEIGHTS[key] = (1., .85)
    elif variant == "wide-mix":
        submission.WEIGHTS[key] = (1., .65, .35, .15)
    elif variant == "adaptive-offset":
        submission.OFFSET[key] = .5
    elif variant == "trial-memory":
        submission.MEMORY[key] = .10
    else:
        raise ValueError(variant)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=list("ABC"))
    p.add_argument("variant",choices=("recall-rescue","fixed-expert","no-memory","no-offset","restrict-mix","wide-mix","adaptive-offset","trial-memory"))
    args=p.parse_args()
    configure(args.task,args.variant)
    benchmark.OUT=benchmark.OUT.parent/"ablation"/args.variant
    benchmark.run(args.task,"enterprise",0)
