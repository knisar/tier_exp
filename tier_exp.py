#!env python3

import argparse
import os
import subprocess
import random

import numpy as np

import wandb

workspace = "knisar"
project = "new_refit_202411"

"""
Tier 1: Team Group (sharing level)
Tier 2: Project (folder comparisons will be drawn from)
Tier 3: Variant of the experiment
Tier 4: Model
Tier 5: Component
Tier 6: Subcomponent of component model (Host Job/PID)
"""


def log_process_events(wb_logger_name):
    print(f"Running process for {wb_logger_name}")
    run_name, exp_name, model_name, component_name, subcomponent_name = wb_logger_name
    # run_name = wb_logger_name[0]
    # exp_name = wb_logger_name[1]
    # model_name = wb_logger_name[2]
    # component_name = wb_logger_name[3]
    # subcomponent_name = wb_logger_name[4]
    wandb.init(
        mode="online",
        project=project,
        entity=workspace,
        name=run_name,
    )
    wandb.config.update({"learning_rate": 0.01, "batch_size": 2 ** np.random.choice(12),
                        "exp": exp_name, "model": model_name, "component": int(component_name), "subcomponent": subcomponent_name})
    for epoch in range(10):
        for step in range(10):
            loss = random.uniform(0, 1)
            accuracy = random.uniform(0.5, 1)
            wandb.log({"epoch": epoch, "step": step, "loss": loss, "accuracy": accuracy})
    wandb.finish()


def get_runs(exp):
    nsubcomp = [18, 16, 16, 12, 16, 15, 12, 16, 19, 15, 11, 19, 10, 17, 15, 17, 18, 13, 17, 16]
    for model in [f"model_{l}" for l in list("abcdef")]:
        for ii, component in enumerate([str(c + 1) for c in range(20)]):
            for subcomponent in range(nsubcomp[ii]):
                run = f"{exp}_{model}_{component}_{subcomponent}"
                yield run, exp, model, component, subcomponent


def grid_experiment():
    """in practice this is fully parallel on multiple cpus/hosts"""
    for exp in ["base", "variant1", "variant2", "variant3"]:
        for wb_logger_name in get_runs(exp):
            subprocess.run(["python3", "tier_exp.py", "--model", str(wb_logger_name)])


def main():
    parser = argparse.ArgumentParser(prog="tier_exp", description="Tiered Experiment Test")
    parser.add_argument("-m", "--model", dest="logger_name", default=None)
    args = parser.parse_args()
    if args.logger_name is None:
        grid_experiment()
    else:
        log_process_events(eval(args.logger_name))


if __name__ == "__main__":
    main()
