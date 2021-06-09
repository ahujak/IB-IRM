# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import hashlib
import pprint
import json
# import git
import os
import datasets
import models
import pdb
import utils


def run_experiment(args, device):
    # build directory name
    # commit = git.Repo(search_parent_directories=True).head.object.hexsha[:10]
    # results_dirname = os.path.join(args["output_dir"], commit + "/")
    results_dirname = args["output_dir"]
    os.makedirs(results_dirname, exist_ok=True)

    # build file name
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    results_fname = os.path.join(results_dirname, md5_fname + ".jsonl")
    # results_file = open(results_fname, "w")

    utils.set_seed(args["data_seed"])
    dataset = datasets.DATASETS[args["dataset"]](
        dim_inv=args["dim_inv"],
        dim_spu=args["dim_spu"],
        n_envs=args["n_envs"]
    )

    # Oracle trained on test mode (scrambled)
    train_split = "train" if args["model"] != "Oracle" else "test"

    # sample the envs
    envs = {}
    for key_split, split in zip(("train", "validation", "test", "test_peak20", "test_peak100", "test_peak500"),
                                (train_split, train_split, "test", "test", "test", "test")):
        envs[key_split] = {"keys": [], "envs": []}

        if key_split == "test_peak20":
            num_samples = 20
        elif key_split == "test_peak100":
            num_samples = 100
        elif key_split == "test_peak500":
            num_samples = 500
        else:
            num_samples = args["num_samples"]

        for env in dataset.envs:
            envs[key_split]["envs"].append(dataset.sample(
                n=num_samples,
                env=env, split=split)
            )
            envs[key_split]["keys"].append(env)

    # offsetting model seed to avoid overlap with data_seed
    utils.set_seed(args["model_seed"] + 1000)

    # selecting model
    args["num_dim"] = args["dim_inv"] + args["dim_spu"]
    model = models.MODELS[args["model"]](
        in_features=args["num_dim"],
        out_features=1,
        task=dataset.task,
        hparams=args["hparams"],
        device=device
    )

    # update this field for printing purposes
    args["hparams"] = model.hparams

    # fit the dataset
    model.fit(
        envs=envs,
        num_iterations=args["num_iterations"],
        callback=args["callback"])

    # compute the train, validation and test errors
    for split in ("train", "validation", "test", "test_peak20", "test_peak100", "test_peak500"):
        key = "error_" + split
        for k_env, env in zip(envs[split]["keys"], envs[split]["envs"]):
            args[key + "_" +
                 k_env] = utils.compute_error(model, *env)

    # write results
    results_file = open(results_fname, "w")
    results_file.write(json.dumps(args))
    results_file.close()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--model', type=str, default="ERM")
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--dataset', type=str, default="Example1")
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--callback', action='store_true')
    args = parser.parse_args()

    pprint.pprint(run_experiment(vars(args)))
