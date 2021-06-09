import main
import random
import models
import datasets
import argparse
import getpass
import torch
from tqdm import tqdm
import pdb
import hashlib
import os


"""
python scripts/sweep.py --models IB_IRM --datasets Example3 --d 2 
python scripts/sweep.py --models IB_IRM --output_dir test_results/3envs --n_envs 3 --datasets Example1 --d 2
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--datasets', nargs='+', default=[])
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_data_seeds', type=int, default=50)
    parser.add_argument('--num_model_seeds', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="test_results")
    parser.add_argument('--callback', action='store_true')
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--jobs_cluster', type=int, default=512)

    parser.add_argument("--device", "--d", type=int)
    parser.add_argument("--clean", action="store_true")
    args = vars(parser.parse_args())

    try:
        import submitit
    except:
        args["cluster"] = False
        pass

    if torch.cuda.is_available() and args["device"] is not None:
        device = torch.device("cuda:{}".format(args["device"]))
    else:
        device = torch.device("cpu")

    all_jobs = []
    if len(args["models"]):
        model_lists = args["models"]
    else:
        model_lists = models.MODELS.keys()
    if len(args["datasets"]):
        dataset_lists = args["datasets"]
    else:
        dataset_lists = datasets.DATASETS.keys()

    for model in model_lists:
        for dataset in dataset_lists:
            for data_seed in range(args["num_data_seeds"]):
                for model_seed in range(args["num_model_seeds"]):
                    train_args = {
                        "model": model,
                        "num_iterations": args["num_iterations"],
                        "hparams": "random" if model_seed else "default",
                        "dataset": dataset,
                        "dim_inv": args["dim_inv"],
                        "dim_spu": args["dim_spu"],
                        "n_envs": args["n_envs"],
                        "num_samples": args["num_samples"],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "output_dir": args["output_dir"],
                        "callback": args["callback"],
                    }

                    all_jobs.append(train_args)

    random.shuffle(all_jobs)

    print("Launching {} jobs...".format(len(all_jobs)))

    if args["cluster"]:
        executor = submitit.SlurmExecutor(
            folder=f"/checkpoint/{getpass.getuser()}/submitit/")
        executor.update_parameters(
            time=3*24*60,
            gpus_per_node=0,
            array_parallelism=args["jobs_cluster"],
            cpus_per_task=1,
            comment="",
            partition="learnfair")

        executor.map_array(main.run_experiment, all_jobs)
    else:
        count = 0
        for job in tqdm(all_jobs):
            if args["clean"]:
                results_dirname = job["output_dir"]
                md5_fname = hashlib.md5(str(job).encode('utf-8')).hexdigest()
                results_fname = os.path.join(results_dirname, md5_fname + ".jsonl")
                if os.path.exists(results_fname):
                    os.remove(results_fname)
                    count += 1

            else:
                result = main.run_experiment(job, device)
                # print(result)

        if args["clean"]:
            print("REMOVE COUNT:", count)
