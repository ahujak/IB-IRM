import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import pdb

import numpy as np
import torch
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

'''

python -m domainbed.scripts.sweep_train --steps 2001 --freq 100 --debug --type 0 --ratio 0.9 --lr 1e-1 --bs 128 --xyl --freq 2000 --hf 0.2 --test_val --d 0
'''


def train(args, hparams, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hparams["data_type"] = args.type
    hparams["ratio"] = args.ratio
    hparams["env_seed"] = args.env_seed
    hparams["no_aug"] = args.no_aug
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # train/val split
        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    dataset.N_WORKERS = 0
    train_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights, batch_size=hparams['batch_size'],
                                        num_workers=dataset.N_WORKERS) for i, (env, env_weights) in enumerate(in_splits)
                     if i not in args.test_envs]
    eval_loaders = [FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
                    for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)
    algorithm.to(device)

    last_results_keys = None
    best_val_acc = 0.
    best_val_train_acc = 0.
    best_test_acc = 0.
    best_iter = 0
    best_oracle_val_acc = 0.
    best_oracle_test_acc = 0.
    best_oracle_train_acc = 0.
    best_oracle_iter = 0

    checkpoint_vals = collections.defaultdict(lambda: [])
    acc_result_ls = []
    for step in range(0, n_steps):
        step_start_time = time.time()

        if "FullColoredMNIST" != args.dataset:
            minibatches_device = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
        else:
            minibatches_device = [(x.to(device), y.to(device)) for x, y, color_l in next(train_minibatches_iterator)]

        step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % checkpoint_freq == 0 or step == n_steps - 1:
            results = {'step': step, 'epoch': step / steps_per_epoch}
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            val_acc = []
            train_acc = []
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device, with_color_label=("FullColoredMNIST" == args.dataset))
                results[name+'_acc'] = acc
                # for eval pick
                if name in ["env{}_out".format(i) for i, (env, _) in enumerate(in_splits) if i not in args.test_envs]:
                    val_acc.append(acc)
                if name in ["env{}_in".format(i) for i, (env, _) in enumerate(in_splits) if i not in args.test_envs]:
                    train_acc.append(acc)
            if args.test_val:
                val_acc = [results["env{}_out_acc".format(i)] for i in args.test_envs]
                val_acc = sum(val_acc) / len(val_acc)
            else:
                val_acc = sum(val_acc) / len(val_acc)
            train_acc = sum(train_acc) / len(train_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_train_acc = train_acc
                best_iter = step
                best_test_acc = [results["env{}_in_acc".format(i)] for i in args.test_envs]
                best_test_acc = sum(best_test_acc) / len(best_test_acc)
            results['val_acc'] = val_acc
            results['train_acc'] = train_acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=7)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=7)

            acc_result_ls.append(results)

    return best_test_acc, best_val_acc


def run(args, hparams, device):
    final_test_accs = []
    final_val_accs = []
    for seed in range(args.n_restarts):
        args.seed = args.trial_seed = seed
        print("RRRRRRR args.seed:{:3}     args.trial_seed:{:3}".format(args.seed, args.trial_seed))
        test_acc, val_acc = train(args, hparams, device)
        final_test_accs.append(test_acc)
        final_val_accs.append(val_acc)

    return (100 * np.mean(final_test_accs), 100 * np.std(final_test_accs),
            100 * np.mean(final_val_accs), 100 * np.std(final_val_accs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', default='~/data/', type=str)
    parser.add_argument('--dataset', type=str, default="FullColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=2001,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', "--freq", type=int, default=2000,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', "--hf", type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')

    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_step', type=int, default=0)
    parser.add_argument('--vrex_lambda', type=float)
    parser.add_argument('--vrex_step', type=int, default=0)
    parser.add_argument('--ib_lambda', type=float)
    parser.add_argument('--ib_step', type=int, default=0)
    parser.add_argument('--class_condition', action="store_true")
    parser.add_argument('--normalize', action="store_true", help="normalize the loss of irm / vrex")
    parser.add_argument('--dro_eta', type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--sch_size', default=-1, type=int)

    parser.add_argument('--d', default=0, type=int)
    parser.add_argument('--debug', action='store_true', help='if debugging, then save on somewhere else')
    parser.add_argument('--type', default=0, type=int, help="type of data, 0 for varying bernouli coef"
                                            "1 for varying digit-color correlation seed")
    parser.add_argument('--ratio', type=float, default=0.9, help="the ratio of the data which is bias-aligned "
                                        "if 1.0, digit & color has 1 to 1 corresponding")
    parser.add_argument('--env_seed', type=int, default=1, help="decide the correlation relation between color & digit")
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--xyl', default=1, type=int)

    parser.add_argument("--n_restarts", "--nr", default=5, type=int)
    parser.add_argument("--test_val", action="store_true")
    args = parser.parse_args()

    algorithm_dict = None
    if args.debug:
        args.output_dir = 'debug_output'
    print("Args.output_dir is: ", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Args:", args)
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        hparams["lr"] = args.lr if args.lr is not None else hparams["lr"]
        hparams['weight_decay'] = args.wd if args.wd is not None else hparams['weight_decay']
        hparams["irm_lambda"] = args.irm_lambda if args.irm_lambda is not None else hparams["irm_lambda"]
        hparams["irm_penalty_anneal_iters"] = args.irm_step if args.irm_step is not None else hparams["irm_penalty_anneal_iters"]
        hparams["ib_lambda"] = args.ib_lambda if args.ib_lambda is not None else hparams["ib_lambda"]
        hparams["ib_penalty_anneal_iters"] = args.ib_step if args.ib_step is not None else hparams["ib_penalty_anneal_iters"]
        hparams["class_condition"] = args.class_condition

        hparams["normalize"] = args.normalize

        hparams["xylopt"] = args.xyl
        hparams["xylnn"] = args.xyl
        hparams['batch_size'] = args.bs if args.bs is not None else hparams["batch_size"]
        hparams["dataset"] = args.dataset

        if args.sch_size > 0:
            hparams["sch_size"] = args.sch_size
        else:
            hparams["sch_size"] = 600

    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))

    print("Hparams:", hparams)

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.d))
    else:
        device = "cpu"

    print("RRRRRRR Hyper sweep for ERM")
    args.algorithm = "ERM"
    test_acc, test_std, val_acc, val_std = run(args, hparams, device)
    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
        val_acc, val_std, test_acc, test_std))

    print("RRRRRRR Hyper sweep for IB-ERM")
    report_val_acc = 0
    report_hyper = {}
    args.algorithm = "IBERM"
    hparams['ib_penalty_anneal_iters'] = 0
    for ib_lambda in [0.1, 1, 10, 1e2, 1e3, 1e4]:
        for times in [1, 1.2, 1.4, 1.6, 1.8]:
            hparams["ib_lambda"] = ib_lambda * times
            hparams["class_condition"] = False
            test_acc, test_std, val_acc, val_std = run(args, hparams, device)
            print("RRRRRRR ib_lambda:{:9.2f} "
                  "val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                ib_lambda * times, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"ib_lambda": ib_lambda * times}

    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
        report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)


    print("RRRRRRR Hyper sweep for IRM")
    report_val_acc = 0
    report_hyper = {}
    args.algorithm = "IRM"
    hparams["normalize"] = True
    hparams['irm_penalty_anneal_iters'] = 0
    # for irm_lambda in [0.1, 0.5, 1, 5, 10, 1e2]:
    for irm_lambda in [0.1, 1, 10, 1e2, 1e3, 1e4]:
        for times in [1, 1.2, 1.4, 1.6, 1.8]:
            hparams["irm_lambda"] = irm_lambda * times
            test_acc, test_std, val_acc, val_std = run(args, hparams, device)
            print("RRRRRRR irm_lambda:{:9.2f}    "
                  "val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                irm_lambda * times, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"irm_lambda": irm_lambda * times}
    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
        report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)


    print("RRRRRRR Hyper sweep for IB-IRM")
    report_val_acc = 0
    report_hyper = {}
    args.algorithm = "IBIRM"
    hparams['ib_penalty_anneal_iters'] = 0
    hparams["normalize"] = True
    hparams['irm_penalty_anneal_iters'] = 0
    for ib_lambda in [0, 0.1, 0.5, 1, 10, 1e2]:  # 没有5
    # for ib_lambda in [0, 0.1,]:
        hparams["ib_lambda"] = ib_lambda
        hparams["class_condition"] = False
        # for irm_lambda in [0.1, 0.5, 1, 5, 10, 1e2, 1e3, 1e4]:
        for irm_lambda in [0, 0.1, 0.5, 1, 10, 1e2]:
        # for irm_lambda in [0, 0.1,]:
            hparams["irm_lambda"] = irm_lambda
            test_acc, test_std, val_acc, val_std = run(args, hparams, device)
            print("RRRRRRR ib_lambda:{:9.2f}   irm_lambda:{:9.2f} "
                  "val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                ib_lambda, irm_lambda, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"ib_lambda": ib_lambda, "irm_lambda": irm_lambda}

    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
        report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)