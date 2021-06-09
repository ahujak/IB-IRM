import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import pdb
from time import time

from config import args, add_path

add_path('../../../utils')
from utils import mkdir, pretty_print
from architectures.network import MLP, cmnist_cnn

TEST_VALID = args.test_valid

def get_mnist():
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('~/data/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    return mnist_train, mnist_val


def make_environment(images, labels, e, device, flip_label=0.25):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]

    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    # labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    labels = torch_xor(labels, torch_bernoulli(flip_label, len(labels)))

    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    # images: (25000, 2, 14, 14)  dtype=torch.unit8
    # labels: (25000, )
    return {
        'images': (images.float() / 255.).to(device),
        'labels': labels[:, None].to(device)  # --> (25000, 1)
    }


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
    scale = torch.tensor(1.).to(y.device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def train(n_steps, envs, model, optimizer, device, l2_weight, p_weight, penalty_anneal_iters, args, method='erm',
          start_step=0, freq=50):
    for step in range(start_step, start_step + n_steps):
        test_acc_ls = []
        val_acc_ls = []
        for idx, env in enumerate(envs):
            # full batch!
            if idx == 3:
                logits = model(env['images'])
                env['acc'] = mean_accuracy(logits, env['labels'])
            elif idx == 2:
                if TEST_VALID:
                    val_size = int(env['images'].shape[0] * args.val)
                    val_logits = model(env['images'][:val_size])
                    val_acc_ls.append(mean_accuracy(val_logits, env['labels'][:val_size]))
                else:
                    val_size = 0

                logits = model(env['images'][val_size:])
                env['acc'] = mean_accuracy(logits, env['labels'][val_size:])
                test_acc_ls.append(env['acc'])

            else:
                if TEST_VALID:
                    val_size = 0
                    if args.inter != 0:
                        logits, inter_logits = model(env['images'], inter=args.inter)
                    else:
                        logits = model(env['images'])
                else:
                    val_size = int(env['images'].shape[0] * args.val)
                    val_logits = model(env['images'][:val_size])
                    cur_val_acc = mean_accuracy(val_logits, env['labels'][:val_size])
                    val_acc_ls.append(cur_val_acc)

                    if args.inter != 0:
                        logits, inter_logits = model(env['images'][val_size:], inter=args.inter)
                    else:
                        logits = model(env['images'][val_size:])

                env['nll'] = mean_nll(logits, env['labels'][val_size:])
                env['acc'] = mean_accuracy(logits, env['labels'][val_size:])
                env['penalty'] = penalty(logits, env['labels'][val_size:])

                if args.ib_lambda > 0.:
                    if args.class_condition:
                        num_classes = 2
                        index = [env['labels'].squeeze() == i for i in range(num_classes)]
                        env['var'] = sum(inter_logits[ind].var(dim=0).mean() for ind in index)
                        env['var'] /= num_classes
                    else:
                        env['var'] = inter_logits.var(dim=0).mean()

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += l2_weight * weight_norm

        # "info bottleneck"
        if args.ib_lambda > 0.:
            ib_weight = args.ib_lambda if step >= args.ib_step else 0.
            var_loss = torch.stack([envs[0]['var'], envs[1]['var']]).mean()
            loss += ib_weight * var_loss

        if method == 'irm':
            train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
            penalty_weight = (p_weight if step >= penalty_anneal_iters else .0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= penalty_weight
        elif method == 'erm':
            penalty_weight = 0.
            train_penalty = envs[0]['penalty'] * 0.  # so that this term is a tensor
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = sum(test_acc_ls) / len(test_acc_ls)
        val_acc = sum(val_acc_ls) / len(val_acc_ls)
    return train_acc, val_acc, test_acc


def run(args, mnist_train, device):
    final_train_accs = []
    final_val_accs = []
    final_test_accs = []

    for restart in range(args.n_restarts):
        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        # Build environments
        prob_flip = args.prob_flip
        envs = [
            make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2, device, flip_label=prob_flip),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, device, flip_label=prob_flip),
            make_environment(mnist_val[0], mnist_val[1], 0.9, device, flip_label=prob_flip),
            make_environment(mnist_val[0], mnist_val[1], 0.1, device, flip_label=prob_flip)
        ]

        # Define and instantiate the model
        if args.nn == 'cnn':
            mlp = cmnist_cnn().to(device)
        elif args.nn == 'mlp':
            mlp = MLP(args).to(device)
        else:
            raise NotImplementedError

        optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

        train_acc, val_acc, test_acc = train(args.steps, envs, mlp, optimizer, device, args.l2_regularizer_weight,
            args.penalty_weight, args.penalty_anneal_iters, args, method=args.method)

        final_train_accs.append(100 * train_acc.detach().cpu().numpy())
        final_val_accs.append(100 * val_acc.detach().cpu().numpy())
        final_test_accs.append(100 * test_acc.detach().cpu().numpy())

    return (np.mean(final_train_accs), np.std(final_train_accs),
          np.mean(final_val_accs), np.std(final_val_accs),
          np.mean(final_test_accs), np.std(final_test_accs))


if __name__ == "__main__":
    print('args:', args)

    mkdir("./checkpoint")
    device = torch.device('cuda:{}'.format(args.d))
    mnist_train, mnist_val = get_mnist()

    print("for ERM")
    args.method = "erm"
    args.ib_lambda = 0
    train_acc, train_std, val_acc, val_std, test_acc, test_std = run(args, mnist_train, device)
    print("train acc: {:.2f} +- {:.2f} val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                      train_acc, train_std, val_acc, val_std, test_acc, test_std))

    print("Hyper sweep for IB-ERM")
    report_val_acc = 0
    report_hyper = {}
    args.method = "erm"
    args.inter = 4  # fix the use of intermediate feature
    for ib_lambda in [1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
        for times in [1, 1.2, 1.4, 1.6, 1.8]:
            args.ib_lambda = ib_lambda * times
            args.cc = cc = False

            train_acc, train_std, val_acc, val_std, test_acc, test_std = run(args, mnist_train, device)
            print("ib_lambda:{:8.2f}     cc:{:5}    "
                  "train acc: {:.2f} +- {:.2f} val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                ib_lambda * times, cc,
                train_acc, train_std, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"ib_lambda": ib_lambda * times}

    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
            report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)


    print("Hyper sweep for IRM")
    # print("Hyper sweep for REx")
    report_val_acc = 0
    report_hyper = {}
    args.method = "irm"
    for penalty_weight in [1e-1, 1, 1e1, 1e2, 1e3, 1e4,]:
        for times in [1, 1.2, 1.4, 1.6, 1.8]:
            args.penalty_weight = penalty_weight * times

            train_acc, train_std, val_acc, val_std, test_acc, test_std = run(args, mnist_train, device)
            print("penalty_weight:{:9.2f}    "
                  "train acc: {:.2f} +- {:.2f} val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                penalty_weight * times,
                train_acc, train_std, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"penalty_weight": penalty_weight * times}

    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
            report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)


    print("Hyper sweep for IB-IRM")
    report_val_acc = 0
    report_hyper = {}
    args.method = "irm"
    args.inter = 4  # fix the use of intermediate feature
    for ib_lambda in [0, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
        args.ib_lambda = ib_lambda
        args.cc = cc = False
        for penalty_weight in [0, 1e2, 1e3, 1e4, 1e5, 1e6]:
            args.penalty_weight = penalty_weight

            train_acc, train_std, val_acc, val_std, test_acc, test_std = run(args, mnist_train, device)
            print("ib_lambda:{:8.2f}     cc:{:5}    p_weight:{:9.2f}    "
                  "train acc: {:.2f} +- {:.2f} val acc: {:.2f} +- {:.2f} test acc: {:.2f} +- {:.2f}".format(
                  ib_lambda, cc, penalty_weight,
                  train_acc, train_std, val_acc, val_std, test_acc, test_std))

            if val_acc > report_val_acc:
                report_val_acc = val_acc
                report_val_std = val_std
                report_test_acc = test_acc
                report_test_std = test_std
                report_hyper = {"ib_lambda": ib_lambda, "penalty_weight": penalty_weight}

    print("Report with best val performance: val acc:{:.2f} +- {:.2f}  test acc:{:.2f} +- {:.2f}".format(
        report_val_acc, report_val_std, report_test_acc, report_test_std))
    print("Corresponding hypers:", report_hyper)
