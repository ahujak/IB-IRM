import torch
from torchvision import datasets


def get_mnist():
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('/home/zhangdh/data/mnist', train=True, download=True)
    '''
    mnist has no attribute 'data'
    mnist.train_data (60000, 28, 28)
    mnist.train_labels (60000,)
    '''
    # mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    # mnist_val = (mnist.data[50000:], mnist.targets[50000:])
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

