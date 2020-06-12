import os
import sys
import torch
import torch.utils.data
import numpy as np
from torchvision import datasets, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(sys.argv[1])

N_up = 1
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up])
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from net_wrappers import MF_BNN_cat
from train_BNN import train_VI_classification
from src.probability import variational_categorical, fixed_probs
from stochastic_resnet_models import arq_uncert_fc_resnet, arq_uncert_conv2d_resnet
from src.plots import evaluate_per_depth_classsification, plot_depth_distributions

# Get dataset

# dname = sys.argv[2]

for dname in ['MNIST', 'FashionMNIST', 'SVHN']:
    if dname == 'MNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=transform_train)
        valset = datasets.MNIST(root='../data', train=False, download=True,
                                transform=transform_test)

    elif dname == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        trainset = datasets.FashionMNIST(root='../data', train=True, download=True,
                                        transform=transform_train)
        valset = datasets.FashionMNIST(root='../data', train=False, download=True,
                                    transform=transform_test)

    elif dname == 'KMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1918,), std=(0.3483,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1918,), std=(0.3483,))
        ])

        trainset = datasets.KMNIST(root='../data', train=True, download=True,
                                transform=transform_train)
        valset = datasets.KMNIST(root='../data', train=False, download=True,
                                transform=transform_test)

    elif dname == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])

        trainset = datasets.SVHN(root='../data', split="train", download=True,
                                transform=transform_train)
        valset = datasets.SVHN(root='../data', split="test", download=True,
                            transform=transform_test)
    for _ in range(4):
        for d in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]:

            # Declare model

            cuda = torch.cuda.is_available()

            if dname in ["MNIST", "FashionMNIST", "KMNIST"]:
                input_chan = 1
            elif dname in ["SVHN"]:
                input_chan = 3
            output_dim = 10
            n_layers = d
            outer_width = 64
            inner_width = 32

            probs = np.zeros(n_layers + 1)
            probs[-1] = 1

            prob_model = fixed_probs(n_layers, probs=probs, distribution_name='cat',
                                    cuda=True)

            tags = {'direct_trained':True}

            model = arq_uncert_conv2d_resnet(input_chan, output_dim, outer_width,
                                            inner_width, n_layers, prob_model)

            N_train = len(trainset)
            lr = 1e-1
            schedule = [30]

            net = MF_BNN_cat(model, N_train, lr=lr, cuda=cuda, schedule=schedule)

            # Train model

            exp_name = '/deterministic_depth_scan/'
            name = 'CNN_BNN_' + dname + '_' + net.model.prob_model.name + exp_name + '%d' % d
            save_dir = '../saves/logs'

            print('cuda', torch.cuda.is_available())

            batch_size = 512
            nb_epochs = 500
            early_stop = 30
            nb_its_dev = 1
            save_freq = 10

            exp, mloglike_train, KL_train, ELBO_train, err_train, mloglike_dev, \
                err_dev = train_VI_classification(net, name, save_dir, batch_size, 
                nb_epochs, trainset, valset, cuda, flat_ims=False,
                nb_its_dev=nb_its_dev, early_stop=early_stop, load_path=None,
                tags=tags, stop_criteria='test_ELBO')
