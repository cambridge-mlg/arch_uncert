from __future__ import division, print_function
import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(sys.argv[1])

N_up = 1
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up])
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from src.utils import Datafeed
from src.datasets import make_spirals
from net_wrappers import MF_BNN_cat
from src.probability import variational_categorical
from train_BNN import train_VI_classification
from stochastic_resnet_models import arq_uncert_fc_resnet
from src.plots import evaluate_per_depth_classsification, plot_depth_distributions, plot_predictive_2d_classification, plot_layer_contributions_2d_classification, plot_calibration_curve



# Gen dataset

dname = 'spirals'
if dname == 'spirals':
    X, y = make_spirals(n_samples=2000, shuffle=True, noise=0.2, random_state=1234,\
                                         n_arms=2, start_angle=0, stop_angle=720)
else:
    X, y = datasets.make_moons(n_samples=2000, shuffle=True, noise=0.2, random_state=1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=1234)

x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)

X_train = ((X_train - x_means) / x_stds).astype(np.float32)
X_test = ((X_test - x_means) / x_stds).astype(np.float32)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

trainset = Datafeed(X_train, y_train, transform=None)
valset = Datafeed(X_test, y_test, transform=None)

# Declare model

for d in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]:

    cuda = torch.cuda.is_available()

    input_dim = 2
    width = 20
    n_layers = d
    output_dim = 2

    prior_probs = 0.85 ** (1 + np.arange(n_layers + 1))
    prior_probs = prior_probs / prior_probs.sum()

    # prior_probs = np.ones(n_layers + 1)/(n_layers + 1)

    prob_model = variational_categorical(n_layers, prior_probs, temp=0.1, eps=1e-10, cuda=cuda)

    # prob_model = fixed_probs(n_layers, probs=[1/(n_layers+1)], distribution_name='cat', cuda=True)

    tags = {'direct_trained': True}

    model = arq_uncert_fc_resnet(input_dim=input_dim, output_dim=output_dim,
                                 width=width, n_layers=n_layers, prob_model=prob_model)
    N_train = len(trainset)
    lr = 1e-1
    net = MF_BNN_cat(model, N_train, lr=lr, cuda=cuda, schedule=None)

# Train Models

    exp_name = '/depth_scan/'
    name = 'fc_BNN_' + dname + '_' + net.model.prob_model.name + exp_name + '%d' % d
    save_dir = '../saves/'

    print('cuda', torch.cuda.is_available())

    batch_size = 512  #
    nb_epochs = 8000  # We can do less iterations as this method has faster convergence
    early_stop = 500
    nb_its_dev = 20

    exp, mloglike_train, KL_train, ELBO_train, err_train, mloglike_dev, err_dev = \
        train_VI_classification(net, name, save_dir, batch_size, nb_epochs, trainset, valset, cuda,
                                flat_ims=False, nb_its_dev=nb_its_dev, early_stop=early_stop,
                                load_path=None, tags=tags, stop_criteria='train_ELBO')

    subfolders = [f.path for f in os.scandir(save_dir + '/' + name) if f.is_dir()]
    subfolders.sort(key=lambda x: os.path.getmtime(x))
    filename = subfolders[-1] + '/models/theta_best.dat'
    net.load(filename)

    prob_out = net.sample_predict(X_test, grad=False).data
    pred_out = prob_out.mean(dim=0).max(dim=1)[1].cpu()
    err = pred_out.ne(torch.Tensor(y_test)).sum()

    print('Done, accuracy: ', 1 - err.numpy() / X_test.shape[0])

    # Generate plots
    exp_version = exp.version
    media_dir = exp.get_media_path(name, exp_version)

    train_err_vec, train_ce_vec, test_err_vec, test_ce_vec = evaluate_per_depth_classsification(net, X_train, y_train,
                                                                                                X_test, y_test)
    savefile = media_dir + '/' + 'distribution'
    plot_depth_distributions(savefile, net, train_ce_vec=train_ce_vec, test_ce_vec=test_ce_vec, train_err_vec=train_err_vec,
                             test_err_vec=test_err_vec,
                             dpi=200, show=False, legend=False)

    savefile = media_dir + '/' + 'predictive'
    plot_predictive_2d_classification(savefile, net, X_train, y_train, extent=5.5, stepdim=350, dpi=200, show=False,
                                      batch_size=128)

    savefile = media_dir + '/' + 'callibration'
    plot_calibration_curve(savefile, net, X_test, y_test, n_bins=10, dpi=200, show=False)

    savefile = media_dir + '/' + 'perdepth_function'
    plot_layer_contributions_2d_classification(savefile, net, X_train, y_train, depths=None, extent=5.5, stepdim=150,
                                               dpi=100, show=False)