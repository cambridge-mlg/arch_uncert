from __future__ import print_function
from __future__ import division
import torch
import time
import numpy as np
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from test_tube import Experiment

from src.utils import mkdir, cprint


def train_VI_classification(net, name, save_dir, batch_size, nb_epochs, trainset, valset, cuda,
                                     flat_ims=False, nb_its_dev=1, early_stop=None, load_path=None, save_freq=20,
                                    stop_criteria='test_ELBO', tags=None, show=False):
    exp = Experiment(name=name,
                     debug=False,
                     save_dir=save_dir,
                     autosave=True)

    if load_path is not None:
        net.load(load_path)

    exp_version = exp.version

    media_dir = exp.get_media_path(name, exp_version)
    models_dir = exp.get_data_path(name, exp_version) + '/models'
    mkdir(models_dir)

    exp.tag({
        'n_layers': net.model.n_layers,
        'batch_size': batch_size,
        'init_lr': net.lr,
        'lr_schedule': net.schedule,
        'nb_epochs': nb_epochs,
        'early_stop': early_stop,
        'stop_criteria': stop_criteria,
        'nb_its_dev': nb_its_dev,
        'model_loaded': load_path,
        'cuda': cuda,
    })

    if net.model.__class__.__name__ == 'arq_uncert_conv2d_resnet':
        exp.tag({
            'outer_width': net.model.outer_width,
            'inner_width': net.model.inner_width
            })
    else: 
        exp.tag({'width': net.model.width})

    exp.tag({
        'prob_model': net.model.prob_model.name,
        'prob_model_summary': net.model.prob_model.summary
    })
    if tags is not None:
        exp.tag(tags)

    if cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)
    ## ---------------------------------------------------------------------------------------------------------------------
    # net dims
    cprint('c', '\nNetwork:')
    epoch = 0
    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    mloglike_train = np.zeros(nb_epochs)
    KL_train = np.zeros(nb_epochs)
    ELBO_train = np.zeros(nb_epochs)
    ELBO_test = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    mloglike_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_epoch = 0
    best_train_ELBO = -np.inf
    best_test_ELBO = -np.inf
    best_dev_ll = -np.inf

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)

            KL, minus_loglike, err = net.fit(x, y)
            err_train[i] += err
            mloglike_train[i] += minus_loglike/len(trainloader)
            KL_train[i] += KL/len(trainloader)
            nb_samples += len(x)

        # mloglike_train[i] *= nb_samples
        # KL_train[i] *= nb_samples
        ELBO_train[i] = (-KL_train[i] - mloglike_train[i]) * nb_samples
        err_train[i] /= nb_samples

        toc = time.time()

        # ---- print
        print("it %d/%d, sample minus loglike = %f, sample KL = %.10f, err = %f, ELBO = %f" % \
              (i, nb_epochs, mloglike_train[i], KL_train[i], err_train[i], ELBO_train[i]), end="")
        exp.log({'epoch': i, 'MLL': mloglike_train[i], 'KLD': KL_train[i], 'err': err_train[i], 'ELBO': ELBO_train[i]})
        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i, 0.1)

        # ---- dev
        if i % nb_its_dev == 0:
            tic = time.time()
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                minus_loglike, err = net.eval(x, y)

                mloglike_dev[i] += minus_loglike / len(valloader)
                err_dev[i] += err
                nb_samples += len(x)
                
            ELBO_test[i] = (-KL_train[i] - mloglike_dev[i]) * nb_samples

            ELBO_test[i] = (-KL_train[i] - mloglike_dev[i]) * nb_samples
            err_dev[i] /= nb_samples
            toc = time.time()

            cprint('g', '    sample minus loglike = %f, err = %f, ELBO = %f\n' % (mloglike_dev[i], err_dev[i], ELBO_test[i]), end="")
            cprint('g', '    (prev best it = %i, sample minus loglike = %f, ELBO = %f)\n' % (best_epoch, best_dev_ll, best_test_ELBO), end="")
            cprint('g', '    time: %f seconds\n' % (toc - tic))
            exp.log({'epoch': i, 'MLL_val': mloglike_dev[i], 'err_val': err_dev[i], 'ELBO_val': ELBO_test[i]})

            if stop_criteria == 'test_LL' and -mloglike_dev[i] > best_dev_ll:
                best_dev_ll = -mloglike_dev[i]
                best_epoch = i
                cprint('b', 'best test loglike: %d' % best_dev_ll)
                net.save(models_dir + '/theta_best.dat')
                probs = net.model.prob_model.get_q_probs().data.cpu().numpy()
                cuttoff = np.max(probs)*0.95
                exp.tag({"q_vec": net.model.get_q_vector().cpu().detach().numpy(),
                         "q_probs": net.model.prob_model.get_q_probs().cpu().detach().numpy(),
                         "expected_depth": np.sum(probs * np.arange(net.model.n_layers + 1)),
                         "95th_depth": np.argmax(probs > cuttoff),                         "best_epoch": best_epoch,
                         "best_dev_ll": best_dev_ll})

            if stop_criteria == 'test_ELBO' and ELBO_test[i] > best_test_ELBO:
                best_test_ELBO = ELBO_test[i]
                best_epoch = i
                cprint('b', 'best test ELBO: %d' % best_test_ELBO)
                net.save(models_dir + '/theta_best.dat')
                probs = net.model.prob_model.get_q_probs().data.cpu().numpy()
                cuttoff = np.max(probs)*0.95
                exp.tag({"q_vec": net.model.get_q_vector().cpu().detach().numpy(), 
                        "q_probs": net.model.prob_model.get_q_probs().cpu().detach().numpy(),
                        "expected_depth": np.sum(probs * np.arange(net.model.n_layers + 1)),
                        "95th_depth": np.argmax(probs > cuttoff),
                        "best_epoch": best_epoch,
                        "best_test_ELBO": best_test_ELBO})
        
        if stop_criteria == 'train_ELBO' and ELBO_train[i] > best_train_ELBO:
            best_train_ELBO = ELBO_train[i]
            best_epoch = i
            cprint('b', 'best train ELBO: %d' % best_train_ELBO)
            net.save(models_dir + '/theta_best.dat')
            probs = net.model.prob_model.get_q_probs().data.cpu().numpy()
            cuttoff = np.max(probs)*0.95
            exp.tag({"q_vec": net.model.get_q_vector().cpu().detach().numpy(), 
                     "q_probs": net.model.prob_model.get_q_probs().cpu().detach().numpy(),
                     "expected_depth": np.sum(probs * np.arange(net.model.n_layers + 1)),
                     "95th_depth": np.argmax(probs > cuttoff),
                     "best_epoch": best_epoch,
                     "best_train_ELBO": best_train_ELBO})

        if save_freq is not None and i % save_freq == 0:
            exp.tag({
                "final_q_vec": net.model.get_q_vector().cpu().detach().numpy(),
                "final_q_probs": net.model.prob_model.get_q_probs().cpu().detach().numpy(),
                "final_expected_depth": np.sum(net.model.prob_model.get_q_probs().data.cpu().numpy() * np.arange(net.model.n_layers + 1))
                })
            net.save(models_dir + '/theta_last.dat')

        if early_stop is not None and (i - best_epoch) > early_stop:
            exp.tag({"early_stop_epoch": i})
            cprint('r', '   stopped early!\n')
            break

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(i + 1)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its
    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, i, nb_its_dev), np.clip(mloglike_dev[:i:nb_its_dev], a_min=-5, a_max=5), 'b-')
    ax1.plot(np.clip(mloglike_train[:i], a_min=-5, a_max=5), 'r--')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['test', 'train'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(media_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show:
        plt.show()

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, i), KL_train[:i], 'b-')
    ax1.set_ylabel('KL')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['KL'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('KL divideed by number of samples')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(media_dir + '/KL.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show:
        plt.show()


    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, i), ELBO_train[:i], 'b-')
    ax1.set_ylabel('nats')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['ELBO'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('ELBO')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(media_dir + '/ELBO.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show:
        plt.show()


    plt.figure(dpi=100)
    fig, ax2 = plt.subplots()
    ax2.set_ylabel('% error')
    ax2.semilogy(range(0, i, nb_its_dev), err_dev[:i:nb_its_dev], 'b-')
    ax2.semilogy(err_train[:i], 'r--')
    ax2.set_ylim(top=1, bottom=1e-3)
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(media_dir + '/err.png', bbox_extra_artists=(lgd,), box_inches='tight')
    if show:
        plt.show()

    return exp, mloglike_train, KL_train, ELBO_train, err_train, mloglike_dev, err_dev

