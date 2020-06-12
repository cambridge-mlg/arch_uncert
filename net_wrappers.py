from __future__ import division, print_function
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from src.utils import BaseNet, cprint, to_variable


class MF_BNN_cat(BaseNet):  # for categorical distributions (classification)
    def __init__(self, model, N_train, lr=1e-2, cuda=True, schedule=None):
        super(MF_BNN_cat, self).__init__()

        cprint('y', 'MF BNN categorical output')
        self.lr = lr
        self.model = model
        self.cuda = cuda
        self.f_neg_loglike = F.cross_entropy  # TODO restructure declaration of this function

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = schedule  # [] #[50,200,400,600]
        self.epoch = 0

    def create_net(self):
        # torch.manual_seed(42)
        if self.cuda:
            # torch.cuda.manual_seed(42)
            pass
        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5, weight_decay=0)

    def fit(self, x, y):
        """Optimise ELBO treating model weights as hyperparameters"""
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()

        act_vec = self.model.forward_get_acts(x)
        mean_minus_loglike = self.model.prob_model.efficient_E_loglike(act_vec, y, self.f_neg_loglike)  # returns sample mean over batch
        probs = self.model.prob_model.efficient_predict(act_vec, softmax=True).sum(dim=0).data

        KL_persample = self.model.get_KL() / self.N_train
        m_ELBO = mean_minus_loglike +  KL_persample
        m_ELBO.backward()
        self.optimizer.step()

        pred = probs.max(dim=1, keepdim=False)[1]  # get the index of the max probability
        err = pred.ne(y.data).sum()
        return KL_persample.item(), mean_minus_loglike.data.item(), err

    def eval(self, x, y):
        # TODO: make computationally stable with logsoftmax and nll loss
        self.set_mode_train(train=False)
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        act_vec = self.model.forward_get_acts(x)
        probs = self.model.prob_model.efficient_predict(act_vec, softmax=True).sum(dim=0).data

        minus_loglike = F.nll_loss(torch.log(probs), y, reduction='mean')
        pred = probs.max(dim=1, keepdim=False)[1]
        err = pred.ne(y.data).sum()

        return minus_loglike, err

    def vec_predict(self, x, bin_mat):
        """Get predictions for specific binary vector configurations"""
        self.set_mode_train(train=False)
        x, = to_variable(var=(x, ), cuda=self.cuda)
        out = x.data.new(bin_mat.shape[0], x.shape[0], self.model.output_dim)
        for s in range(bin_mat.shape[0]):
            out[s] = self.model.vec_forward(x, bin_mat[s,:]).data
        prob_out = F.softmax(out, dim=2)
        return prob_out

    def sample_predict(self, x, grad=False):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x,), cuda=self.cuda)
        act_vec = self.model.forward_get_acts(x)
        probs = self.model.prob_model.efficient_predict(act_vec, softmax=True)
        # Note that these are weighed probs that need to be summed in dim 0 to be actual probs
        if grad:
            return probs
        else:
            return probs.data

    def partial_predict(self, x, depth=None):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x,), cuda=self.cuda)
        
        if depth is None:
            _, depth = self.model.prob_model.get_q_probs().max(dim=0)

        act_vec = self.model.forward_get_acts(x, depth=depth)

        probs = self.model.prob_model.efficient_predict_d(act_vec, depth, softmax=True)
        # Note that these are weighed probs that need to be summed in dim 0 to be actual probs
        return probs
