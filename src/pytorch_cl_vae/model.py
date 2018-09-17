import numpy as np
import scipy.stats
from sklearn.metrics import accuracy_score

from collections import namedtuple
import itertools

import torch
from torch.nn import Module, Linear, CrossEntropyLoss, BCELoss
import torch.nn.functional as F
from torch.distributions import Normal

"""
Here is the general flow of the classifying VAE:

Inputs:
X - input
w - label space variables (for k-classes problem we have k variables)
z - latent space variables

            |---------|                         |---------|      
X --------->| Encoder |----> z_mean, z_var ---->| Decoder |----> X^
            |---------|                         |---------|
                 ^                                  ^                                                  
                 |----------------------------------|
                                                    |
       |------------|                               |
w ---->| Classifier |----> w_mean, w_var ----> w^ --|
       |------------|

Losses:
Dkl(z, N(0,I))   - Kullback–Leibler divergence for latent variables z and normal distribution
Dkl(w^, LN(0,I)) - Kullback–Leibler divergence for label variables w and logit-normal distribution
CCE(w^, w)       - Categorical Cross Entropy loss for label variables (simple classification error)
BCE(X^, X)       - Binary Cross Entropy loss as a reconstruction loss for input X and reconstructed X^

Notes:
1)  During training we use true labels w for training Encoder and Decoder instead of w^. Thus during training we have 
    2 separate streams of computational graph: one for Classifier and consists of computing Dkl(w^, LN(0,I)) 
    and CCE(w^, w), another one for Encoder and Decoder and consists of computing Dkl(z, N(0,I)) and BCE(X^, X).
    During test we just substitute w with w^.
2)  We may want to have several classifiers at the same time e.g. style classifier, content classifier, etc. Thus,
    Encoder and Decoder should get a list of label vectors as an input
"""


class Encoder(Module):
    def __init__(self, **params):
        super(Encoder, self).__init__()
        self.__input_dim = params['original_dim']
        self.__classes_dims = params['classes_dim']
        self.__input_size = np.sum(self.__classes_dims) + self.__input_dim
        num_hidden = params['encoder_hidden_size']
        num_latent = params['latent_dim']
        self.__hidden = Linear(self.__input_size,  num_hidden)
        self.__latent_mean = Linear(num_hidden,  num_latent)
        self.__latent_var = Linear(num_hidden,  num_latent)

    def forward(self, x_input, ws):
        w_flat = torch.cat(ws, dim=-1)
        x = torch.cat((x_input, w_flat), dim=-1)
        x = self.__hidden(x)
        x = F.relu(x)
        z_mean = self.__latent_mean(x)
        z_log_var = self.__latent_var(x)
        return z_mean, z_log_var


class Decoder(Module):
    def __init__(self, **params):
        super(Decoder, self).__init__()

        self.__X_dim = params['original_dim']
        self.__latent_dim = params['latent_dim']
        self.__classes_dims = params['classes_dim']
        self.__input_size = np.sum(self.__classes_dims) + self.__latent_dim
        num_hidden = params['decoder_hidden_size']
        num_out = self.__X_dim
        self.__hidden = Linear(self.__input_size, num_hidden)
        self.__out = Linear(num_hidden, num_out)

    def forward(self, z, ws):
        w_flat = torch.cat(ws, dim=-1)
        x = torch.cat((z, w_flat), dim=-1)
        x = self.__hidden(x)
        x = F.relu(x)
        x = self.__out(x)
        x_decoded = torch.sigmoid(x)
        return x_decoded


class Classifier(Module):
    def __init__(self, **params):
        super(Classifier, self).__init__()
        self.__X_dim = params['original_dim']
        self.__classes_dim = params['label_dim']
        num_hidden = params['classifier_hidden_size']
        self.__hidden = Linear(self.__X_dim, num_hidden)
        self.__w_mean = Linear(num_hidden, self.__classes_dim-1)
        self.__w_log_var = Linear(num_hidden, self.__classes_dim-1)

    def forward(self, x):
        x = self.__hidden(x)
        x = F.relu(x)
        w_mean = self.__w_mean(x)
        w_log_var = self.__w_log_var(x)
        return w_mean, w_log_var


class ClVaeModel:
    def __init__(self, **kwargs):
        self.__params = kwargs
        self.__encoder = Encoder(**kwargs)
        self.__decoder = Decoder(**kwargs)
        self.__classifiers = [Classifier(label_dim=v, **kwargs) for i, v in enumerate(kwargs['classes_dim'])]
        self.__encoder_optimizer = torch.optim.Adam(self.__encoder.parameters(), lr=kwargs['vae_learning_rate'])
        self.__decoder_optimizer = torch.optim.Adam(self.__decoder.parameters(), lr=kwargs['vae_learning_rate'])
        self.__classifier_optimizers = [torch.optim.Adam(classifier.parameters(), lr=kwargs['classifier_learning_rate'])
                                        for classifier in self.__classifiers]
        self.__optimizers = self.__classifier_optimizers
        self.__optimizers.extend([self.__encoder_optimizer, self.__decoder_optimizer])

        self.__current_epoch = 0
        self.__current_losses = []
        self.__curent_accuracies = []

        losses_names = ['rec_loss', 'z_dkl_loss']
        acc_names = []
        for i in range(len(self.__classifiers)):
            losses_names.extend(['class_loss_'+str(i), 'w_dkl_loss_'+str(i)])
            acc_names.append('accuracy_'+str(i))
        self.Losses = namedtuple('Losses', losses_names)
        self.Accuracies = namedtuple("accuracies", acc_names)

    # Losses
    @staticmethod
    def z_Dkl_loss(z_mean, z_log_var):
        # loss = 0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - z_log_var - 1, dim=-1)
        # zs = ClVaeModel.z_sample(z_mean, z_log_var)
        # z_mean1 = zs.mean()
        # z_log_var1 = (zs.std()**2).log()
        #
        z_mean1 = z_mean.mean(dim=0)
        z_log_var1 = z_log_var.mean(dim=0) + (z_mean**2).mean(dim=0) - z_mean1**2

        loss = 0.5 * torch.sum(torch.exp(z_log_var1) + z_mean1**2 - z_log_var1 - 1, dim=-1).mean()
        return loss

    @staticmethod
    def w_Dkl_loss(w_mean, w_log_var, w_log_var_prior):
        # ws = ClVaeModel.w_sample(w_mean, w_log_var)
        # w_mean1 = ws.mean()
        # w_log_var1 = (ws.std() ** 2).log()
        w_mean1 = w_mean.mean(dim=0)
        w_log_var1 = w_log_var.mean(dim=0) + (w_mean ** 2).mean(dim=0) - w_mean1 ** 2

        vs = 1 - w_log_var_prior + w_log_var1 - torch.exp(w_log_var1) / torch.exp(w_log_var_prior)\
             - w_mean1**2 / torch.exp(w_log_var_prior)
        # loss = -0.5 * torch.sum(vs, dim=-1)
        loss = -0.5 * torch.sum(vs, dim=-1).mean()
        return loss

    @staticmethod
    def w_CCE_loss(w, w_true):
        num_dim = w.shape[-1] # as in the original code loss should be reduced sample-wise
        predictions = w
        # loss = CrossEntropyLoss()(predictions, w_true) * num_dim
        loss = CrossEntropyLoss()(predictions, w_true)
        return loss

    @staticmethod
    def x_BCE_loss(x, x_true):
        num_dim = x.shape[-1] # as in the original code loss should be reduced sample-wise
        # loss = BCELoss()(x, x_true) * num_dim
        loss = BCELoss()(x, x_true)
        return loss

    # Sampling
    @staticmethod
    def sample_x(x_mean):
        return 1.0 * (np.random.rand(len(x_mean.squeeze())) <= x_mean)

    @staticmethod
    def sample_w(*args, nsamps=1, nrm_samp=False, add_noise=True):
        w_mean, w_log_var = args
        if nsamps == 1:
            eps = np.random.randn(*((1, w_mean.flatten().shape[0])))
        else:
            eps = np.random.randn(*((nsamps,) + w_mean.shape))
        if eps.T.shape == w_mean.shape:
            eps = eps.T
        if add_noise:
            w_norm = w_mean + np.exp(w_log_var / 2) * eps
        else:
            w_norm = w_mean + 0 * np.exp(w_log_var / 2) * eps
        if nrm_samp:
            return w_norm
        if nsamps == 1:
            w_norm = np.hstack([w_norm, np.zeros((w_norm.shape[0], 1))])
            return np.exp(w_norm) / np.sum(np.exp(w_norm), axis=-1)[:, None]
        else:
            w_norm = np.dstack([w_norm, np.zeros(w_norm.shape[:-1] + (1,))])
            return np.exp(w_norm) / np.sum(np.exp(w_norm), axis=-1)[:, :, None]

    @staticmethod
    def sample_z(*args, nsamps=1):
        Z_mean, Z_log_var = args
        if nsamps == 1:
            eps = np.random.randn(*Z_mean.squeeze().shape)
        else:
            eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
        return Z_mean + np.exp(Z_log_var / 2) * eps

    @staticmethod
    def z_sample(*args):
        z_mean, z_log_var = args
        nrm = Normal(torch.zeros(z_mean.shape), torch.ones(z_mean.shape))
        eps = nrm.sample()
        # eps = torch.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + torch.exp(z_log_var / 2) * eps

    @staticmethod
    def w_sample(*args):
        """
            sample from a logit-normal with params w_mean and w_log_var
            (n.b. this is very similar to a logistic-normal distribution)
        """
        w_mean, w_log_var = args
        nrm = Normal(torch.zeros(w_mean.shape), torch.ones(w_mean.shape))
        eps = nrm.sample()
            # K.random_normal(shape=(batch_size, class_dim - 1), mean=0., stddev=1.0)
        # w_norm = w_mean + torch.exp(w_log_var / 2) * eps
        # # w_max = w_norm.max(1)[0]
        # # w_norm = w_norm - w_max.view(-1, 1).expand_as(w_norm) # trick to avoid inf in exp(w)
        # # need to add '0' so we can sum it all to 1
        # ones = torch.ones((w_mean.shape[0], 1))
        # zeros = torch.zeros((w_mean.shape[0], 1))
        # sums = 1 + torch.sum(torch.exp(w_norm), dim=1)
        # w_norm = torch.cat([w_norm, ones], dim=1)
        # w_sampled = torch.exp(w_norm) / sums[:, None]
        # return w_sampled

        w_mean, w_log_var = args
        # nrm = Normal(torch.zeros(w_mean.shape), torch.ones(w_mean.shape))
        w_norm = w_mean + torch.exp(w_log_var / 2) * eps
        # need to add '0' so we can sum it all to 1
        w_norm = torch.cat([w_norm, torch.zeros(w_mean.shape[0], 1)], dim=1)
        return torch.exp(w_norm) / torch.sum(torch.exp(w_norm), dim=-1)[:, None]

    def train_step(self, batch_x, batch_ws):
        # zero grad all optimizers
        for _, optimizer in enumerate(self.__optimizers):
            optimizer.zero_grad()

        # forward prop

        # generalization of vae
        w_sampled = batch_ws
        # encode
        z_mean, z_log_var = self.__encoder(batch_x, w_sampled)
        # sample z
        z = self.z_sample(z_mean, z_log_var)
        # decode
        x_decoded = self.__decoder(z, w_sampled)
        # classifiers forward prop
        ws_predicted = [classifier(batch_x) for classifier in self.__classifiers]

        # losses
        losses = []
        losses.append(self.x_BCE_loss(x_decoded, batch_x))
        losses.append(self.z_Dkl_loss(z_mean, z_log_var))

        accuracies = []

        w_true = w_sampled
        for i, (w_mean_pred, w_log_var_pred) in enumerate(ws_predicted):
            # sample w_pred
            w_pred = self.w_sample(w_mean_pred, w_log_var_pred)
            labels = w_true[i].max(1)[1].squeeze()
            labels_predict = w_pred.max(1)[1].squeeze()
            acc = accuracy_score(labels, labels_predict)
            accuracies.append(acc)
            w_cce_loss = self.w_CCE_loss(w_pred, labels)
            w_dkl_loss = self.w_Dkl_loss(w_mean_pred, w_log_var_pred, w_log_var_prior=torch.zeros(w_log_var_pred.shape))
            losses.append(w_cce_loss)
            losses.append(w_dkl_loss)

        # backward
        total_loss = torch.sum(torch.stack(losses))
        total_loss.backward()

        # step
        for optimizer in self.__optimizers:
            optimizer.step()

        named_losses = self.Losses._make(losses)
        named_accuracies = self.Accuracies._make(accuracies)
        return named_losses, named_accuracies

    def test(self, x, ws_true):
        labels_true = [ws_true[i].max(1)[1].squeeze() for i in range(len(ws_true))]
        ws_pred, w_means_and_log_vars_pred, labels_predict = self.classify(x)
        z, z_mean, z_log_var = self.encode(x, ws_pred)
        x_decoded = self.decode(z, ws_pred)

        cce_losses = [self.w_CCE_loss(ws_pred[i], labels_true[i]) for i in range(len(ws_true))]
        w_dkl_losses = [self.w_Dkl_loss(*w_means_and_log_vars_pred[i],
                                        w_log_var_prior=torch.zeros(w_means_and_log_vars_pred[i][1].shape)) for i in range(len(ws_true))]
        bce_loss = self.x_BCE_loss(x_decoded, x)
        z_dkl_loss = self.z_Dkl_loss(z_mean, z_log_var)


        accuracies = [accuracy_score(labels_true[i], labels_predict[i]) for i in range(len(labels_true))]

        losses = self.Losses._make([bce_loss, z_dkl_loss, *itertools.chain(*zip(cce_losses, w_dkl_losses))])
        accuracies = self.Accuracies._make(accuracies)
        return losses, accuracies

    def generate(self, **kwargs):
        raise NotImplementedError()
        pass

    def encode(self, x, ws):
        with torch.no_grad():
            z_mean, z_log_var = self.__encoder(x, ws)
            z = self.z_sample(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def decode(self, z, ws):
        with torch.no_grad():
            x_decoded = self.__decoder(z, ws)
        return x_decoded

    def classify(self, x):
        with torch.no_grad():
            w_predict = []
            labels_predict = []
            w_means_and_log_vars = [classifier(x) for classifier in self.__classifiers]
            for (w_mean, w_log_var) in w_means_and_log_vars:
                w = self.w_sample(w_mean, w_log_var)
                l = w.max(1)[1].squeeze()
                w_predict.append(w)
                labels_predict.append(l)
        return w_predict, w_means_and_log_vars, labels_predict

    def save_ckpt(self, fname):
        ckpt = {
            'params': self.__params,
            'epoch': self.__current_epoch,
            'losses': self.__current_losses,
            'accuracies': self.__curent_accuracies,
            'encoder': self.__encoder.state_dict(),
            'decoder': self.__decoder.state_dict(),
            'classifiers': [c.state_dict() for c in self.__classifiers],
            'optimizers': [optim.state_dict() for optim in self.__optimizers]
        }
        torch.save(ckpt, fname)

    @staticmethod
    def load_from_ckpt(fname):
        ckpt = torch.load(fname)
        model = ClVaeModel(**ckpt['params'])
        model.__load_from_ckpt(ckpt)
        return model

    def __load_from_ckpt(self, ckpt):
        self.__params = ckpt['params']
        self.__current_epoch = ckpt['params']
        self.__current_losses = ckpt['losses']
        self.__curent_accuracies = ckpt['accuracies']

        self.__encoder.load_state_dict(ckpt['encoder'])
        self.__decoder.load_state_dict(ckpt['decoder'])

        for i, c in enumerate(self.__classifiers):
            c.load_state_dict(ckpt['classifiers'][i])

        for i, o in enumerate(self.__optimizers):
            o.load_state_dict(ckpt['optimizers'][i])
