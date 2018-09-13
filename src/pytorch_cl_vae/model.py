import json
import numpy as np
import scipy.stats
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
        x_decoded = F.sigmoid(x)
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
        self.__encoder = Encoder(**kwargs)
        self.__decoder = Decoder(**kwargs)
        self.__classifiers = [Classifier(label_dim=v, **kwargs) for i, v in enumerate(kwargs['classes_dim'])]
        self.__encoder_optimizer = torch.optim.Adam(self.__encoder.parameters(), lr=kwargs['vae_learning_rate'])
        self.__decoder_optimizer = torch.optim.Adam(self.__decoder.parameters(), lr=kwargs['vae_learning_rate'])
        self.__classifier_optimizers = [torch.optim.Adam(classifier.parameters(), lr=kwargs['classifier_learning_rate'])
                                        for classifier in self.__classifiers]
        self.__optimizers = self.__classifier_optimizers
        self.__optimizers.extend([self.__encoder_optimizer, self.__decoder_optimizer ])
    # Losses
    @staticmethod
    def z_Dkl_loss(z_mean, z_log_var):
        loss = 0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - z_log_var - 1, dim=-1)
        return loss

    @staticmethod
    def w_Dkl_loss(w_mean, w_log_var, w_log_var_prior):
        vs = 1 - w_log_var_prior + w_log_var - torch.exp(w_log_var) / torch.exp(w_log_var_prior)\
             - w_mean**2 / torch.exp(w_log_var_prior)
        return -0.5 * torch.sum(vs, dim=-1)

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

    def z_sample(self, *args):
        z_mean, z_log_var = args
        nrm = Normal(torch.zeros(z_mean.shape), torch.ones(z_mean.shape))
        eps = nrm.sample()
        # eps = torch.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + torch.exp(z_log_var / 2) * eps

    def w_sample(self, *args):
        """
                sample from a logit-normal with params w_mean and w_log_var
                    (n.b. this is very similar to a logistic-normal distribution)
                """
        w_mean, w_log_var = args
        nrm = Normal(torch.zeros(w_mean.shape), torch.ones(w_mean.shape))
        eps = nrm.sample()
            # K.random_normal(shape=(batch_size, class_dim - 1), mean=0., stddev=1.0)
        w_norm = w_mean + torch.exp(w_log_var / 2) * eps
        w_max = w_norm.max(1)[0]
        w_norm = w_norm - w_max.view(-1, 1).expand_as(w_norm) # trick to avoid inf in exp(w)
        # need to add '0' so we can sum it all to 1
        ones = torch.ones((w_mean.shape[0], 1))
        sums = 1 + torch.sum(torch.exp(w_norm), dim=1)
        w_norm = torch.cat([w_norm, ones], dim=1)
        w_sampled = torch.exp(w_norm) / sums[:, None]
        return w_sampled


    def train_step(self, batch_x, batch_ws):
        # zero grad all optimizers
        for _, optimizer in enumerate(self.__optimizers):
            optimizer.zero_grad()

        # forward prop

        # TODO: need to sample w before forward prop? Probably wee need to add noise even to true labels for better
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

        # TODO: need to add noise to true w?
        w_true = w_sampled

        for i, (w_mean_pred, w_log_var_pred) in enumerate(ws_predicted):
            # sample w_pred
            w_pred = self.w_sample(w_mean_pred, w_log_var_pred)
            labels = w_true[i].max(1)[1].squeeze()
            w_cce_loss = self.w_CCE_loss(w_pred, labels)
            w_dkl_loss = self.w_Dkl_loss(w_mean_pred, w_log_var_pred, w_log_var_prior=torch.ones(w_log_var_pred.shape))
            losses.append(w_cce_loss)
            losses.append(w_dkl_loss)

        # backward
        for loss in losses:
            loss.sum().backward(retain_graph=True)

        # step
        for optimizer in self.__optimizers:
            optimizer.step()

        return losses

    def test(self, **kwargs):
        pass

    def generate(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass



##################################################################
# END
##################################################################

#
#
# def generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, nsteps, w_val=None, use_z_prior=False, do_reset=True, w_sample=False, use_x_prev=False):
#     """
#     for t = 1:nsteps
#         1. encode x_seed -> w_mean, w_log_var
#         2. sample w_t ~ logit-N(w_mean, exp(w_log_var/2))
#         3. encode x_seed, w_t -> z_mean, z_log_var
#         4. sample z_t ~ N(z_mean, exp(z_log_var/2))
#         3. decode w_t, z_t -> x_mean
#         4. sample x_t ~ Bern(x_mean)
#         5. update x_seed := x_t
#     """
#     original_dim = x_seed.shape[0]
#     Xs = np.zeros([nsteps, original_dim])
#     x_prev = np.expand_dims(x_seed, axis=0)
#     x_prev_t = x_prev
#     if w_val is None:
#         w_t = sample_w(w_enc_model.predict(x_prev), add_noise=w_sample)
#     else:
#         w_t = w_val
#     for t in xrange(nsteps):
#         z_mean, z_log_var = z_enc_model.predict([x_prev, w_t])
#         if use_z_prior:
#             z_t = sample_z((0*z_mean, 0*z_log_var))
#         else:
#             z_t = sample_z((z_mean, z_log_var))
#         if use_x_prev:
#             zc = [w_t, z_t, x_prev_t]
#         else:
#             zc = [w_t, z_t]
#         x_t = sample_x(dec_model.predict(zc))
#         Xs[t] = x_t
#         x_prev_t = x_prev
#         x_prev = x_t
#     return Xs
#
# def sample_x(x_mean):
#     return 1.0*(np.random.rand(len(x_mean.squeeze())) <= x_mean)
#
# def sample_w(args, nsamps=1, nrm_samp=False, add_noise=True):
#     w_mean, w_log_var = args
#     if nsamps == 1:
#         eps = np.random.randn(*((1, w_mean.flatten().shape[0])))
#     else:
#         eps = np.random.randn(*((nsamps,) + w_mean.shape))
#     if eps.T.shape == w_mean.shape:
#         eps = eps.T
#     if add_noise:
#         w_norm = w_mean + np.exp(w_log_var/2)*eps
#     else:
#         w_norm = w_mean + 0*np.exp(w_log_var/2)*eps
#     if nrm_samp:
#         return w_norm
#     if nsamps == 1:
#         w_norm = np.hstack([w_norm, np.zeros((w_norm.shape[0], 1))])
#         return np.exp(w_norm)/np.sum(np.exp(w_norm), axis=-1)[:,None]
#     else:
#         w_norm = np.dstack([w_norm, np.zeros(w_norm.shape[:-1]+ (1,))])
#         return np.exp(w_norm)/np.sum(np.exp(w_norm), axis=-1)[:,:,None]
#
# def sample_z(args, nsamps=1):
#     Z_mean, Z_log_var = args
#     if nsamps == 1:
#         eps = np.random.randn(*Z_mean.squeeze().shape)
#     else:
#         eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
#     return Z_mean + np.exp(Z_log_var/2) * eps
#
# def make_w_encoder(model, original_dim, batch_size=1):
#     x = Input(batch_shape=(batch_size, original_dim), name='x')
#
#     # build label encoder
#     h_w = model.get_layer('h_w')(x)
#     w_mean = model.get_layer('w_mean')(h_w)
#     w_log_var = model.get_layer('w_log_var')(h_w)
#
#     mdl = Model(x, [w_mean, w_log_var])
#     return mdl
#
# def make_z_encoder(model, original_dim, class_dim, (latent_dim_0, latent_dim), batch_size=1):
#     x = Input(batch_shape=(batch_size, original_dim), name='x')
#     w = Input(batch_shape=(batch_size, class_dim), name='w')
#     xw = concatenate([x, w], axis=-1)
#
#     # build latent encoder
#     if latent_dim_0 > 0:
#         h = model.get_layer('h')(xw)
#         z_mean = model.get_layer('z_mean')(h)
#         z_log_var = model.get_layer('z_log_var')(h)
#     else:
#         z_mean = model.get_layer('z_mean')(xw)
#         z_log_var = model.get_layer('z_log_var')(xw)
#
#     mdl = Model([x, w], [z_mean, z_log_var])
#     return mdl
#
# def make_decoder(model, (latent_dim_0, latent_dim), class_dim, original_dim=88, use_x_prev=False, batch_size=1):
#     w = Input(batch_shape=(batch_size, class_dim), name='w')
#     z = Input(batch_shape=(batch_size, latent_dim), name='z')
#     if use_x_prev:
#         xp = Input(batch_shape=(batch_size, original_dim), name='history')
#     if use_x_prev:
#         xpz = concatenate([xp, z], axis=-1)
#     else:
#         xpz = z
#     wz = concatenate([w, xpz], axis=-1)
#
#     # build x decoder
#     decoder_mean = model.get_layer('x_decoded_mean')
#     if latent_dim_0 > 0:
#         decoder_h = model.get_layer('decoder_h')
#         h_decoded = decoder_h(wz)
#         x_decoded_mean = decoder_mean(h_decoded)
#     else:
#         x_decoded_mean = decoder_mean(wz)
#
#     if use_x_prev:
#         mdl = Model([w, z, xp], x_decoded_mean)
#     else:
#         mdl = Model([w, z], x_decoded_mean)
#     return mdl
#
# def get_model(batch_size, original_dim,
#     (latent_dim_0, latent_dim),
#     (class_dim_0, class_dim), optimizer,
#     class_weight=1.0, kl_weight=1.0, use_x_prev=False,
#     w_kl_weight=1.0, w_log_var_prior=0.0):
#
#     x = Input(batch_shape=(batch_size, original_dim), name='x')
#     if use_x_prev:
#         xp = Input(batch_shape=(batch_size, original_dim), name='history')
#
#     # build label encoder
#     h_w = Dense(class_dim_0, activation='relu', name='h_w')(x)
#     w_mean = Dense(class_dim-1, name='w_mean')(h_w)
#     w_log_var = Dense(class_dim-1, name='w_log_var')(h_w)
#
#     # sample label
#     def w_sampling(args):
#         """
#         sample from a logit-normal with params w_mean and w_log_var
#             (n.b. this is very similar to a logistic-normal distribution)
#         """
#         w_mean, w_log_var = args
#         eps = K.random_normal(shape=(batch_size, class_dim-1), mean=0., stddev=1.0)
#         w_norm = w_mean + K.exp(w_log_var/2) * eps
#         # need to add '0' so we can sum it all to 1
#         w_norm = concatenate([w_norm, K.tf.zeros(batch_size, 1)[:,None]])
#         return K.exp(w_norm)/K.sum(K.exp(w_norm), axis=-1)[:,None]
#     w = Lambda(w_sampling, name='w')([w_mean, w_log_var])
#
#     # build latent encoder
#     xw = concatenate([x, w], axis=-1)
#     if latent_dim_0 > 0:
#         h = Dense(latent_dim_0, activation='relu', name='h')(xw)
#         z_mean = Dense(latent_dim, name='z_mean')(h)
#         z_log_var = Dense(latent_dim, name='z_log_var')(h)
#     else:
#         z_mean = Dense(latent_dim, name='z_mean')(xw)
#         z_log_var = Dense(latent_dim, name='z_log_var')(xw)
#
#     # sample latents
#     def sampling(args):
#         z_mean, z_log_var = args
#         eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
#         return z_mean + K.exp(z_log_var/2) * eps
#     z = Lambda(sampling, name='z')([z_mean, z_log_var])
#
#     # build decoder
#     if use_x_prev:
#         xpz = concatenate([xp, z], axis=-1)
#     else:
#         xpz = z
#     wz = concatenate([w, xpz], axis=-1)
#     decoder_mean = Dense(original_dim, activation='sigmoid', name='x_decoded_mean')
#     if latent_dim_0 > 0:
#         decoder_h = Dense(latent_dim_0, activation='relu', name='decoder_h')
#         h_decoded = decoder_h(wz)
#         x_decoded_mean = decoder_mean(h_decoded)
#     else:
#         x_decoded_mean = decoder_mean(wz)
#
#     def vae_loss(x, x_decoded_mean):
#         return original_dim * losses.binary_crossentropy(x, x_decoded_mean)
#
#     def kl_loss(z_true, z_args):
#         Z_mean = z_args[:,:latent_dim]
#         Z_log_var = z_args[:,latent_dim:]
#         return -0.5*K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)
#
#     def w_rec_loss(w_true, w):
#         return (class_dim-1) * losses.categorical_crossentropy(w_true, w)
#
#     # w_log_var_prior = 1.0
#     def w_kl_loss(w_true, w):
#         # w_log_var_prior
#         # return -0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
#         vs = 1 - w_log_var_prior + w_log_var - K.exp(w_log_var)/K.exp(w_log_var_prior) - K.square(w_mean)/K.exp(w_log_var_prior)
#         return -0.5*K.sum(vs, axis=-1)
#
#     w2 = Lambda(lambda x: x+1e-10, name='w2')(w)
#     z_args = concatenate([z_mean, z_log_var], axis=-1, name='z_args')
#     if use_x_prev:
#         model = Model([x, xp], [x_decoded_mean, w, w2, z_args])
#         enc_model = Model([x, xp], [z_mean, w_mean])
#     else:
#         model = Model(x, [x_decoded_mean, w, w2, z_args])
#         enc_model = Model(x, [z_mean, w_mean])
#     model.compile(optimizer=optimizer,
#         loss={'x_decoded_mean': vae_loss, 'w': w_kl_loss, 'w2': w_rec_loss, 'z_args': kl_loss},
#         loss_weights={'x_decoded_mean': 1.0, 'w': w_kl_weight, 'w2': class_weight, 'z_args': kl_weight},
#         metrics={'w': 'accuracy'})
#     if use_x_prev:
#         enc_model = Model([x, xp], [z_mean, w_mean])
#     else:
#         enc_model = Model(x, [z_mean, w_mean])
#     return model, enc_model
#
# def load_model(model_file, optimizer='adam', batch_size=1, no_x_prev=False):
#     """
#     there's a curently bug in the way keras loads models from .yaml
#         that has to do with Lambdas
#     so this is a hack for now...
#     """
#     margs = json.load(open(model_file.replace('.h5', '.json')))
#     # model = model_from_yaml(open(args.model_file))
#     batch_size = margs['batch_size'] if batch_size == None else batch_size
#     if no_x_prev or 'use_x_prev' not in margs:
#         margs['use_x_prev'] = False
#     model, enc_model = get_model(batch_size, margs['original_dim'], (margs['intermediate_dim'], margs['latent_dim']), (margs['intermediate_class_dim'], margs['n_classes']), optimizer, margs['class_weight'], use_x_prev=margs['use_x_prev'])
#     model.load_weights(model_file)
#     return model, enc_model, margs
