import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

from helpers import *

class Layered_Linear_batchnorm_tanh_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z1_dim, z2_dim, z3_dim, n_layers, dropout_p, bias=False, device=None):
        super(Layered_Linear_batchnorm_tanh_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Tanh())
        self.phi_z = nn.Sequential(
            nn.Linear(z3_dim, z3_dim),
            nn.Tanh())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.Dropout(p=dropout_p),
            nn.Tanh(),
            nn.Linear(z1_dim, z2_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.Dropout(p=dropout_p),
            nn.Tanh())
        self.enc_mean = nn.Linear(z2_dim, z3_dim)
        self.enc_std = nn.Linear(z2_dim, z3_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z3_dim),
            nn.Tanh())
        self.prior_mean = nn.Linear(z3_dim, z3_dim)
        self.prior_std = nn.Linear(z3_dim, z3_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z3_dim + h_dim, z2_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.Dropout(p=dropout_p),
            nn.Tanh(),
            nn.Linear(z2_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.Dropout(p=dropout_p),
            nn.Tanh())
        self.dec_mean = nn.Linear(z1_dim, x_dim)
        self.dec_std = nn.Linear(z1_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z3_dim, h_dim, 2)

        # classifier
        # self.fc_out = nn.Linear(h_dim, 2)
        self.fc_out = nn.Linear(h_dim, 1)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)

        out = self.fc_out(all_h[-1])
        # out_prob = torch.sigmoid(out).squeeze(0)
        out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class Layered_3_Linear_batchnorm_tanh_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z1_dim, z2_dim, n_layers, dropout_p, bias=False, device=None):
        super(Layered_3_Linear_batchnorm_tanh_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Tanh())
        self.phi_z = nn.Sequential(
            nn.Linear(z2_dim, z2_dim),
            nn.Tanh())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.Dropout(p=dropout_p),
            # nn.Tanh(),
            # nn.Linear(z1_dim, z2_dim),
            # nn.BatchNorm1d(num_features=z2_dim),
            # nn.Dropout(p=dropout_p),
            nn.Tanh())
        self.enc_mean = nn.Linear(z1_dim, z2_dim)
        self.enc_std = nn.Linear(z1_dim, z2_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z2_dim),
            nn.Tanh())
        self.prior_mean = nn.Linear(z2_dim, z2_dim)
        self.prior_std = nn.Linear(z2_dim, z2_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z2_dim + h_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.Dropout(p=dropout_p),
            # nn.Tanh(),
            # nn.Linear(z2_dim, z1_dim),
            # nn.BatchNorm1d(num_features=z1_dim),
            # nn.Dropout(p=dropout_p),
            nn.Tanh())
        self.dec_mean = nn.Linear(z1_dim, x_dim)
        self.dec_std = nn.Linear(z1_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z2_dim, h_dim, 2)

        # classifier
        # self.fc_out = nn.Linear(h_dim, 2)
        self.fc_out = nn.Linear(h_dim, 1)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)

        out = self.fc_out(all_h[-1])
        # out_prob = torch.sigmoid(out).squeeze(0)
        out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class Layered_2_Linear_batchnorm_tanh_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z1_dim, z2_dim, z3_dim, z4_dim, n_layers, dropout_p, bias=False, device=None):
        super(Layered_2_Linear_batchnorm_tanh_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.z4_dim = z4_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Tanh())
        self.phi_z = nn.Sequential(
            nn.Linear(z4_dim, z4_dim),
            nn.Tanh())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.Dropout(p=dropout_p),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Linear(z1_dim, z2_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.Dropout(p=dropout_p),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Linear(z2_dim, z3_dim),
            nn.BatchNorm1d(num_features=z3_dim),
            nn.Dropout(p=dropout_p),
            # nn.ELU()
            nn.LeakyReLU(),
            )
        self.enc_mean = nn.Linear(z3_dim, z4_dim)
        self.enc_std = nn.Linear(z3_dim, z4_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z4_dim),
            # nn.ELU()
            nn.LeakyReLU()
        )
        self.prior_mean = nn.Linear(z4_dim, z4_dim)
        self.prior_std = nn.Linear(z4_dim, z4_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z4_dim + h_dim, z3_dim),
            nn.BatchNorm1d(num_features=z3_dim),
            nn.Dropout(p=dropout_p),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Linear(z3_dim, z2_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.Dropout(p=dropout_p),
            # nn.ELU(),
            nn.LeakyReLU(),
            nn.Linear(z2_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            # nn.Dropout(p=dropout_p),
            # nn.ELU()
            nn.LeakyReLU()
        )
        self.dec_mean = nn.Linear(z1_dim, x_dim)
        self.dec_std = nn.Linear(z1_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z4_dim, h_dim, 2)

        # classifier
        # self.fc_out = nn.Linear(h_dim, 2)
        self.fc_out = nn.Linear(h_dim, 1)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            # print('encoder')
            # print(enc_mean_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            # print('prior')
            # print(prior_mean_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            # print('posterior')
            # print(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            # print('decoder')
            # print(dec_mean_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)
            # print('gruu')
            # print(h)

            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)

        out = self.fc_out(all_h[-1])
        # print(out)
        # out_prob = torch.sigmoid(out).squeeze(0)
        out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.m():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)
            # nn.init.normal_(weight, -stv, stv)
            # nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class Linear_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
        super(Linear_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU())
        # self.phi_x = nn.Sequential(
        # 	nn.Linear(x_dim, h_dim),
        # 	nn.ReLU(),
        # 	nn.Linear(h_dim, h_dim),
        # 	nn.ReLU())
        # self.phi_z = nn.Sequential(
        # 	nn.Linear(z_dim, h_dim),
        # 	nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            # nn.Linear(h_dim + h_dim, z_dim),
            nn.Linear(x_dim + h_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU())  ###################################################
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)


        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Linear(z_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            # nn.Linear(h_dim + h_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())  #####################################################
        self.dec_std = nn.Linear(x_dim, x_dim)
        self.dec_mean = nn.Linear(x_dim, x_dim)
        # self.dec_mean = nn.Sequential(
        # 	nn.Linear(x_dim, x_dim),
        # 	nn.Sigmoid())  ####################################

        # recurrence
        # self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        # self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)

        self.fc_out = nn.Linear(h_dim, 2)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

        out = self.fc_out(all_h[-1])
        out_prob = torch.sigmoid(out).squeeze(0)



        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class BiVRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, out_ch, dropout_p, isdecaying, FFA, isreparam=False, issampling=False, device=None):
        super(BiVRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.out_ch = out_ch
        self.dropout_p = dropout_p
        self.isdecaying = isdecaying
        self.device = device
        self.FFA = FFA
        self.isreparam = isreparam
        self.issampling = issampling

        self.build()

    def build(self):
        self.VRNN_fwd = Linear_batchnorm_tanh_VRNN_GRUU(self.x_dim, self.h_dim, self.z_dim, self.n_layers, self.out_ch, self.dropout_p, self.isdecaying, self.FFA, self.isreparam, self.issampling)
        self.VRNN_bwd = Linear_batchnorm_tanh_VRNN_GRUU(self.x_dim, self.h_dim, self.z_dim, self.n_layers, self.out_ch, self.dropout_p, self.isdecaying, self.FFA, self.isreparam, self.issampling)

    def forward(self, x_fwd, m_fwd, delats_fwd, x_bwd, m_bwd, delats_bwd):
        vrnn_f = self.VRNN_fwd(x_fwd, m_fwd, delats_fwd)
        vrnn_b = self.VRNN_bwd(x_bwd, m_bwd, delats_bwd)
        return vrnn_f, vrnn_b


class Linear_batchnorm_tanh_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, out_ch, dropout_p, isdecaying, FFA=False, isreparam=False, issampling=False, device=None):
        super(Linear_batchnorm_tanh_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.out_ch = out_ch
        self.dropout_p = dropout_p
        self.isdecaying = isdecaying
        self.device = device
        self.FFA = FFA
        self.isreparam = isreparam
        self.issampling = issampling

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(x_dim, x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh())
            # nn.ELU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            # nn.BatchNorm1d(num_features=z_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(z_dim, z_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh())
            # nn.LeakyReLU())
            # nn.ELU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Linear(z_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        self.dec_std = nn.Linear(x_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2, self.out_ch, isdecaying=self.isdecaying)  # GRU-U
        # self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers, bias)  # Vanilla GRU

        # attention
        self.attention = nn.Sequential(
            nn.Linear(h_dim, 1, bias=False),
            nn.Tanh())
        self.attention_bias = nn.Parameter(torch.zeros(1))


        self.add_nn = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            # nn.LeakyReLU())
            nn.Tanh())
        self.nn_bias = nn.Parameter(torch.zeros(h_dim))

        # self.add_nn_2 = nn.Sequential(
        #     nn.Linear(h_dim, 1, bias=False),
        #     nn.LeakyReLU())
        # self.nn_bias_2 = nn.Parameter(torch.zeros(1))


        # classifier
        self.fc_out = nn.Linear(h_dim, 1)
        # self.fc_out = nn.Linear(h_dim, 2)
        # self.fc_out = nn.Sequential(
        # 	nn.Linear(h_dim, 2),
        # 	nn.Dropout(p=0.3))

        self.reset_parameters()

    def forward(self, x, m, delta):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        all_x_hat_set, all_unc_set = [], []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        all_combi = []

        unc_min = []
        unc_max = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t.type(torch.FloatTensor).cuda(), h[-1].type(torch.FloatTensor).cuda()], 1))  # feature extractor
            # enc_t = self.enc(torch.cat([x[t].type(torch.FloatTensor).cuda(), h[-1].type(torch.FloatTensor).cuda()], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1].type(torch.FloatTensor).cuda())
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t.type(torch.FloatTensor).cuda(), h[-1].type(torch.FloatTensor).cuda()], 1))  # feature extractor
            # dec_t = self.dec(torch.cat([z_t.type(torch.FloatTensor).cuda(), h[-1].type(torch.FloatTensor).cuda()], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            if self.isreparam:
                if self.issampling:
                    # sampled_latent_list = Normal(prior_mean_t, prior_std_t).sample(sample_shape=torch.Size([sampling_freq]))
                    sampling_freq = 5
                    x_hat_t_set = []
                    unc_t_set = []
                    for s in range(sampling_freq):
                        x_repar_t = self.reparameterize(dec_mean_t, dec_std_t)
                        x_hat = (m[:, t, :] * x[t]) + (
                                    (1 - m[:, t, :]) * x_repar_t.clone().contiguous())  # replace x_repar_t
                        unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + (
                                    (1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
                        h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat).type(torch.FloatTensor).cuda(),
                                                                phi_z_t.type(torch.FloatTensor).cuda(),
                                                                m[:, t, :].type(torch.FloatTensor).cuda(),
                                                                unc.type(torch.FloatTensor).cuda(),
                                                                h.type(torch.FloatTensor).cuda())
                        x_hat_t_set.append(x_hat)
                        unc_t_set.append(unc)
                else:
                    x_repar_t = self.reparameterize(dec_mean_t, dec_std_t)
            else:
                x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())  # replace decoder mean
            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())



            # unc = dec_std_t.clone().mul(0.5).exp_()
            # unc = (1e-3 * m[:, t, :] * torch.ones_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            # unc_min.append(torch.min(unc))
            # unc_max.append(torch.max(unc))

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)



            # h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat).type(torch.FloatTensor).cuda(),
            #                                         phi_z_t.type(torch.FloatTensor).cuda(),
            #                                         m[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         unc.type(torch.FloatTensor).cuda(),
            #                                         h.type(torch.FloatTensor).cuda())
            # time decaying + feature extractor
            # h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat).type(torch.FloatTensor).cuda(),
            #                                         phi_z_t.type(torch.FloatTensor).cuda(),
            #                                         m[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         delta[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         unc.type(torch.FloatTensor).cuda(),
            #                                         h.type(torch.FloatTensor).cuda())
            # time decaying
            # h, update_gate, reset_gate = self.gru_u(x_hat.type(torch.FloatTensor).cuda(),
            #                                         z_t.type(torch.FloatTensor).cuda(),
            #                                         m[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         delta[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         unc.type(torch.FloatTensor).cuda(),
            #                                         h.type(torch.FloatTensor).cuda())
            # input decaying (feature extraction x) + hidden decaying
            # h, update_gate, reset_gate = self.gru_u(x[t],
            #                                         dec_mean_t.type(torch.FloatTensor).cuda(),
            #                                         z_t.type(torch.FloatTensor).cuda(),
            #                                         m[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         delta[:, t, :].type(torch.FloatTensor).cuda(),
            #                                         unc.type(torch.FloatTensor).cuda(),
            #                                         h.type(torch.FloatTensor).cuda())
            # input decaying (feature extraction) + hidden decaying
            h, update_gate, reset_gate, combi_t = self.gru_u(x[t],
                                                    dec_mean_t.type(torch.FloatTensor).cuda(),
                                                    phi_z_t.type(torch.FloatTensor).cuda(),
                                                    m[:, t, :].type(torch.FloatTensor).cuda(),
                                                    delta[:, t, :].type(torch.FloatTensor).cuda(),
                                                    unc.type(torch.FloatTensor).cuda(),
                                                    h.type(torch.FloatTensor).cuda())



            # _, h = self.rnn(torch.cat([self.phi_x(phi_x_t), phi_z_t], 1).unsqueeze(0), h)

            all_combi.append(combi_t)

            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)

            if self.issampling:
                all_x_hat_set.append(x_hat_t_set)
                all_unc_set.append(unc_t_set)



        if self.FFA:  # feed-forward attention (FFA)
            sum_h = torch.zeros_like(all_h[0])
            total_t = len(all_h)
            for t in range(total_t):
                e_t = torch.add(self.attention(all_h[t]), self.attention_bias)
                alpha_t = F.softmax(e_t, dim=1)

                sum_h += alpha_t * all_h[t]  # attention alpha
                # sum_h += (1 / total_t) * torch.add(sum_h, all_h[t])  # 1/T

            s = torch.add(self.add_nn(sum_h), self.nn_bias)  # attention
            # s = sum_h  # 1/T

            logits = self.fc_out(s)  # sum_h
        else:  # last hidden state
            logits = self.fc_out(all_h[-1])

        out = torch.sigmoid(logits).clamp(1e-8, 1-1e-8)

        logits = logits[-1].squeeze(1)
        out = out[-1].squeeze(1)
        ############

        vrnn = {}
        vrnn['prior_mean'] = all_prior_mean
        vrnn['prior_std'] = all_prior_std
        vrnn['x'] = all_x
        vrnn['enc_mean'] = all_enc_mean
        vrnn['enc_std'] = all_enc_std
        vrnn['dec_mean'] = all_dec_mean
        vrnn['dec_std'] = all_dec_std
        vrnn['logits'] = logits
        vrnn['out'] = out
        vrnn['x_hat'] = all_x_hat
        vrnn['unc'] = all_unc
        vrnn['update_gate'] = all_update_gate
        vrnn['reset_gate'] = all_reset_gate
        vrnn['combi'] = all_combi

        if self.issampling:
            vrnn['x_hat_set'] = all_x_hat_set
            vrnn['unc_set'] = all_unc_set

        return vrnn

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)
            # nn.init.xavier_uniform(weight)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class IN_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, dropout_p, bias=False, device=None):
        super(IN_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),

            nn.InstanceNorm1d(num_features=x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(x_dim, x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh()
        )
            # nn.ELU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.InstanceNorm1d(num_features=z_dim),
            # nn.BatchNorm1d(num_features=z_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(z_dim, z_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh())
            # nn.LeakyReLU())
            # nn.ELU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z_dim),
            nn.InstanceNorm1d(num_features=z_dim),
            # nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(z_dim, z_dim),
            nn.InstanceNorm1d(num_features=z_dim),
            # nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.InstanceNorm1d(num_features=z_dim),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Linear(z_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            nn.InstanceNorm1d(num_features=x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(x_dim, x_dim),
            nn.InstanceNorm1d(num_features=x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        self.dec_std = nn.Linear(x_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)  # GRU-U
        # self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers, bias)  # Vanilla GRU

        # attention
        self.attention = nn.Sequential(
            nn.Linear(h_dim, 1, bias=False),
            nn.Tanh())
        self.attention_bias = nn.Parameter(torch.zeros(1))

        # classifier
        self.fc_out = nn.Linear(h_dim, 1)
        # self.fc_out = nn.Linear(h_dim, 2)
        # self.fc_out = nn.Sequential(
        # 	nn.Linear(h_dim, 2),
        # 	nn.Dropout(p=0.3))

        self.reset_parameters()

    def instance_norm(self, x, gamma=None, beta=None):
        mean = F.mean(x, axis=-1)
        mean = F.mean(mean, axis=-1)
        mean = F.broadcast_to(mean[Ellipsis, None, None], x.shape)
        var = F.squared_difference(x, mean)
        std = F.sqrt(var + 1e-5)
        x_hat = (x - mean) / std
        if gamma is not None:
            gamma = F.broadcast_to(gamma[None, Ellipsis, None, None], x.shape)
            beta = F.broadcast_to(beta[None, Ellipsis, None, None], x.shape)
            return gamma * x_hat + beta
        else:
            return x_hat

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        unc_min = []
        unc_max = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            x_repar_t = self.reparameterize(dec_mean_t, dec_std_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())  # replace decoder mean
            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * x_repar_t.clone().contiguous())  # replace x_repar_t

            # unc = dec_std_t.clone().mul(0.5).exp_()
            # unc = (1e-3 * m[:, t, :] * torch.ones_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            # unc_min.append(torch.min(unc))
            # unc_max.append(torch.max(unc))

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)

            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)
            # _, h = self.rnn(torch.cat([self.phi_x(phi_x_t), phi_z_t], 1).unsqueeze(0), h)

            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)


        # attention
        sum_h = torch.zeros_like(all_h[0])
        total_t = len(all_h)
        for t in range(total_t):
            e_t = torch.add(self.attention(all_h[t]), self.attention_bias)
            alpha_t = F.softmax(e_t, dim=1)

            # sum_h += alpha_t * all_h[t]  # attention alpha
            sum_h += (1 / total_t) * torch.add(sum_h, all_h[t])  # 1/T

        out = self.fc_out(all_h[-1])  # last hidden state
        # out = self.fc_out(sum_h)  # sum_h

        # out_prob = torch.sigmoid(out).squeeze(0)  # (Fx2)
        out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)
        # print("logit is", out_prob)
        # out_prob = out[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class Sampling_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, dropout_p, bias=False, device=None):
        super(Sampling_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(x_dim, x_dim),
            # nn.BatchNorm1d(num_features=x_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh())
            # nn.ELU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            # nn.BatchNorm1d(num_features=z_dim),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.Linear(z_dim, z_dim),
            # nn.Dropout(p=0.2),
            nn.Tanh())
            # nn.LeakyReLU())
            # nn.ELU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Linear(z_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.Dropout(p=dropout_p),
            # nn.ReLU())
            # nn.ELU())
            nn.Tanh())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        self.dec_std = nn.Linear(x_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)  # GRU-U
        # self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers, bias)  # Vanilla GRU

        # attention
        self.attention = nn.Sequential(
            nn.Linear(h_dim, 1, bias=False),
            nn.Tanh())
        self.attention_bias = nn.Parameter(torch.zeros(1))

        # classifier
        self.fc_out = nn.Linear(h_dim, 1)
        # self.fc_out = nn.Linear(h_dim, 2)
        # self.fc_out = nn.Sequential(
        # 	nn.Linear(h_dim, 2),
        # 	nn.Dropout(p=0.3))

        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        sampling_freq = 10
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)



            sampled_latent_list = Normal(prior_mean_t, prior_std_t).sample(sample_shape=torch.Size([sampling_freq]))

            for s in range(sampling_freq):
                dec = self.dec(torch.cat([sampled_latent_list[s], h[-1]], 1))
                dec_mean = self.dec_mean(dec)
                dec_std = self.dec_std(dec)
                log_likelihood = Normal(dec_mean, dec_std.mul(0.5).exp_()).log_prob(x[t]).sum()
                # print(log_likelihood)

                if s == 0:
                    best_log_likelihood = log_likelihood

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood

                    prior_t = sampled_latent_list[s]
                    prior_mean_t = self.prior_mean(prior_t)
                    prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)

            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            # x_repar_t = self.reparameterize(dec_mean_t, dec_std_t)


            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())  # replace decoder mean
            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * x_repar_t.clone().contiguous())  # replace x_repar_t

            # unc = dec_std_t.clone().mul(0.5).exp_()
            # unc = (1e-3 * m[:, t, :] * torch.ones_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            # unc_min.append(torch.min(unc))
            # unc_max.append(torch.max(unc))

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)

            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)
            # _, h = self.rnn(torch.cat([self.phi_x(phi_x_t), phi_z_t], 1).unsqueeze(0), h)

            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)


        # attention
        sum_h = torch.zeros_like(all_h[0])
        total_t = len(all_h)
        for t in range(total_t):
            e_t = torch.add(self.attention(all_h[t]), self.attention_bias)
            alpha_t = F.softmax(e_t, dim=1)

            # sum_h += alpha_t * all_h[t]  # attention alpha
            sum_h += (1 / total_t) * torch.add(sum_h, all_h[t])  # 1/T

        out = self.fc_out(all_h[-1])  # last hidden state
        # out = self.fc_out(sum_h)  # sum_h

        # out_prob = torch.sigmoid(out).squeeze(0)  # (Fx2)
        # out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = torch.sigmoid(out).clamp(1e-8, 1-1e-8)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)
        # print("logit is", out_prob)
        # out_prob = out[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class SELU_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, device=None):
        super(SELU_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.SELU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.SELU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + h_dim, z_dim),
            nn.SELU(),
            nn.Linear(z_dim, z_dim),
            nn.SELU())
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.SELU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Linear(z_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            nn.SELU(),
            nn.Linear(x_dim, x_dim),
            nn.SELU())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        self.dec_std = nn.Linear(x_dim, x_dim)

        # gru
        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)  # GRU-U
        # self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers, bias)  # Vanilla GRU

        # attention
        self.attention = nn.Sequential(
            nn.Linear(h_dim, 1, bias=False),
            nn.Tanh())
        self.attention_bias = nn.Parameter(torch.zeros(1))

        # classifier
        self.fc_out = nn.Linear(h_dim, 1)

        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_update_gate, all_reset_gate = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        unc_min = []
        unc_max = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            x_repar_t = self.reparameterize(dec_mean_t, dec_std_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())  # replace decoder mean
            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * x_repar_t.clone().contiguous())  # replace x_repar_t

            # unc = dec_std_t.clone().mul(0.5).exp_()
            # unc = (1e-3 * m[:, t, :] * torch.ones_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())
            # unc_min.append(torch.min(unc))
            # unc_max.append(torch.max(unc))

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)

            h, update_gate, reset_gate = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)
            # _, h = self.rnn(torch.cat([self.phi_x(phi_x_t), phi_z_t], 1).unsqueeze(0), h)

            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

            all_update_gate.append(update_gate)
            all_reset_gate.append(reset_gate)


        # attention
        sum_h = torch.zeros_like(all_h[0])
        total_t = len(all_h)
        for t in range(total_t):
            e_t = torch.add(self.attention(all_h[t]), self.attention_bias)
            alpha_t = F.softmax(e_t, dim=1)

            # sum_h += alpha_t * all_h[t]  # attention alpha
            sum_h += (1 / total_t) * torch.add(sum_h, all_h[t])  # 1/T

        out = self.fc_out(all_h[-1])  # last hidden state
        # out = self.fc_out(sum_h)  # sum_h

        # out_prob = torch.sigmoid(out).squeeze(0)  # (Fx2)
        out_prob = torch.sigmoid(out).clamp(0, 1)  # (Fx1)
        out_prob = out_prob[-1].squeeze(1)
        # print("logit is", out_prob)
        # out_prob = out[-1].squeeze(1)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc), (all_update_gate, all_reset_gate)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # Re-parameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu).add_(1e-6)
        return z

class Batchnorm_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
        super(Batchnorm_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU())
        # self.phi_x = nn.Sequential(
        # 	nn.Linear(x_dim, h_dim),
        # 	nn.ReLU(),
        # 	nn.Linear(h_dim, h_dim),
        # 	nn.ReLU())
        # self.phi_z = nn.Sequential(
        # 	nn.Linear(z_dim, h_dim),
        # 	nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            # nn.Linear(h_dim + h_dim, z_dim),
            nn.Linear(x_dim + h_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU())  ###################################################
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            # nn.Linear(h_dim + h_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())  #####################################################
        self.dec_std = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        # self.dec_mean = nn.Sequential(
        # 	nn.Linear(x_dim, x_dim),
        # 	nn.Sigmoid())  ####################################

        # recurrence
        # self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        # self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)

        self.fc_out = nn.Linear(h_dim, 2)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

        out = self.fc_out(all_h[-1])
        out_prob = torch.sigmoid(out).squeeze(0)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


class Layered_VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z1_dim, z2_dim, z3_dim, n_layers, bias=False, device=None):
        super(Layered_VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z3_dim = z3_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z3_dim, z3_dim),
            nn.BatchNorm1d(num_features=z3_dim),
            nn.ReLU(),
            nn.Linear(z3_dim, z3_dim),
            nn.BatchNorm1d(num_features=z3_dim),
            nn.ReLU())

        # self.phi_x = nn.Sequential(
        # 	nn.Linear(x_dim, h_dim),
        # 	nn.ReLU(),
        # 	nn.Linear(h_dim, h_dim),
        # 	nn.ReLU())
        # self.phi_z = nn.Sequential(
        # 	nn.Linear(z_dim, h_dim),
        # 	nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            # nn.Linear(h_dim + h_dim, z_dim),
            nn.Linear(x_dim + h_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.ReLU(),
            nn.Linear(z1_dim, z2_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.ReLU())  ###################################################
        self.enc_mean = nn.Linear(z2_dim, z3_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(z2_dim, z3_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z3_dim),
            nn.BatchNorm1d(num_features=z3_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z3_dim, z3_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(z3_dim, z3_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z3_dim + h_dim, z2_dim),
            # nn.Linear(h_dim + h_dim, x_dim),
            nn.BatchNorm1d(num_features=z2_dim),
            nn.ReLU(),
            nn.Linear(z2_dim, z1_dim),
            nn.BatchNorm1d(num_features=z1_dim),
            nn.ReLU())  #####################################################
        self.dec_std = nn.Sequential(
            nn.Linear(z1_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(z1_dim, x_dim)
        # self.dec_mean = nn.Sequential(
        # 	nn.Linear(x_dim, x_dim),
        # 	nn.Sigmoid())  ####################################

        # recurrence
        # self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        # self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.gru_u = GRUU(x_dim, z3_dim, h_dim, 2)

        self.fc_out = nn.Linear(h_dim, 2)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

        out = self.fc_out(all_h[-1])
        out_prob = torch.sigmoid(out).squeeze(0)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

class VRNN_GRUU_tanh(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
        super(VRNN_GRUU_tanh, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Tanh(),
            nn.Linear(x_dim, x_dim),
            nn.Tanh())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh())
        # self.phi_x = nn.Sequential(
        # 	nn.Linear(x_dim, h_dim),
        # 	nn.ReLU(),
        # 	nn.Linear(h_dim, h_dim),
        # 	nn.ReLU())
        # self.phi_z = nn.Sequential(
        # 	nn.Linear(z_dim, h_dim),
        # 	nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            # nn.Linear(h_dim + h_dim, z_dim),
            nn.Linear(x_dim + h_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU())  ###################################################
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            # nn.Linear(h_dim + h_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())  #####################################################
        self.dec_std = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        # self.dec_mean = nn.Sequential(
        # 	nn.Linear(x_dim, x_dim),
        # 	nn.Sigmoid())  ####################################

        # recurrence
        # self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        # self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)

        self.fc_out = nn.Linear(h_dim, 2)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

        out = self.fc_out(all_h[-1])
        out_prob = torch.sigmoid(out).squeeze(0)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

class VRNN_GRUU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
        super(VRNN_GRUU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU())
        # self.phi_x = nn.Sequential(
        # 	nn.Linear(x_dim, h_dim),
        # 	nn.ReLU(),
        # 	nn.Linear(h_dim, h_dim),
        # 	nn.ReLU())
        # self.phi_z = nn.Sequential(
        # 	nn.Linear(z_dim, h_dim),
        # 	nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            # nn.Linear(h_dim + h_dim, z_dim),
            nn.Linear(x_dim + h_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU())  ###################################################
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + h_dim, x_dim),
            # nn.Linear(h_dim + h_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU())  #####################################################
        self.dec_std = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(x_dim, x_dim)
        # self.dec_mean = nn.Sequential(
        # 	nn.Linear(x_dim, x_dim),
        # 	nn.Sigmoid())  ####################################

        # recurrence
        # self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        # self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.gru_u = GRUU(x_dim, z_dim, h_dim, 2)

        self.fc_out = nn.Linear(h_dim, 2)
        self.reset_parameters()

    def forward(self, x, m):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_x = []

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        all_h = []
        all_x_hat = []
        all_unc = []

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * phi_x_t.clone().contiguous())
            x_hat = (m[:, t, :] * x[t]) + ((1 - m[:, t, :]) * dec_mean_t.clone().contiguous())
            unc = (m[:, t, :] * torch.zeros_like(dec_std_t).to(self.device)) + ((1 - m[:, t, :]) * dec_std_t.clone().mul(0.5).exp_())

            # recurrence
            # _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # h = self.gru_u(torch.cat([phi_x_t, phi_z_t], 1), m[:,t,:], dec_std_t, h)

            # h = self.gru_u(phi_x_t, phi_z_t, m[:, t, :], dec_std_t, h)
            # h = self.gru_u(x_hat, phi_z_t, m[:, t, :], unc, h)
            h = self.gru_u(self.phi_x(x_hat), phi_z_t, m[:, t, :], unc, h)


            all_h.append(h)

            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_x.append(x[t])

            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            all_x_hat.append(x_hat)
            all_unc.append(unc)

        out = self.fc_out(all_h[-1])
        out_prob = torch.sigmoid(out).squeeze(0)

        return (all_prior_mean, all_prior_std, all_x), \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std), out_prob,\
               (all_x_hat, all_unc)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        #         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class GRUU(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, out_ch, isdecaying):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.

        Implemented based on the paper:
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }

        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUU, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.isdecaying = isdecaying
        self.out_ch = out_ch
        self.featurecorr = FeatureRegression(input_size)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size)
            # self.identity = torch.eye(input_size).cuda()
            self.zeros_x = Variable(torch.zeros(input_size).cuda(), requires_grad=False)
            self.zeros_h = Variable(torch.zeros(hidden_size).cuda(), requires_grad=False)
            # self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros_x = Variable(torch.zeros(input_size))
            self.zeros_h = Variable(torch.zeros(hidden_size))
            # self.X_mean = Variable(torch.Tensor(X_mean))

        self.phi_x = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh())


        self.zl = nn.Linear(input_size + input_size + latent_size + hidden_size + self.mask_size, hidden_size, bias=True)
        self.rl = nn.Linear(input_size + input_size + latent_size + hidden_size + self.mask_size, hidden_size, bias=True)
        # self.hl = nn.Linear(input_size + input_size + latent_size + hidden_size + self.mask_size, hidden_size, bias=True)
        self.hl = nn.Sequential(nn.Linear(input_size + input_size + latent_size + hidden_size + self.mask_size, hidden_size, bias=True),
                                nn.Tanh())

        ######## input decaying
        # self.gamma_x_l = nn.Sequential(FilterLinear(self.delta_size, self.delta_size, self.identity),
        #                                nn.ReLU()) # independent gamma_u
        self.gamma_x_l = nn.Linear(self.delta_size, self.delta_size, bias=True)


        ######## uncertainty decaying
        self.gamma_u_l = FilterLinear(self.delta_size, self.delta_size, self.identity)  # independent gamma_u
        # self.gamma_u_l = nn.Linear(self.delta_size, self.delta_size, bias=True)  # dependent gamma_u

        ######## hidden decaying
        # self.gamma_h_l = nn.Sequential(nn.Linear(self.delta_size, self.hidden_size, bias=True),
        #                                nn.ReLU())    # independent gamma_u
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size, bias=True)  # independent gamma_u

        self.conv1d = nn.Conv1d(2, self.out_ch, 1)
        self.pooling = nn.MaxPool2d((self.out_ch, 1))

        self.beta_x = nn.Sequential(nn.Linear(self.delta_size+self.delta_size, self.delta_size, bias=True),
                                    nn.ReLU())

        self.W_x = nn.Linear(self.delta_size, self.delta_size, bias=True)
        self.U_x = nn.Linear(self.delta_size, self.delta_size, bias=True)

        self.fc_out = nn.Linear(self.hidden_size, output_size)
        # self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def step(self, orig_x, x, z, h, m, delta, u):

        batch_size = x.shape[0]
        dim_size = x.shape[1]

        h = h.squeeze(0)
        gamma_u = torch.exp(-torch.max(self.zeros_x, self.gamma_u_l(u)))  # input decaying factor by uncertainty

        if self.isdecaying == 'both_decay':
            gamma_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
            gamma_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta)))  # hidden decaying factor by time gap

            beta = self.beta_x(torch.cat((gamma_x, m), 1))
            c = beta * orig_x + (1-beta) * x
            h = h * gamma_h

        elif self.isdecaying == 'input_decay':
            gamma_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
            beta = self.beta_x(torch.cat((gamma_x, m), 1))
            c = beta * orig_x + (1 - beta) * x

        elif self.isdecaying == 'hidden_decay':
            c = x
            gamma_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta)))  # hidden decaying factor by time gap
            h = h * gamma_h
        elif self.isdecaying == 'none':
            c = x
        elif self.isdecaying == 'fr_1':
            gamma_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
            beta = self.beta_x(torch.cat((gamma_x, m), 1))
            x_f = self.featurecorr(x)
            c = beta * x_f + (1 - beta) * x
        elif self.isdecaying == 'fr_2':
            gamma_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
            beta = self.beta_x(torch.cat((gamma_x, m), 1))
            x_f = self.featurecorr(x)
            c_t = beta * orig_x + (1 - beta) * x
            x_f_c = torch.cat((x_f.unsqueeze(1), c_t.unsqueeze(1)), 1)
            x_f_c = self.conv1d(x_f_c)
            c = self.pooling(x_f_c)
            c = c.squeeze(1)
        elif self.isdecaying == 'fr_3':
            gamma_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(delta)))
            beta = self.beta_x(torch.cat((gamma_x, m), 1))
            x_f = self.featurecorr(x)
            c_t = beta * orig_x + (1 - beta) * x
            x_f_c = torch.cat((x_f.unsqueeze(1), c_t.unsqueeze(1)), 1)
            x_f_c = self.conv1d(x_f_c)
            x_f_c_pooled = self.pooling(x_f_c)
            c = m * orig_x + (1 - m) * x_f_c_pooled.squeeze(1)
        else:
            pass

        combi_t = c.clone()

        c = torch.cat((self.phi_x(c), gamma_u, z), 1)
        combined = torch.cat((c, h, m), 1)

        update_gate = torch.sigmoid(self.zl(combined)).clamp(-1+1e-8, 1-1e-8)
        reset_gate = torch.sigmoid(self.rl(combined)).clamp(-1+1e-8, 1-1e-8)

        combined_r = torch.cat((c, reset_gate * h, m), 1)
        # h_tilde = torch.tanh(self.hl(combined_r))
        h_tilde = self.hl(combined_r)


        h = (1 - update_gate) * h + update_gate * h_tilde
        # h = gamma_h * h
        return h.unsqueeze(0), update_gate, reset_gate, combi_t

    def forward(self, orig_input, input, latent, mask, delta, unc, Hidden_State):
        Hidden_State, update_gate, reset_gate, combi_t = self.step(orig_input, input, latent, Hidden_State, mask, delta, unc)
        return Hidden_State, update_gate, reset_gate, combi_t

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State

# class VRNN(nn.Module):
# 	def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
# 		super(VRNN, self).__init__()
#
# 		self.x_dim = x_dim
# 		self.h_dim = h_dim
# 		self.z_dim = z_dim
# 		self.n_layers = n_layers
# 		self.device = device
#
# 		# feature-extracting transformations
# 		self.phi_x = nn.Sequential(
# 			nn.Linear(x_dim, x_dim),
# 			nn.ReLU(),
# 			nn.Linear(x_dim, x_dim),
# 			nn.ReLU())
# 		self.phi_z = nn.Sequential(
# 			nn.Linear(z_dim, z_dim),
# 			nn.ReLU())
# 		# self.phi_x = nn.Sequential(
# 		# 	nn.Linear(x_dim, h_dim),
# 		# 	nn.ReLU(),
# 		# 	nn.Linear(h_dim, h_dim),
# 		# 	nn.ReLU())
# 		# self.phi_z = nn.Sequential(
# 		# 	nn.Linear(z_dim, h_dim),
# 		# 	nn.ReLU())
#
# 		# encoder
# 		self.enc = nn.Sequential(
# 			# nn.Linear(h_dim + h_dim, z_dim),
# 			nn.Linear(x_dim + h_dim, z_dim),
# 			nn.ReLU(),
# 			nn.Linear(z_dim, z_dim),
# 			nn.ReLU())  ###################################################
# 		self.enc_mean = nn.Linear(z_dim, z_dim)
# 		self.enc_std = nn.Sequential(
# 			nn.Linear(z_dim, z_dim),
# 			nn.Softplus())
#
# 		# prior
# 		self.prior = nn.Sequential(
# 			nn.Linear(h_dim, z_dim),
# 			nn.ReLU())
# 		self.prior_mean = nn.Linear(z_dim, z_dim)
# 		self.prior_std = nn.Sequential(
# 			nn.Linear(z_dim, z_dim),
# 			nn.Softplus())
#
# 		# decoder
# 		self.dec = nn.Sequential(
# 			nn.Linear(z_dim + h_dim, x_dim),
# 			# nn.Linear(h_dim + h_dim, x_dim),
# 			nn.ReLU(),
# 			nn.Linear(x_dim, x_dim),
# 			nn.ReLU())  #####################################################
# 		self.dec_std = nn.Sequential(
# 			nn.Linear(x_dim, x_dim),
# 			nn.Softplus())
# 		self.dec_mean = nn.Linear(x_dim, x_dim)
# 		# self.dec_mean = nn.Sequential(
# 		# 	nn.Linear(x_dim, x_dim),
# 		# 	nn.Sigmoid())  ####################################
#
# 		# recurrence
# 		self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers, bias)
# 		# self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
# 		self.fc_out = nn.Linear(h_dim, 2)
#
# 		self.reset_parameters()
#
# 	def forward(self, x):
#
# 		all_enc_mean, all_enc_std = [], []
# 		all_dec_mean, all_dec_std = [], []
# 		all_prior_mean, all_prior_std = [], []
# 		all_x = []
#
# 		h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
# 		all_h = []
# 		for t in range(x.size(0)):
# 			phi_x_t = self.phi_x(x[t])
#
# 			# encoder
# 			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
# 			enc_mean_t = self.enc_mean(enc_t)
# 			enc_std_t = self.enc_std(enc_t)
#
# 			# prior
# 			prior_t = self.prior(h[-1])
# 			prior_mean_t = self.prior_mean(prior_t)
# 			prior_std_t = self.prior_std(prior_t)
#
# 			# sampling and reparameterization
# 			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
# 			phi_z_t = self.phi_z(z_t)
#
# 			# decoder
# 			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
# 			dec_mean_t = self.dec_mean(dec_t)
# 			dec_std_t = self.dec_std(dec_t)
#
# 			# recurrence
# 			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
#
# 			all_h.append(h)
#
# 			all_enc_mean.append(enc_mean_t)
# 			all_enc_std.append(enc_std_t)
#
# 			all_prior_mean.append(prior_mean_t)
# 			all_prior_std.append(prior_std_t)
# 			all_x.append(x[t])
#
# 			all_dec_mean.append(dec_mean_t)
# 			all_dec_std.append(dec_std_t)
#
# 		out = self.fc_out(all_h[-1])
# 		out_prob = torch.sigmoid(out).squeeze(0)
#
# 		return (all_prior_mean, all_prior_std, all_x), \
# 			   (all_enc_mean, all_enc_std), \
# 			   (all_dec_mean, all_dec_std), out_prob
#
# 	def sample(self, seq_len):
#
# 		sample = torch.zeros(seq_len, self.x_dim)
#
# 		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
# 		for t in range(seq_len):
# 			# prior
# 			prior_t = self.prior(h[-1])
# 			prior_mean_t = self.prior_mean(prior_t)
# 			prior_std_t = self.prior_std(prior_t)
#
# 			# sampling and reparameterization
# 			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
# 			phi_z_t = self.phi_z(z_t)
#
# 			# decoder
# 			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
# 			dec_mean_t = self.dec_mean(dec_t)
# 			# dec_std_t = self.dec_std(dec_t)
#
# 			phi_x_t = self.phi_x(dec_mean_t)
#
# 			# recurrence
# 			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
#
# 			sample[t] = dec_mean_t.data
#
# 		return sample
#
# 	def reset_parameters(self):
# 		for weight in self.parameters():
# 			if len(weight.size()) == 1:
# 				continue
# 			stv = 1. / math.sqrt(weight.size(1))
# 			nn.init.uniform_(weight, -stv, stv)
#
# 	def _init_weights(self, stdv):
# 		pass
#
# 	def _reparameterized_sample(self, mean, std):
# 		"""using std to sample"""
# 		eps = torch.FloatTensor(std.size()).normal_().to(self.device)
# 		eps = Variable(eps)
# 		return eps.mul(std).add_(mean)


# class VRNN(nn.Module):
# 	def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, device=None):
# 		super(VRNN, self).__init__()
#
# 		self.x_dim = x_dim
# 		self.h_dim = h_dim
# 		self.z_dim = z_dim
# 		self.n_layers = n_layers
# 		self.device = device
#
# 		# feature-extracting transformations
# 		self.phi_x = nn.Sequential(
# 			nn.Linear(x_dim, h_dim),
# 			nn.ReLU(),
# 			nn.Linear(h_dim, h_dim),
# 			nn.ReLU()) #####################################################
# 		self.phi_z = nn.Sequential(
# 			nn.Linear(z_dim, h_dim),
# 			nn.ReLU()) #####################################################
#
# 		# encoder
# 		self.enc = nn.Sequential(
# 			nn.Linear(h_dim + h_dim, h_dim),
# 			nn.ReLU(),
# 			nn.Linear(h_dim, h_dim),
# 			nn.ReLU()) ###################################################
# 		self.enc_mean = nn.Linear(h_dim, z_dim)
# 		self.enc_std = nn.Sequential(
# 			nn.Linear(h_dim, z_dim),
# 			nn.Softplus())
#
# 		# prior
# 		self.prior = nn.Sequential(
# 			nn.Linear(h_dim, h_dim),
# 			nn.ReLU()) ###################################################
# 		self.prior_mean = nn.Linear(h_dim, z_dim)
# 		self.prior_std = nn.Sequential(
# 			nn.Linear(h_dim, z_dim),
# 			nn.Softplus())
#
# 		# decoder
# 		self.dec = nn.Sequential(
# 			nn.Linear(h_dim + h_dim, h_dim),
# 			nn.ReLU(),
# 			nn.Linear(h_dim, h_dim),
# 			nn.ReLU()) #####################################################
# 		self.dec_std = nn.Sequential(
# 			nn.Linear(h_dim, x_dim),
# 			nn.Softplus())
# 		#self.dec_mean = nn.Linear(h_dim, x_dim)
# 		self.dec_mean = nn.Sequential(
# 			nn.Linear(h_dim, x_dim),
# 			nn.Sigmoid()) ####################################
#
# 		#recurrence
# 		self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
#
# 		self.fc_out = nn.Linear(h_dim, 2)
#
#
# 	def forward(self, x):
#
# 		all_enc_mean, all_enc_std = [], []
# 		all_dec_mean, all_dec_std = [], []
# 		kld_loss = 0
# 		nll_loss = 0
#
# 		h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
# 		all_h = []
# 		for t in range(x.size(0)):
#
# 			phi_x_t = self.phi_x(x[t])
#
# 			#encoder
# 			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
# 			enc_mean_t = self.enc_mean(enc_t)
# 			enc_std_t = self.enc_std(enc_t)
#
# 			#prior
# 			prior_t = self.prior(h[-1])
# 			prior_mean_t = self.prior_mean(prior_t)
# 			prior_std_t = self.prior_std(prior_t)
#
# 			#sampling and reparameterization
# 			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
# 			phi_z_t = self.phi_z(z_t)
#
# 			#decoder
# 			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
# 			dec_mean_t = self.dec_mean(dec_t)
# 			dec_std_t = self.dec_std(dec_t)
#
# 			#recurrence
# 			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
#
# 			all_h.append(h)
#
# 			#computing losses
# 			kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
# 			nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
# 			# nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
#
# 			all_enc_std.append(enc_std_t)
# 			all_enc_mean.append(enc_mean_t)
# 			all_dec_mean.append(dec_mean_t)
# 			all_dec_std.append(dec_std_t)
#
# 		out = self.fc_out(all_h[-1])
# 		out_prob = torch.sigmoid(out).squeeze(0)
#
# 		return kld_loss, nll_loss, \
# 			(all_enc_mean, all_enc_std), \
# 			(all_dec_mean, all_dec_std), out_prob
#
#
# 	def sample(self, seq_len):
#
# 		sample = torch.zeros(seq_len, self.x_dim)
#
# 		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to(self.device)
# 		for t in range(seq_len):
#
# 			#prior
# 			prior_t = self.prior(h[-1])
# 			prior_mean_t = self.prior_mean(prior_t)
# 			prior_std_t = self.prior_std(prior_t)
#
# 			#sampling and reparameterization
# 			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
# 			phi_z_t = self.phi_z(z_t)
#
# 			#decoder
# 			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
# 			dec_mean_t = self.dec_mean(dec_t)
# 			#dec_std_t = self.dec_std(dec_t)
#
# 			phi_x_t = self.phi_x(dec_mean_t)
#
# 			#recurrence
# 			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
#
# 			sample[t] = dec_mean_t.data
#
# 		return sample
#
#
# 	def reset_parameters(self, stdv=1e-1):
# 		for weight in self.parameters():
# 			weight.data.normal_(0, stdv)
#
#
# 	def _init_weights(self, stdv):
# 		pass
#
#
# 	def _reparameterized_sample(self, mean, std):
# 		"""using std to sample"""
# 		eps = torch.FloatTensor(std.size()).normal_().to(self.device)
# 		eps = Variable(eps)
# 		return eps.mul(std).add_(mean)
#
#
# 	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
# 		"""Using std to compute KLD"""
#
# 		kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
# 			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
# 			std_2.pow(2) - 1)
# 		return 0.5 * torch.sum(kld_element)
#
#
# 	def _nll_bernoulli(self, theta, x):
# 		return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))
#
#
# 	def _nll_gauss(self, mean, std, x):
#
# 		ss = std.pow(2)
# 		norm = x-mean
# 		z = torch.div(norm.pow(2), ss)
# 		denom_log = torch.log(2*np.pi*ss)
#
# 		result = torch.sum(z + denom_log)
# 		# result = -Normal(mean, std.mul(0.5).exp_()).log_prob(x).sum(1)
#
# 		return torch.sum(result) # pass


# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device=None, is_gating=False, emb_size=None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.is_gating = is_gating

        if is_gating == False:
            emb_size = input_size

        self.rnn = nn.GRU(emb_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True,
                          bidirectional=False)

        self.fc_output = nn.Linear(hidden_size, num_classes, bias=False)

        self.gatex = nn.Linear(input_size, emb_size)
        self.gater = nn.Linear(input_size, emb_size)
        self.gatei = nn.Linear(input_size, emb_size)

        # for layer_p in self.rnn._all_weights:
        #     for p in layer_p:
        #         if 'weight' in p:
        #             init.uniform_(self.rnn.__getattr__(p), -0.05, 0.05)
        #
        # nn.init.uniform_(self.fc_output.weight, -0.05, 0.05)


    def unc_aware_attention(self, x_hat, unc):
        x_tilde = []
        for t in range(x_hat.shape[1]):
            x_hat_t = x_hat[:, t, :] * F.softmax(-unc[:, t, :], 1)
            x_hat_t = self.batchnorm(x_hat_t).unsqueeze(1)
            if t == 0:
                x_tilde = x_hat_t
            else:
                x_tilde = torch.cat([x_tilde, x_hat_t], 1)

        return x_tilde

    def unc_aware_attention_gating(self, x_hat, unc):
        x_tilde = []
        for t in range(x_hat.shape[1]):
            wx = self.gatex(x_hat[:,t,:])
            wr = torch.sigmoid(self.gater(unc[:,t,:]))
            wi = torch.sigmoid(self.gatei(unc[:,t,:]))
            x_hat_t = torch.tanh((wx * wr) + (wx * wi))
            if t == 0:
                x_tilde = x_hat_t.unsqueeze(1)
            else:
                x_tilde = torch.cat([x_tilde, x_hat_t.unsqueeze(1)], 1)

        return x_tilde

    def forward(self, input, hidden=None, is_uncertain=False, uncertainty=[]):
        if is_uncertain:
            if self.is_gating:
                emb = self.unc_aware_attention_gating(input, uncertainty)  # s x ch x t x v
            else:
                emb = self.unc_aware_attention(input, uncertainty)  # s x ch x t x v
        else:
            emb = input  # s x t x v

        # Get Dimensionality
        s, t, v = emb.size()

        # RNN Forwarding and get the last output only as the prediction result
        g_out = self.rnn(emb)
        o = self.fc_output(g_out[0].contiguous().view(-1, self.hidden_size))
        out = F.softmax(o, 1).reshape(s, t, self.num_classes)[:,-1,:]

        return out

# VAE for MIMIC3
class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, device):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(z_dim, z_dim)
        self.enc_std = nn.Linear(z_dim, z_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU())
        self.dec_std = nn.Linear(x_dim, x_dim)
        self.dec_mean = nn.Linear(x_dim, x_dim)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        # Encoding
        encoder = self.enc(x)
        enc_mean = self.enc_mean(encoder)
        enc_std = self.enc_std(encoder)
        z = self.reparameterize(enc_mean, enc_std)

        # Decoding
        decoder = self.dec(z)
        dec_mean = self.dec_mean(decoder)
        dec_std = self.dec_std(decoder)

        return z, enc_mean, enc_std, dec_mean, dec_std


class BiGRU(nn.Module):
    def __init__(self, x_dim, h_dim, n_layers, dropout_p, device):
        super(BiGRU, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        self.build()

        self.fc_out = nn.Linear(self.h_dim, 1)

    def build(self):
        self.GRU_fwd = nn.GRU(self.x_dim, self.h_dim, self.n_layers, bias=True, batch_first=True, bidirectional=False)
        self.GRU_bwd = nn.GRU(self.x_dim, self.h_dim, self.n_layers, bias=True, batch_first=True, bidirectional=False)

    def forward(self, x_fwd, x_bwd):
        self.hidden_fwd = Variable(torch.zeros(self.n_layers, x_fwd.size(0), self.h_dim)).to(self.device)
        self.hidden_bwd = Variable(torch.zeros(self.n_layers, x_bwd.size(0), self.h_dim)).to(self.device)

        gru_f_o, _ = self.GRU_fwd(x_fwd, self.hidden_fwd)
        gru_b_o, _ = self.GRU_bwd(x_bwd, self.hidden_bwd)

        logits_f = self.fc_out(gru_f_o.permute(1, 0, 2)[-1])
        logits_b = self.fc_out(gru_b_o.permute(1, 0, 2)[-1])

        out_f = torch.sigmoid(logits_f).clamp(1e-8, 1-1e-8)
        out_b = torch.sigmoid(logits_b).clamp(1e-8, 1-1e-8)

        logits_f = logits_f.squeeze(1)
        out_f = out_f.squeeze(1)

        logits_b = logits_b.squeeze(1)
        out_b = out_b.squeeze(1)

        return logits_f, out_f, logits_b, out_b