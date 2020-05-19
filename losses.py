import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class WeightedBCE(nn.Module):
    def __init__(self,  device):
        super(WeightedBCE, self).__init__()
        self.device = device

    def forward(self, model, inputs, targets):
        pos_num = len(np.where(targets == 1)[0])
        neg_num = len(np.where(targets == 0)[0])
        if pos_num == 0:
            pos_weight = 1.0
        else:
            pos_weight = neg_num / pos_num
        weights = torch.zeros(len(targets))

        for i in range(len(targets)):
            if i == 1:
                weights[i] = pos_weight
            else:
                weights[i] = 1.0

        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weights)

        # neg_weight = len(np.where(targets == 0)[0]) / (len(np.where(targets == 1)[0]) + len(np.where(targets == 0)[0]))
        # loss = torch.mean(-(pos_weight * (targets * torch.log(inputs)) + neg_weight * ((1-targets) * torch.log(1-inputs))))
        return loss


class VRNNLoss(nn.Module):
    def __init__(self,  lambda1, device, isreconmsk=True):
        super(VRNNLoss, self).__init__()
        self.lambda1 = torch.tensor(lambda1).to(device)
        self.device = device
        self.isreconmsk = isreconmsk

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (std_2 - std_1 +
                       (torch.exp(std_1) + (mean_1 - mean_2).pow(2)) / torch.exp(std_2) - 1)
        # kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
        #                (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        #                std_2.pow(2) - 1)

        return 0.5 * torch.sum(kld_element, 1)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        ss = std.pow(2)
        norm = x.sub(mean)
        z = torch.div(norm.pow(2), ss)
        denom_log = torch.log(2 * np.pi * ss)

        # result = 0.5 * torch.sum(z + denom_log)
        result = 0.5 * torch.sum(z + denom_log, 1)
        # result = -Normal(mean, std.mul(0.5).exp_()).log_prob(x).sum(1)

        return result  # pass

    def forward(self, model, all_prior_mean, all_prior_std, all_x, all_enc_mean, all_enc_std,
                all_dec_mean, all_dec_std, msk, eval_x, eval_msk, beta=1):

        kld_loss, nll_loss, mae_loss = 0, 0, 0
        nll_loss_2 = 0

        for t in range(len(all_x)):
            kld_loss += beta * self._kld_gauss(all_enc_mean[t], all_enc_std[t], all_prior_mean[t],
                                               all_prior_std[t])

            if self.isreconmsk:

                mu = all_dec_mean[t] * msk[:, t, :]
                std = (all_dec_std[t] * msk[:, t, :]).mul(0.5).exp_()

                cov = []
                for vec in std:
                    cov.append(torch.diag(vec))
                cov = torch.stack(cov)

                nll_loss += - MultivariateNormal(mu, cov).log_prob(all_x[t] * msk[:, t, :]).sum()


                # nll_loss_2 += - Normal(all_dec_mean[t][msk[:, t, :] == 1],
                #                         all_dec_std[t][msk[:, t, :] == 1].mul(0.5).exp_()).log_prob(
                #                         all_x[t][msk[:, t, :] == 1]).sum()
                #
                #
                # nll_loss += - Normal(mu, std).log_prob(all_x[t] * msk[:, t, :]).sum()

                mae_loss += torch.abs(all_dec_mean[t][eval_msk[:, t, :] == 1] - eval_x[:, t, :][eval_msk[:, t, :] == 1]).sum()
            else:
                nll_loss += - Normal(all_dec_mean[t], all_dec_std[t].mul(0.5).exp_()).log_prob(all_x[t]).sum(1)
                mae_loss += torch.abs(all_dec_mean[t] - all_x[t]).sum(1)

        if self.isreconmsk:
            # loss = kld_loss.mean() + (mae_loss + nll_loss) / len(kld_loss)  # KL + MAE + NLL
            loss = kld_loss.mean() + (nll_loss) / len(kld_loss)  # KL + NLL
        else:
            loss = torch.mean(kld_loss + mae_loss + nll_loss)  # KL + MAE + NLL

            # nll_loss += self._nll_gauss(all_dec_mean[t], all_dec_std[t], all_x[t])  # NLL2
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            # kld_loss += - beta * 0.5 * torch.sum(1 + all_enc_std[t] - all_enc_mean[t].pow(2) - all_enc_std[t].exp(), 1)
            # nll_loss += - Normal(all_dec_mean[t], all_dec_std[t].mul(0.5).exp_()).log_prob(all_x[t]).sum(1)  # NLL1
            # print('kld' + str(kld_loss) + 'nll_loss' + str(nll_loss) + 'mae_loss' + str(mae_loss))

        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # loss_total = loss + (self.lambda1 * l1_regularization)
        loss_total = loss
        return loss_total


class FocalLoss(nn.Module):
    def __init__(self,  lambda1, device, alpha=1, gamma=0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device
        self.lambda1 = torch.tensor(lambda1).to(device)

    def forward(self, model, inputs, targets):
        # inputs += 1e-10
        # inputs = inputs.clamp(1e-10, 1.0)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            print("BCE_loss is", BCE_loss)
            # BCE_loss = nn.BCEWithLogitsLoss()
        else:
            # if np.shape(np.where(np.isnan(inputs.cpu().detach().numpy())==True))[1]>0:
            #     print(np.shape(np.where(np.isnan(inputs.cpu().detach().numpy()) == True))[1])
            #     inputs = torch.tensor(np.nan_to_num(inputs.cpu().detach().numpy())).to(self.device)
            #     print(inputs)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            # BCE_loss = nn.BCELoss()
        pt = torch.exp(-1*BCE_loss)
        # pt = torch.exp(-1 * BCE_loss(inputs, targets))
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss(inputs, targets)

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        # loss = torch.mean(F_loss) + (self.lambda1 * l1_regularization)
        loss = torch.mean(F_loss)

        return loss


# SVAELOSS
class SVAELoss(torch.nn.Module):

    def __init__(self, lambda1, device):
        super(SVAELoss, self).__init__()
        self.device = device
        self.lambda1 = torch.tensor(lambda1).to(device)

    def forward(self, model, x, enc_mu, enc_logvar, dec_mu, dec_logvar, beta=1):
        # Reconstruction Loss
        recon_loss = -Normal(dec_mu, dec_logvar.mul(0.5).exp_()).log_prob(x).sum(1)

        # Variational Encoder Loss
        KLD_enc = - beta * 0.5 * torch.sum(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp(), 1)

        # Regularization
        # l1_regularization = torch.tensor(0).float().to(self.device)
        # for param in model.parameters():
        #     l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        # loss = torch.mean(recon_loss + KLD_enc) + (self.lambda1 * l1_regularization)
        loss = torch.mean(recon_loss + KLD_enc)

        return loss

# Asymetric Similarity Loss
class AsymSimiliarityLoss(torch.nn.Module):

    def __init__(self, beta, lambda1, device):
        super(AsymSimiliarityLoss, self).__init__()
        self.beta = beta
        self.lambda1 = lambda1
        self.device = device

    def forward(self, model, y_pred, y):
        nom = (1 + self.beta**2) * torch.sum(y_pred * y.float())
        denom = ((1 + self.beta**2) * torch.sum(y_pred * y.float())) + \
                (self.beta**2 * torch.sum((1-y_pred) * y.float())) + \
                (torch.sum(y_pred * (1 - y).float()))
        asym_sim_loss = nom / denom

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        # loss = asym_sim_loss + (self.lambda1 * l1_regularization)
        loss = asym_sim_loss

        return loss