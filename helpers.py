import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
from scipy.stats.mstats import winsorize
import random

import matplotlib.pyplot as plt
# import pandas as pd
import math
import torch
from torch.optim.optimizer import Optimizer
from typing import NamedTuple

class Config(NamedTuple):
    h_dim: int = 64
    z_dim: int = 16
    z1_dim: int = 128
    z2_dim: int = 64
    z3_dim: int = 32
    z4_dim: int = 16
    w_decay: float = 1e-5
    n_epochs: int = 100
    batch_size: int = 64
    lr_ratio: float = 0.1
    org_learning_rate: float = 1e-3
    learning_rate_decay: int = 10
    dropout_p: float = 0.3
    clip: float = 1
    key_dim: int = 64
    value_dim: int = 64
    predict_dim: int = 2
    output_dim: int = 64
    num_heads: int = 2
    out_ch: int = 3
    conv_kernel: int = 10
    len_mask: int = 10
    out_ch: int = 5
    n_layers: int = 2
    # @classmethod
    # def from_json(cls, file):
    #     return cls(**json.load(open(file, "r")))

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# # Normalization
# def normalize(data, data_mask, mean, std):
#     n_patients = data.shape[0]
#     n_hours = data.shape[1]
#     n_variables = data.shape[2]
#
#     mask = data_mask.copy().reshape(n_patients * n_hours, n_variables)
#     measure = data.copy().reshape(n_patients * n_hours, n_variables)
#
#     isnew = 0
#     if len(mean) == 0 or len(std) == 0:
#         isnew = 1
#         mean_set = np.zeros([n_variables])
#         std_set = np.zeros([n_variables])
#     else:
#         mean_set = mean
#         std_set = std
#     for v in range(n_variables):
#         idx = np.where(mask[:,v] == 1)[0]
#
#         if idx.sum()==0:
#             continue
#
#         if isnew:
#             measure_mean = np.mean(measure[:, v][idx])
#             measure_std = np.std(measure[:, v][idx])
#
#             # Save the Mean & STD Set
#             mean_set[v] = measure_mean
#             std_set[v] = measure_std
#         else:
#             measure_mean = mean[v]
#             measure_std = std[v]
#
#         for ix in idx:
#             if measure_std != 0:
#                 measure[:, v][ix] = (measure[:, v][ix] - measure_mean) / measure_std
#                 measure[:, v][ix] = np.log(measure[:, v][ix])
#             else:
#                 measure[:, v][ix] = measure[:, v][ix] - measure_mean
#                 measure[:, v][ix] = np.log(measure[:, v][ix])
#
#     normalized_data = measure.reshape(n_patients, n_hours, n_variables)
#
#     return normalized_data, mean_set, std_set

# Normalization
def Winsorize(data):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]
    mask = ~np.isnan(data) * 1
    mask = mask.reshape(n_patients * n_hours, n_variables)
    measure = data.copy().reshape(n_patients * n_hours, n_variables)
    measure_orig = data.copy().reshape(n_patients * n_hours, n_variables)

    for v in range(n_variables):
       idx = np.where(mask[:,v] == 1)[0]
       if len(idx)>0:
           limit = 0.02
           measure[:, v][idx] = winsorize(measure[:, v][idx], limits=limit)

    normalized_data = measure.reshape(n_patients, n_hours, n_variables)
    # print(np.array_equal(measure_orig,normalized_data))
    return normalized_data

def normalize(data, mean, std):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]
    mask = ~np.isnan(data) * 1
    mask = mask.reshape(n_patients * n_hours, n_variables)
    measure = data.copy().reshape(n_patients * n_hours, n_variables)

    # Log Transform
    # measure[np.where(measure == 0)] = measure[np.where(measure == 0)] + 1e-10
    # measure[np.where(mask == 1)] = np.log(measure[np.where(mask == 1)])

    isnew = 0
    if len(mean) == 0 or len(std) == 0:
       isnew = 1
       mean_set = np.zeros([n_variables])
       std_set = np.zeros([n_variables])
    else:
       mean_set = mean
       std_set = std
    for v in range(n_variables):
       idx = np.where(mask[:,v] == 1)[0]
       if idx.sum()==0:
           continue
       if isnew:
           measure_mean = np.mean(measure[:, v][idx])
           measure_std = np.std(measure[:, v][idx])
           # Save the Mean & STD Set
           mean_set[v] = measure_mean
           std_set[v] = measure_std
       else:
           measure_mean = mean[v]
           measure_std = std[v]
       for ix in idx:
           if measure_std != 0:
               measure[:, v][ix] = (measure[:, v][ix] - measure_mean) / measure_std
           else:
               measure[:, v][ix] = measure[:, v][ix] - measure_mean
    normalized_data = measure.reshape(n_patients, n_hours, n_variables)
    return normalized_data, mean_set, std_set

def parse_delta(masks, dir_):
    # if dir_ == 'backward':
    #     masks = masks[::-1]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35))
        else:
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(ori_values, ori_masks, values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}

    rec['ori_values'] = np.nan_to_num(ori_values).tolist()
    rec['ori_masks'] = ori_masks.astype('int32').tolist()

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec

def collate_fn(recs):
    rec_dict = {'values_fwd': torch.FloatTensor(np.array([r['values_fwd'] for r in recs])),
                'masks_fwd': torch.FloatTensor(np.array([r['masks_fwd'] for r in recs])),
                'evals_fwd': torch.FloatTensor(np.array([r['evals_fwd'] for r in recs])),
                'eval_masks_fwd': torch.FloatTensor(np.array([r['eval_masks_fwd'] for r in recs])),
                'deltas_fwd': torch.FloatTensor(np.array([r['deltas_fwd'] for r in recs])),
                'values_bwd': torch.FloatTensor(np.array([r['values_bwd'] for r in recs])),
                'masks_bwd': torch.FloatTensor(np.array([r['masks_bwd'] for r in recs])),
                'evals_bwd': torch.FloatTensor(np.array([r['evals_bwd'] for r in recs])),
                'eval_masks_bwd': torch.FloatTensor(np.array([r['eval_masks_bwd'] for r in recs])),
                'deltas_bwd': torch.FloatTensor(np.array([r['deltas_fwd'] for r in recs])),
                'labels': torch.FloatTensor(np.array([r['labels'] for r in recs]))
                }
    return rec_dict

def collate_fn_fMRI(recs):
    rec_dict = {'values_fwd': torch.FloatTensor(np.array([r['values_fwd'] for r in recs])),
                'evals_fwd': torch.FloatTensor(np.array([r['evals_fwd'] for r in recs])),
                'labels': torch.FloatTensor(np.array([r['labels'] for r in recs]))
                }
    return rec_dict

def parse_delta(masks):
    [T, D] = masks.shape
    deltas = []

    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D))
        else:
            deltas.append(np.ones(D) + (1 - masks[t]) * deltas[-1])

    return np.array(deltas)

# Define Sample Loader
def sample_loader(data, mask, label, batch_size, isMasking, mask_ratio, ZeroImpute):
    # Random seed
    manualSeed = 128
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)

    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    [N, T, D] = data.shape
    data = data.reshape(N, T*D)
    mask = mask.reshape(N, T*D)
    recs = []

    for i in range(N):
        ori_values = data[i].reshape(T, D)
        ori_masks = ~np.isnan(ori_values).reshape(T, D)

        values = data[i].copy()

        # randomly eliminate 10% values as the imputation ground-truth
        if isMasking:
            indices = np.where(~np.isnan(data[i]))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * mask_ratio))  # 5%/10% scenario
            values[indices] = np.nan

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(data[i]))


        evals_fwd = data[i].reshape(T, D)
        values_fwd = values.reshape(T, D)
        masks_fwd = masks.reshape(T, D)
        eval_masks_fwd = eval_masks.reshape(T, D)

        evals_bwd = evals_fwd[::-1]
        values_bwd = values_fwd[::-1]
        masks_bwd = masks_fwd[::-1]
        eval_masks_bwd = eval_masks_fwd[::-1]

        rec = {}
        rec['labels'] = label[i]
        # replace NaN with zero
        if ZeroImpute:
            evals_fwd[np.isnan(evals_fwd)] = 0
            values_fwd[np.isnan(values_fwd)] = 0

            evals_bwd[np.isnan(evals_bwd)] = 0
            values_bwd[np.isnan(values_bwd)] = 0

        deltas_fwd = parse_delta(masks_fwd)
        deltas_bwd = parse_delta(masks_bwd)

        if ZeroImpute:
            rec['values_fwd'] = np.nan_to_num(values_fwd).tolist()
            rec['masks_fwd'] = masks_fwd.astype('int32').tolist()
            rec['evals_fwd'] = np.nan_to_num(evals_fwd).tolist()
            rec['eval_masks_fwd'] = eval_masks_fwd.astype('int32').tolist()

            rec['values_bwd'] = np.nan_to_num(values_bwd).tolist()
            rec['masks_bwd'] = masks_bwd.astype('int32').tolist()
            rec['evals_bwd'] = np.nan_to_num(evals_bwd).tolist()
            rec['eval_masks_bwd'] = eval_masks_bwd.astype('int32').tolist()

            rec['deltas_fwd'] = deltas_fwd.astype('int32').tolist()
            rec['deltas_bwd'] = deltas_bwd.astype('int32').tolist()

        else:
            rec['values_fwd'] = values_fwd.tolist()
            rec['masks_fwd'] = masks_fwd.astype('int32').tolist()
            rec['evals_fwd'] = evals_fwd.tolist()
            rec['eval_masks_fwd'] = eval_masks_fwd.astype('int32').tolist()

            rec['values_bwd'] = values_bwd.tolist()
            rec['masks_bwd'] = masks_bwd.astype('int32').tolist()
            rec['evals_bwd'] = evals_bwd.tolist()
            rec['eval_masks_bwd'] = eval_masks_bwd.astype('int32').tolist()

            rec['deltas_fwd'] = deltas_fwd.astype('int32').tolist()
            rec['deltas_bwd'] = deltas_bwd.astype('int32').tolist()

        recs.append(rec)

    loader = torch.utils.data.DataLoader(recs,
                                         batch_size=batch_size,
                                         # num_workers=1,
                                         shuffle=True,
                                         pin_memory=True,
                                         collate_fn=collate_fn)
    return loader
    #
    # else:
    #     # Define the loader
    #     dataset = torch.utils.data.TensorDataset(torch.tensor(data).float(),
    #                                                torch.tensor(mask).float(),
    #                                                torch.tensor(label).float())
    #     loader = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=batch_size,
    #                                            # num_workers=1,
    #                                            shuffle=True,
    #                                            pin_memory=True)


# def calculate_performance(y, y_score, y_pred):
#     # Calculate Evaluation Metrics
#     acc = accuracy_score(y_pred, y)
#     tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
#     # total = tn + fp + fn + tp
#     sen = np.nan_to_num(tp / (tp + fn))
#     recall = np.nan_to_num(tp / (tp + fn))
#     p, r, t = precision_recall_curve(y, y_score)
#     auprc = metrics.auc(r, p)
#     spec = np.nan_to_num(tn / (tn + fp))
#     # acc = ((tn + tp) / total) * 100
#     balacc = ((spec + sen) / 2) * 100
#     auc = roc_auc_score(y, y_score)
#     prec = np.nan_to_num(tp / (tp + fp))
#
#     return auc, auprc, acc, balacc, sen, spec, prec, recall

def calculate_performance(y, y_score, y_pred):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score)
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall

# def plot_imputation(dir, x, x_recon_vae, x_imp_vae, x_imp_rnn, k, phase, epoch):
# def plot_imputation(dir, x, m, x_imp_vae, x_imp_rnn, k, phase, epoch):
def plot_imputation(dir, x, m, x_imp_vae, k, phase, epoch):
    fig, axes = plt.subplots(2, 2, sharey=True)

    # cx1 = axes[0, 0].imshow(x.T)
    cx1 = axes[0, 0].imshow(x.T, vmin=0, vmax=1)
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].title.set_text('x (original data)')

    # cx2 = axes[0, 1].imshow(x_recon_vae.T)
    # cx2 = axes[0, 1].imshow(x_recon_vae.T, vmin=0, vmax=1)
    # axes[0, 1].title.set_text(r'$\bar{x}$ (reconstructed data by VAE)')
    cx2 = axes[0, 1].imshow(m.T, vmin=0, vmax=1)
    axes[0, 1].title.set_text('m (mask)')

    # cx3 = axes[1, 0].imshow(x_imp_vae.T)
    cx3 = axes[1, 0].imshow(x_imp_vae.T, vmin=0, vmax=1)
    axes[1, 0].set_ylabel('Variables')
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].title.set_text(r'$\hat{x}$ (imputed data by VAE)')

    # cx4 = axes[1, 1].imshow(x_imp_rnn.T)
    # cx4 = axes[1, 1].imshow(x_imp_rnn.T, vmin=0, vmax=1)
    # axes[1, 1].set_xlabel('Hours')
    # axes[1, 1].title.set_text(r'$\hat{x}$ (imputed data by RNN)')
    fig.colorbar(cx1, ax=axes.ravel().tolist(), orientation='vertical')
    # fig.tight_layout()
    strk = str(k)
    stre = str(epoch)
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    plt.savefig(dir + '/img/' + phase + '/x_' + strk + '_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight')
    plt.close()

def plot_imputation_result(dir, x, m, x_imp_vae, unc, k, phase, epoch):
    fig, axes = plt.subplots(2, 2, sharey=True)

    # cx1 = axes[0, 0].imshow(x.T)
    cx1 = axes[0, 0].imshow(x.T, vmin=0, vmax=1)
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].title.set_text('x (original data)')

    # cx2 = axes[0, 1].imshow(x_recon_vae.T)
    # cx2 = axes[0, 1].imshow(x_recon_vae.T, vmin=0, vmax=1)
    # axes[0, 1].title.set_text(r'$\bar{x}$ (reconstructed data by VAE)')
    cx2 = axes[0, 1].imshow(m.T, vmin=0, vmax=1)
    axes[0, 1].title.set_text('m (mask)')

    # cx3 = axes[1, 0].imshow(x_imp_vae.T)
    cx3 = axes[1, 0].imshow(x_imp_vae.T, vmin=0, vmax=1)
    axes[1, 0].set_ylabel('Variables')
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].title.set_text(r'$\hat{x}$ (imputed data by VAE)')

    # cx4 = axes[1, 1].imshow(x_imp_rnn.T)
    cx4 = axes[1, 1].imshow(unc.T, vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].title.set_text('u (uncertainty)')
    fig.colorbar(cx1, ax=axes.ravel().tolist(), orientation='vertical')
    # fig.tight_layout()
    strk = str(k)
    stre = str(epoch)
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    plt.savefig(dir + '/img/' + phase + '/x_' + strk + '_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight')
    plt.close()

# def plot_predictions(dir, x, eval_x, m, eval_m, x_imp_vae, unc, dec_mean, dec_std, k, epoch, batch_idx, issampling=True):
def plot_predictions(dir, x, eval_x, m, eval_m, comb, unc, k, epoch, batch_idx):

    """
    x: (Batch x Time x Variable)
    eval: (Batch x Time x Variable)
    m: (Batch x Time x Variable)
    eval_m: (Batch x Time x Variable)
    comb: (Time x Batch x Variable)
    unc: (Time x Batch x Variable)
    """

    B, T, V = np.shape(x)
    t_set = [i for i in range(T)]

    for b in range(5):
        for i in range(V):
            x_sample = x[b, :, i]
            m_sample = m[b, :, i]
            eval_x_sample = eval_x[b, :, i]
            eval_m_sample = eval_m[b, :, i]
            comb_sample = comb[:, b, i]
            unc_sample = unc[:, b, i]

            t_observation = np.where(m_sample == 1)[0]
            eval_t_observation = np.where(eval_m_sample == 1)[0]
            x_observation = x_sample[m_sample == 1]
            eval_x_observation = eval_x_sample[eval_m_sample == 1]

            plt.xlim(0, 48)
            plt.axes([0.025, 0.025, 0.95, 0.95])

            plt.plot(t_set, comb_sample, color='blue')
            plt.fill_between(t_set, comb_sample - unc_sample, comb_sample + unc_sample,
                             facecolor='#089FFF')
            plt.scatter(t_observation, x_observation, c='red')
            plt.scatter(eval_t_observation, eval_x_observation, c='green')

            plt.xlabel('Time')
            plt.ylabel('Normalized signal values')
            plt.grid(True)
            plt.savefig(dir + '/img/var_' + str(i) + '_batch_' + str(b) + '_fold_' + str(k) + '_epoch_' + str(epoch) + '.png', bbox_inches='tight')
            plt.close()
            # axes[i//7, i%7].plot(t_set, x_imp_vae_sample, color='blue')
            # axes[i//7, i%7].fill_between(t_set, x_imp_vae_sample - unc_sample, x_imp_vae_sample + unc_sample, facecolor='#089FFF')
            # axes[i//7, i%7].scatter(t_observation, x_observation, c='red')
            # # axes[i//7, i%7].set_xlabel('Time')
            # # axes[i//7, i%7].set_ylabel('Normalized signal values')
            # axes[i//7, i%7].grid(True)
            # plt.show()


def plot_imputation_vrnn(dir, x, m, x_imp_vae, unc, k, phase, epoch, update, reset):
    # fig, axes = plt.subplots(3, 2, sharey=True)
    fig, axes = plt.subplots(3, 2, sharex=True)

    # cx1 = axes[0, 0].imshow(x.T)
    # cx1 = axes[0, 0].imshow(x.T, vmin=0, vmax=1)
    cx1 = axes[0, 0].imshow(x.T, vmin=x.min(), vmax=x.max(), cmap='jet', aspect="auto")
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].title.set_text('x (original data)')
    fig.colorbar(cx1, ax=axes[0, 0], orientation='vertical', pad=0.03, fraction=0.0338)

    # cx2 = axes[0, 1].imshow(x_recon_vae.T)
    # cx2 = axes[0, 1].imshow(x_recon_vae.T, vmin=0, vmax=1)
    # axes[0, 1].title.set_text(r'$\bar{x}$ (reconstructed data by VAE)')
    cx2 = axes[0, 1].imshow(m.T, vmin=m.min(), vmax=m.max(), cmap='gray', aspect="auto")
    # cx2 = axes[0, 1].imshow(m.T, vmin=0, vmax=1)
    axes[0, 1].title.set_text('m (mask)')
    fig.colorbar(cx2, ax=axes[0, 1], orientation='vertical', pad=0.03, fraction=0.0338)

    # cx3 = axes[1, 0].imshow(x_imp_vae.T)
    # cx3 = axes[1, 0].imshow(x_imp_vae.T, vmin=0, vmax=1)
    cx3 = axes[1, 0].imshow(x_imp_vae.T, vmin=x_imp_vae.min(), vmax=x_imp_vae.max(), cmap='jet', aspect="auto")
    axes[1, 0].set_ylabel('Variables')
    # axes[1, 0].set_xlabel('Hours')
    axes[1, 0].title.set_text(r'$\hat{x}$ (imputed data by VRNN)')
    fig.colorbar(cx3, ax=axes[1, 0], orientation='vertical', pad=0.03, fraction=0.0338)

    # cx4 = axes[1, 1].imshow(x_imp_rnn.T)
    # cx4 = axes[1, 1].imshow(unc.T, vmin=0, vmax=1)
    cx4 = axes[1, 1].imshow(unc.T, vmin=unc.min(), vmax=unc.max(), cmap='YlOrRd', aspect="auto")
    # axes[1, 1].set_xlabel('Hours')
    axes[1, 1].title.set_text('u (uncertainty)')
    fig.colorbar(cx4, ax=axes[1, 1], orientation='vertical', pad=0.03, fraction=0.0338)
    # fig.colorbar(cx4, ax=axes.ravel().tolist(), orientation='vertical')
    # fig.tight_layout()

    cx5 = axes[2, 0].imshow(update.T, vmin=update.min(), vmax=update.max(), cmap='cool', aspect="auto")
    axes[2, 0].set_xlabel('Hours')
    axes[2, 0].set_ylabel('Hidden nodes')
    axes[2, 0].title.set_text('Update gate')
    fig.colorbar(cx5, ax=axes[2, 0], orientation='vertical', pad=0.03, fraction=0.0338)

    cx6 = axes[2, 1].imshow(reset.T, vmin=reset.min(), vmax=reset.max(), cmap='cool', aspect="auto")
    axes[2, 1].set_xlabel('Hours')
    axes[2, 1].title.set_text('Reset gate')
    fig.colorbar(cx6, ax=axes[2, 1], orientation='vertical', pad=0.03, fraction=0.0338)

    strk = str(k)
    stre = str(epoch)
    plt.subplots_adjust(left=0.1, right=0.7,wspace=0.5, hspace=0.5)
    # plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
    plt.savefig(dir + '/img/' + phase + '/x_' + strk + '_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight')
    plt.close()


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

# Define Sample Loader
def sample_loader_fMRI(data, label, batch_size):
    # Random seed
    manualSeed = 128
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)

    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    [N, T, D] = data.shape
    data = data.reshape(N, T*D)
    recs = []

    for i in range(N):
        ori_values = data[i].reshape(T, D)
        ori_masks = ~np.isnan(ori_values).reshape(T, D)

        values = data[i].copy()


        evals_fwd = data[i].reshape(T, D)
        values_fwd = values.reshape(T, D)


        rec = {}
        rec['labels'] = label[i]
        rec['values_fwd'] = np.nan_to_num(values_fwd).tolist()
        rec['evals_fwd'] = np.nan_to_num(evals_fwd).tolist()


        recs.append(rec)

    loader = torch.utils.data.DataLoader(recs,
                                         batch_size=batch_size,
                                         # num_workers=1,
                                         shuffle=True,
                                         pin_memory=True,
                                         collate_fn=collate_fn_fMRI)
    return loader