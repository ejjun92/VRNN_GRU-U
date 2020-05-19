import torch.nn as nn
import os
os.environ["GEVENT_SUPPORT"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from losses import FocalLoss, VRNNLoss, WeightedBCE
from model import *
from helpers import *
import pickle
import torch
import torch.optim as optim
import random
import datetime
import numpy as np
import argparse
import GPUtil


parser = argparse.ArgumentParser()
parser.add_argument('--lambda1', type=float, default=1e-6)
parser.add_argument('--gamma', type=float, default=5)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--mask_ratio', type=float, default=0.05)
parser.add_argument('--fold_num', type=int, default=0)
parser.add_argument('--data', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--isreparam', type=int, default=0)
parser.add_argument('--isreconmsk', type=int, default=1)
parser.add_argument('--isdecaying', type=int, default=6)
parser.add_argument('--out_ch', type=int, default=5)
args = parser.parse_args()

if args.data == 0:
    dataset = 'physionet'
else:
    dataset = 'mimic'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if dataset == 'mimic':
    kfold_data = np.load(open('../data/MIMIC3/24_new/data_nan.p', 'rb'), mmap_mode='r', allow_pickle=True)
    kfold_mask = np.load(open('../data/MIMIC3/24_new/mask.p', 'rb'), mmap_mode='r', allow_pickle=True)
    kfold_label = np.load(open('../data/MIMIC3/24_new/label.p', 'rb'), mmap_mode='r', allow_pickle=True)
else:
    kfold_data = np.load(open('../data/data35_avg/kfold_data.p', 'rb'), mmap_mode='r', allow_pickle=True)
    kfold_mask = np.load(open('../data/data35_avg/kfold_mask.p', 'rb'), mmap_mode='r', allow_pickle=True)
    kfold_label = np.load(open('../data/data35_avg/kfold_label.p', 'rb'), mmap_mode='r', allow_pickle=True)


# hyperparameters
if dataset == 'mimic':
    model_cfg = Config(h_dim=64,
                       z1_dim=32,
                       z2_dim=32,
                       z3_dim=16,
                       z4_dim=16,
                       batch_size=64,
                       w_decay=1e-5,
                       n_epochs=80,
                       lr_ratio=0.5,
                       org_learning_rate=1e-3,
                       learning_rate_decay=10,
                       dropout_p=0.3,
                       clip=10,
                       out_ch=args.out_ch
                       )
else:
    model_cfg = Config(h_dim=64,
                       z_dim=16,
                       batch_size=128,
                       w_decay=1e-3,
                       n_epochs=80,
                       lr_ratio=0.5,
                       org_learning_rate=1e-3,
                       learning_rate_decay=20,  #
                       dropout_p=0.2,
                       clip=10,
                       out_ch=args.out_ch
                       )

FFA = False
x_dim = np.shape(kfold_data[0][0])[2]
mask_ratio = args.mask_ratio
fold_num = args.fold_num
isreparam = args.isreparam
isreconmsk = args.isreconmsk
isdecaying = args.isdecaying
n_layers = 1
KFold = len(kfold_data)
beta = 1
beta_VAE = 1
lambda1 = args.lambda1 # 0.0001
gamma = args.gamma # 2
alpha = args.alpha # 0.5

# kfold performance
kfold_acc = []
kfold_balacc = []
kfold_auc = []
kfold_auc_prc = []
kfold_sen = []
kfold_spec = []
kfold_precision = []
kfold_recall = []
kfold_f1_score_pr = []
kfold_f2_score_pr = []
kfold_mae = []
kfold_mre = []

def switch(fold_num):
    return {0: range(0, 1),
            1: range(1, 2),
            2: range(2, 3),
            3: range(3, 4),
            4: range(4, 5)}[fold_num]

def switch_decaying(isdecaying):
    return {0: 'both_decay',
            1: 'input_decay',
            2: 'hidden_decay',
            3: 'none',
            4: 'fr_1',
            5: 'fr_2',
            6: 'fr_3'
            }[isdecaying]

isdecaying = switch_decaying(isdecaying)

if dataset == 'mimic':
    dir = './log/MIMIC3/200515/fold_' + str(fold_num) + '_' + isdecaying + '_BiVRNN_GRUU_reparam_' + str(isreparam) + '_Classification_MIMIC3_' + str(mask_ratio) + '_h_' + str(model_cfg.h_dim) + '_z_' + str(model_cfg.z1_dim) + '_outch_' +str(model_cfg.out_ch) + '_' + str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S')) + '/'
else:
    dir = './log/data35_avg/200515/fold_' + str(fold_num) + '_' + isdecaying + '_BiVRNN_GRUU_reparam_' + str(isreparam) + '_Classification_PhysioNet_' + str(mask_ratio) + '_h_' + str(model_cfg.h_dim) + '_z_' + str(model_cfg.z_dim) + '_outch_' +str(model_cfg.out_ch) + '_' + str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S')) + '/'

def get_lr(optimizer):
   for param_group in optimizer.param_groups:
       return param_group['lr']

if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + 'img/')
    os.makedirs(dir + 'img/train/')
    os.makedirs(dir + 'img/valid/')
    os.makedirs(dir + 'img/test/')
    os.makedirs(dir + 'analysis/')
    os.makedirs(dir + 'analysis/train/')
    os.makedirs(dir + 'analysis/valid/')
    os.makedirs(dir + 'analysis/test/')
    os.makedirs(dir + 'model/')
    os.makedirs(dir + 'tflog/')
    for k in range(KFold):
        os.makedirs(dir + 'model/' + str(k) + '/')
        os.makedirs(dir + 'model/' + str(k) + '/img')
        os.makedirs(dir + 'model/' + str(k) + '/analysis')

# Text Logging
f = open(dir + 'log.txt', 'a')
writelog(f, '---------------')
writelog(f, 'MODEL: Bi-directional VRNN + GRU-U')
writelog(f, 'TRAINING PARAMETER')
writelog(f, 'Learning Rate : ' + str(model_cfg.org_learning_rate))
writelog(f, 'Batch Size : ' + str(model_cfg.batch_size))
writelog(f, 'TRAINING LOG')


def train(epoch, train_loader):
    model.train()

    train_loss = 0
    n_batches = 0

    for batch_idx, data in enumerate(train_loader):
        # Data
        x_fwd = data['values_fwd'].to(device)  # Batch x Time x Variable
        m_fwd = data['masks_fwd'].to(device)  # Batch x Time x Variable
        eval_x_fwd = data['evals_fwd'].to(device)  # Batch x Time x Variable
        eval_m_fwd = data['eval_masks_fwd'].to(device)  # Batch x Time x Variable
        deltas_fwd = data['deltas_fwd'].to(device)  # Batch x Time x Variable

        x_bwd = data['values_bwd'].to(device)  # Batch x Time x Variable
        m_bwd = data['masks_bwd'].to(device)  # Batch x Time x Variable
        eval_x_bwd = data['evals_bwd'].to(device)  # Batch x Time x Variable
        eval_m_bwd = data['eval_masks_bwd'].to(device)  # Batch x Time x Variable
        deltas_bwd = data['deltas_bwd'].to(device)  # Batch x Time x Variable

        y = data['labels'].to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        vrnn_f, vrnn_b = model(x_fwd.transpose(0, 1), m_fwd, deltas_fwd, x_bwd.transpose(0, 1), m_bwd, deltas_bwd)

        combi_f = torch.stack(vrnn_f['combi'])  # Time x Batch x Variable
        combi_b = torch.stack(vrnn_b['combi'])  # Time x Batch x Variable
        combi = (combi_f.to('cpu').detach().numpy() + combi_b.to('cpu').detach().numpy()[::-1]) / 2  # Time x Batch x Variable
        combi = torch.FloatTensor(combi).permute(1, 0, 2).to('cpu').detach().numpy()  # Batch x Time x Variable

        unc_f = torch.stack(vrnn_f['unc'])  # Time x Batch x Variable
        unc_b = torch.stack(vrnn_b['unc'])  # Time x Batch x Variable
        unc = (unc_f.to('cpu').detach().numpy() + unc_b.to('cpu').detach().numpy()[::-1]) / 2  # Time x Batch x Variable
        unc = torch.FloatTensor(unc).permute(1, 0, 2).to('cpu').detach().numpy()  # Batch x Time x Variable


        # (i) VRNN loss
        loss_vrnn_fwd = vrnn_loss(model, vrnn_f['prior_mean'], vrnn_f['prior_std'], vrnn_f['x'], vrnn_f['enc_mean'],
                                  vrnn_f['enc_std'], vrnn_f['dec_mean'], vrnn_f['dec_std'], m_fwd, eval_x_fwd, eval_m_fwd, beta_VAE)
        loss_vrnn_bwd = vrnn_loss(model, vrnn_b['prior_mean'], vrnn_b['prior_std'], vrnn_b['x'], vrnn_b['enc_mean'],
                                  vrnn_b['enc_std'], vrnn_b['dec_mean'], vrnn_b['dec_std'], m_bwd, eval_x_bwd, eval_m_bwd, beta_VAE)
        loss_vrnn = (loss_vrnn_fwd + loss_vrnn_bwd)

        # (ii) consistency loss
        loss_consistency = np.abs(combi_f.to('cpu').detach().numpy() - combi_b.to('cpu').detach().numpy()[::-1]).mean()

        # (iii) imputation loss (MAE)
        eval_data_f = eval_x_fwd.data.cpu().numpy()
        eval_masks_f = eval_m_fwd.data.cpu().numpy()
        eval_ground_truth_f = np.asarray(eval_data_f[np.where(eval_masks_f == 1)])

        imputation = combi
        masked_prediction = np.asarray(imputation[np.where(eval_masks_f == 1)])
        mae = np.abs(eval_ground_truth_f - masked_prediction).mean()

        # (iv) classification loss
        loss_cl_fwd = classification_loss(model, vrnn_f['out'], y)
        loss_cl_bwd = classification_loss(model, vrnn_b['out'], y)
        loss_cl = loss_cl_fwd + loss_cl_bwd

        loss = 0.00001 * loss_vrnn + loss_consistency + 0.01 * mae + loss_cl

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), model_cfg.clip)

        n_batches += 1


    # Averaging the loss
    train_loss = train_loss / n_batches
    writelog(f, 'Loss : ' + str(train_loss))

def test(phase, epoch, test_loader):
    """uses test data to evaluate
    likelihood of the model"""
    model.eval()
    test_loss = 0.0
    n_batches = 0.0

    y_gts = np.array([]).reshape(0)
    y_preds = np.array([]).reshape(0)
    y_scores = np.array([]).reshape(0)


    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Data
            x_fwd = data['values_fwd'].to(device)  # Batch x Time x Variable
            m_fwd = data['masks_fwd'].to(device)  # Batch x Time x Variable
            eval_x_fwd = data['evals_fwd'].to(device)  # Batch x Time x Variable
            eval_m_fwd = data['eval_masks_fwd'].to(device)  # Batch x Time x Variable
            deltas_fwd = data['deltas_fwd'].to(device)  # Batch x Time x Variable

            x_bwd = data['values_bwd'].to(device)  # Batch x Time x Variable
            m_bwd = data['masks_bwd'].to(device)  # Batch x Time x Variable
            eval_x_bwd = data['evals_bwd'].to(device)  # Batch x Time x Variable
            eval_m_bwd = data['eval_masks_bwd'].to(device)  # Batch x Time x Variable
            deltas_bwd = data['deltas_bwd'].to(device)  # Batch x Time x Variable

            y = data['labels'].to(device)
            y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten()])

            # forward + backward + optimize
            vrnn_f, vrnn_b = model(x_fwd.transpose(0, 1), m_fwd, deltas_fwd, x_bwd.transpose(0, 1), m_bwd, deltas_bwd)

            combi_f = torch.stack(vrnn_f['combi'])  # Time x Batch x Variable
            combi_b = torch.stack(vrnn_b['combi'])  # Time x Batch x Variable
            combi = (combi_f.to('cpu').detach().numpy() + combi_b.to('cpu').detach().numpy()[::-1]) / 2  # Time x Batch x Variable
            combi = torch.FloatTensor(combi).permute(1, 0, 2).to('cpu').detach().numpy()  # Batch x Time x Variable

            unc_f = torch.stack(vrnn_f['unc'])  # Time x Batch x Variable
            unc_b = torch.stack(vrnn_b['unc'])  # Time x Batch x Variable
            unc = (unc_f.to('cpu').detach().numpy() + unc_b.to('cpu').detach().numpy()[::-1]) / 2  # Time x Batch x Variable
            unc = torch.FloatTensor(unc).permute(1, 0, 2).to('cpu').detach().numpy()  # Batch x Time x Variable

            # (i) VRNN loss
            loss_vrnn_fwd = vrnn_loss(model, vrnn_f['prior_mean'], vrnn_f['prior_std'], vrnn_f['x'], vrnn_f['enc_mean'],
                                      vrnn_f['enc_std'], vrnn_f['dec_mean'], vrnn_f['dec_std'], m_fwd, eval_x_fwd,
                                      eval_m_fwd, beta_VAE)
            loss_vrnn_bwd = vrnn_loss(model, vrnn_b['prior_mean'], vrnn_b['prior_std'], vrnn_b['x'], vrnn_b['enc_mean'],
                                      vrnn_b['enc_std'], vrnn_b['dec_mean'], vrnn_b['dec_std'], m_bwd, eval_x_bwd,
                                      eval_m_bwd, beta_VAE)
            loss_vrnn = (loss_vrnn_fwd + loss_vrnn_bwd)

            # (ii) consistency loss
            loss_consistency = np.abs(combi_f.to('cpu').detach().numpy() - combi_b.to('cpu').detach().numpy()[::-1]).mean()

            # (iii) classification loss
            loss_cl_fwd = classification_loss(model, vrnn_f['out'], y)
            loss_cl_bwd = classification_loss(model, vrnn_b['out'], y)
            loss_cl = loss_cl_fwd + loss_cl_bwd

            print('====> Test loss: VRNN = {:.4f}, Classification = {:.4f}'.format(loss_vrnn, loss_cl))

            loss = 0.00001 * loss_vrnn + loss_consistency + loss_cl

            test_loss += loss.item()

            y_pred = np.round(y_score.to('cpu').detach().numpy())  # (Fx1)
            y_score = y_score.to('cpu').detach().numpy()  # (Fx1)
            y_preds = np.hstack([y_preds, y_pred.reshape(-1)])
            y_scores = np.hstack([y_scores, y_score.reshape(-1)])

            n_batches += 1


    # Averaging the loss
    test_loss /= n_batches
    writelog(f, 'Loss : ' + str(test_loss))

    auc, auprc, acc, balacc, sen, spec, prec, recall = calculate_performance(y_gts, y_scores, y_preds)

    writelog(f, 'AUC : ' + str(auc))
    writelog(f, 'AUC PRC : ' + str(auprc))
    writelog(f, 'Accuracy : ' + str(acc))
    writelog(f, 'BalACC : ' + str(balacc))
    writelog(f, 'Sensitivity : ' + str(sen))
    writelog(f, 'Specificity : ' + str(spec))
    writelog(f, 'Precision : ' + str(prec))
    writelog(f, 'Recall : ' + str(recall))

    # Tensorboard Logging
    info = {'loss': test_loss,
            'balacc': balacc,
            'auc': auc,
            'auc_prc': auprc,
            'sens': sen,
            'spec': spec,
            'precision': prec,
            'recall': recall
            }

    return auc, auprc, acc, balacc, sen, spec, prec, recall


# Loop for kfold
# for k in range(KFold):
for k in switch(fold_num):
    print(str(k) + 'Fold')
    writelog(f, '---------------')
    writelog(f, 'FOLD ' + str(k))


    # Get dataset
    train_data = kfold_data[k][0]
    train_mask = kfold_mask[k][0]
    miss_idx = np.where(train_mask == 0)
    train_label = kfold_label[k][0]

    valid_data = kfold_data[k][1]
    valid_mask = kfold_mask[k][1]
    miss_idx = np.where(valid_mask == 0)
    valid_label = kfold_label[k][1]

    test_data = kfold_data[k][2]
    test_mask = kfold_mask[k][2]
    miss_idx = np.where(test_mask == 0)
    test_label = kfold_label[k][2]

    # Winsorization (2nd-98th percentile)
    writelog(f, 'Winsorization')
    train_data = Winsorize(train_data)
    valid_data = Winsorize(valid_data)
    test_data = Winsorize(test_data)

    # Normalization
    writelog(f, 'Normalization')
    train_data, mean_set, std_set = normalize(train_data, [], [])
    valid_data, m, s = normalize(valid_data, mean_set, std_set)
    test_data, m, s = normalize(test_data, mean_set, std_set)

    # Get data dimensionality
    n_patients_train = train_data.shape[0]
    n_patients_valid = valid_data.shape[0]
    n_patients_test = test_data.shape[0]
    n_hours = train_data.shape[1]
    n_variables = train_data.shape[2]

    # Define Loaders
    train_loader = sample_loader(train_data, train_mask, train_label, model_cfg.batch_size, isMasking=True, mask_ratio=mask_ratio, ZeroImpute=True)
    valid_loader = sample_loader(valid_data, valid_mask, valid_label, model_cfg.batch_size, isMasking=False, mask_ratio=mask_ratio, ZeroImpute=True)
    test_loader = sample_loader(test_data, test_mask, test_label, model_cfg.batch_size, isMasking=False, mask_ratio=mask_ratio, ZeroImpute=True)

    # Define Model & Optimizer
    if dataset == 'mimic':
        model = BiVRNN(x_dim, model_cfg.h_dim, model_cfg.z1_dim, n_layers, out_ch=model_cfg.out_ch, dropout_p=model_cfg.dropout_p, isdecaying=isdecaying,
                        FFA=FFA, isreparam=isreparam, issampling=False, device=device).to(device)
    else:
        model = BiVRNN(x_dim, model_cfg.h_dim, model_cfg.z_dim, n_layers, out_ch=model_cfg.out_ch, dropout_p=model_cfg.dropout_p, isdecaying=isdecaying,
                        FFA=FFA, isreparam=isreparam, issampling=False, device=device).to(device)

    vrnn_loss = VRNNLoss(lambda1, device, isreconmsk=isreconmsk)

    classification_loss = FocalLoss(lambda1, device, alpha, gamma, logits=False)

    # Reset Best AUC
    bestValidAUC = 0
    best_epoch = 0

    optimizer = RAdam(list(model.parameters()), lr=model_cfg.org_learning_rate, weight_decay=model_cfg.w_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=model_cfg.learning_rate_decay, gamma=0.5)


    for epoch in range(1, model_cfg.n_epochs+1):
        writelog(f, '------ Epoch ' + str(epoch))

        writelog(f, 'Training')
        train(epoch, train_loader)

        writelog(f, 'Validation')
        auc, auprc, acc, balacc, sen, spec, prec, recall = test('valid', epoch, valid_loader)

        if auc > bestValidAUC:
            torch.save(model, dir + 'model/' + str(k) + '/' + str(epoch) + '_vrnn.pt')
            writelog(f, 'Best validation AUC is found! Validation AUC : ' + str(auc))
            writelog(f, 'Models at Epoch ' + str(k) + '/' + str(epoch) + ' are saved!')
            bestValidAUC = auc
            best_epoch = epoch

        writelog(f, 'Test')
        auc, auprc, acc, balacc, sen, spec, prec, recall = test('test', epoch, test_loader)
        scheduler.step()



    # Load Best Validation AUC
    vrnn_best_model = torch.load(dir + 'model/' + str(k) + '/' + str(best_epoch) + '_vrnn.pt')
    writelog(f, 'Final Test')
    auc, auprc, acc, balacc, sen, spec, prec, recall, final_te_pred, final_te_lab = test('test', epoch, test_loader)


    kfold_auc.append(auc)
    kfold_auc_prc.append(auprc)
    kfold_acc.append(acc)
    kfold_balacc.append(balacc)
    kfold_sen.append(sen)
    kfold_spec.append(spec)
    kfold_precision.append(prec)
    kfold_recall.append(recall)


np.savez('./statistical_test/VRNN_pred_' + str(fold_num) + '_' + dataset + '_' + str(mask_ratio) + '.npz', pred=final_te_pred, allow_pickle=True)
np.savez('./statistical_test/VRNN_lab_' + str(fold_num) + '_' + dataset + '_' + str(mask_ratio) + '.npz', pred=final_te_lab, allow_pickle=True)

writelog(f, '---------------')
writelog(f, 'SUMMARY OF ALL KFOLD')

mean_acc = round(np.mean(kfold_acc), 5)
std_acc = round(np.std(kfold_acc), 5)

mean_auc = round(np.mean(kfold_auc), 5)
std_auc = round(np.std(kfold_auc), 5)

mean_auc_prc = round(np.mean(kfold_auc_prc), 5)
std_auc_prc = round(np.std(kfold_auc_prc), 5)

mean_sen = round(np.mean(kfold_sen), 5)
std_sen = round(np.std(kfold_sen), 5)

mean_spec = round(np.mean(kfold_spec), 5)
std_spec = round(np.std(kfold_spec), 5)

mean_precision = round(np.mean(kfold_precision), 5)
std_precision = round(np.std(kfold_precision), 5)

mean_recall = round(np.mean(kfold_recall), 5)
std_recall = round(np.std(kfold_recall), 5)

mean_balacc = round(np.mean(kfold_balacc), 5)
std_balacc = round(np.std(kfold_balacc), 5)


writelog(f, 'AUC : ' + str(mean_auc) + ' + ' + str(std_auc))
writelog(f, 'AUC PRC : ' + str(mean_auc_prc) + ' + ' + str(std_auc_prc))
writelog(f, 'Accuracy : ' + str(mean_acc) + ' + ' + str(std_acc))
writelog(f, 'BalACC : ' + str(mean_balacc) + ' + ' + str(std_balacc))
writelog(f, 'Sensitivity : ' + str(mean_sen) + ' + ' + str(std_sen))
writelog(f, 'Specificity : ' + str(mean_spec) + ' + ' + str(std_spec))
writelog(f, 'Precision : ' + str(mean_precision) + ' + ' + str(std_precision))
writelog(f, 'Recall : ' + str(mean_recall) + ' + ' + str(std_recall))
writelog(f, '---------------')
writelog(f, 'END OF CROSS VALIDATION TRAINING')
f.close()
torch.cuda.empty_cache()
