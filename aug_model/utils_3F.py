import os
import sys
import json
import pickle
import math
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


def roc_auc(trues, preds):
    fpr, tpr, thresholds = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    return fpr, tpr, auc


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


@torch.no_grad()
def evaluate(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    val_trues, val_preds,val_preds_arg = [], [],[]
    val_midpre = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, images2, images3, labels = data
        # images1, labels = data
        sample_num += images1.shape[0]  # batch-size

        pred,x1,x2,x3,temp = model(images1.to(device), images2.to(device), images3.to(device))

        # data1 = pred.numpy()
        # np.savetxt("pred.csv", data1)
        # predict = pred.cpu().numpy()
        # plt.imshow(predict)
        # plt.show()

        # pred = model(images1.to(device), images2.to(device), images3.to(device), labels.to(device))
        # pred = model(images1.to(device))
        '''
        待增加功能：输出pred入excel，最好是整个验证集一起输出而不是一个batchsize输出一次
        '''

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # pre.extend(pred.detach().cpu().numpy())
        # output_pre = {'pre': []}
        # output_pre['pre'] = pre
        # outputPre = pd.DataFrame(output_pre)
        # outputPre.to_excel('predextend.xlsx', index=False)

        val_trues.extend(labels.detach().cpu().numpy())
        val_preds.extend(torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy())
        val_preds_arg.extend(torch.argmax(pred, dim=1).detach().cpu().numpy())
        val_midpre.extend(pred.detach().cpu().numpy())
        # output_excel = {'val_trues': [], 'val_preds': []}
        # output_excel['val_trues'] = val_trues
        # output_excel['val_preds'] = val_preds
        # outputEx = pd.DataFrame(output_excel)
        # outputEx.to_excel('val.xlsx', index=False)

        # loss = torch.mean(loss_function(pred, labels.to(device))) + fusion_loss
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

    fpr, tpr, AUC = roc_auc(val_trues, val_preds)

    confu_matrix = confusion_matrix(val_trues, val_preds_arg)
    # print(confu_matrix)
    # np.savetxt("confu.csv", confu_matrix)
    # output_confu = {'confu':[]}
    # output_confu['confu'] = confu_matrix
    # outputCon = pd.DataFrame(list(output_confu.items()))
    # outputCon.to_excel('confumatrix.xlsx', index=False)


    val_acc = accu_num.item() / sample_num
    val_loss = accu_loss.item() / (step + 1)
    return val_loss, val_acc, fpr, tpr, AUC,confu_matrix,val_trues,val_preds,val_midpre,x1,x2,x3,temp


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
