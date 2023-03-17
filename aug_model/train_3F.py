import os
import argparse
import sys
import random

import torch
import torch.optim as optim
# from medcam import medcam
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from my_dataset_fusion import MyDataSet
# from my_dataset import MyDataSet
from LeNetAtt import LeNetAtt as create_model
# from LeNet5_3d import LeNet as create_model
from utils_3F import create_lr_scheduler, get_params_groups, evaluate, roc_auc
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def main(args):
    name = args.name    #获取参数表中的name
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  #选择GPU或CPU设备
    print(f"using {device} device.")

    data_path_train1 = args.data_path_train1
    data_path_train2 = args.data_path_train2
    data_path_train3 = args.data_path_train3
    data_path_train4 = args.data_path_train4

    data_path_val1 = args.data_path_val1
    data_path_val2 = args.data_path_val2
    data_path_val3 = args.data_path_val3
    data_path_val4 = args.data_path_val4

    train_txt_path=args.train_txt_path
    val_txt_path=args.val_txt_path

    # 测试单期像
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor()

        ]),
        "val": transforms.Compose([
            transforms.ToTensor()
        ])}

    # 实例化训练数据集
    # 三时期期像融合
    train_dataset = MyDataSet(data_path_train1, data_path_train2, data_path_train3, train_txt_path)
    # 单期像训练
    # train_dataset = MyDataSet(data_path_train4, train_txt_path, transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(data_path_val1, data_path_val2, data_path_val3,  val_txt_path)
    # 单期像验证
    # val_dataset = MyDataSet(data_path_val4, val_txt_path, transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw=0
    print('Using {} dataloader workers every process'.format(nw))

    # def setup_seed(seed):
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #
    # # 设置随机数种子
    # setup_seed(20)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    # model_3D = medcam.inject(model, output_dir='E:/Bayer/data/figures/AP', save_maps=True)

    if args.weights != "":
        #是否采用预训练的权重，默认为否
        #若使用的话，在arg.weight中写权重的路径
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 因为预训练模型不一定是做2分类，故删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        # 是否冻结权重，在训练时某些层被冻结则其参数不变，默认为否
        for name_, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name_:
                para.requires_grad_(False)
            else:
                print("training {}".format(name_))

    pg = get_params_groups(model, weight_decay=args.wd) #获取模型参数
    lr=args.lr  #获取学习率
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    optimizer = optim.SGD(pg, lr=lr, momentum=args.mm,weight_decay=args.wd)  #采用SGD和momentum来炼丹

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1,end_factor=1e-3)
    '''
    采用了warmup策略以及学习率衰减来继续炼丹
    每一个lr_scheduler.step()执行一次
    '''

    if os.path.exists("./weights_5") is False:    #设置权重保存路径
        os.makedirs("./weights_5")

    max_iter=args.max_iter
    best_acc = 0
    iter_num = 0
    val_acc_, val_loss_, confu_matrix_ = [], [], []
    train_acc_, train_loss_, train_confumatrix_, train_AUC_ = [], [], [],[]
    train_trues, train_preds, train_preds_arg = [], [], []
    train_midpre = []
    fpr_, tpr_, AUC_ = [], [], []
    val_trues_, val_preds_ = [], []
    val_midpre_ = []
    train_x1_,train_x2_,train_x3_,train_pred_ = [],[],[],[]
    val_x1_,val_x2_,val_x3_,val_pred_ = [],[],[],[]
    for epoch in range(args.epochs):
        # train
        model.train()

        loss_function = torch.nn.CrossEntropyLoss()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        train_data_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_data_loader):
            images1, images2, images3, labels = data
            # images1, labels = data
            sample_num += images1.shape[0]  # batch-size

            # pred, loss = model(images1.to(device), images2.to(device), images3.to(device), labels.to(device))
            pred,x1,x2,x3,temp = model(images1.to(device), images2.to(device), images3.to(device))
            # pred = model(images1.to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            train_trues.extend(labels.detach().cpu().numpy())
            train_preds.extend(torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy())
            train_preds_arg.extend(torch.argmax(pred, dim=1).detach().cpu().numpy())
            train_midpre.extend(pred.detach().cpu().numpy())

            # loss = torch.mean(loss_function(pred, labels.to(device)))+fusion_loss
            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()

            train_loss = accu_loss.item() / (step + 1)
            train_acc = accu_num.item() / sample_num

            train_data_loader.desc = "[train iteration {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                iter_num,
                train_loss,
                train_acc,
                optimizer.param_groups[0]["lr"]
            )

            if not torch.isfinite(loss):  # 如果loss飙升，则停掉模型
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            # update lr
            lr_scheduler.step()

            iter_num = iter_num + 1

            # if iter_num%200==0:
            #     lr_ = lr * 0.92
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            '''
            lr_scheduler.step()的方式衰减貌似更好
            '''
            train_acc_.append(train_acc)
            train_loss_.append(train_loss)
            train_x1_.append(x1)
            train_x2_.append(x2)
            train_x3_.append(x3)
            train_pred_.append(temp)

            result_path = 'trainRuns_5'
            if os.path.exists(result_path) == False:
                os.makedirs(result_path)

            np.savez(result_path + '/result_' + name + '.npz', train_acc=train_acc_, train_loss=train_loss_,
                     train_AUC=train_AUC_, train_confumatrix=train_confumatrix_,  train_trues=train_trues, train_preds=train_preds,
                     train_midpre=train_midpre,x1=train_x1_,x2=train_x2_,x3=train_x3_,pred=train_pred_)

            if iter_num % 10 == 0:  #每10个迭代验证一次

                #validation
                val_data_loader = tqdm(val_loader)
                val_loss, val_acc, fpr, tpr, AUC, confu_matrix,val_trues,val_preds,val_midpre,val_x1,val_x2,val_x3,valpred = evaluate(model, val_data_loader, device)

                # output = model_3D(pred)

                val_loss_.append(val_loss)
                val_acc_.append(val_acc)
                confu_matrix_.append(confu_matrix)
                fpr_.append(fpr)
                tpr_.append(tpr)
                AUC_.append(AUC)
                val_trues_.append(val_trues)
                val_preds_.append(val_preds)
                val_midpre_.append(val_midpre)
                val_x1_.append(val_x1)
                val_x2_.append(val_x2)
                val_x3_.append(val_x3)
                val_pred_.append(valpred)

                print("[val iteration {}] loss: {:.3f}, acc: {:.3f}, AUC: {:.3f}".format(iter_num, val_loss, val_acc, AUC))

                if best_acc <= val_acc:  # 最好的acc时保存模型权重
                    torch.save(model.state_dict(), "./weights_5/" + name + ".pth")
                    best_acc = val_acc

            if iter_num==max_iter:  #无视epoch，达到最大iteration时停止，迭代数，样本量，batcsize三者数量是相关的
                break
        if iter_num == max_iter:
            print('present epoch : '+str(epoch) + ', ended by max iteration break')
            break

        train_fp_list, train_tp_list, train_roc_auc = roc_curve(train_trues, train_preds)
        trian_confumatrix = confusion_matrix(train_trues, train_preds_arg)
        train_confumatrix_.append(trian_confumatrix)
        train_AUC_.append(train_roc_auc)

    result_path = 'runs_5'
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)

    np.savez(result_path + '/result_' + name + '.npz', train_acc=train_acc_, train_loss=train_loss_, train_AUC=train_AUC_,train_confumatrix=train_confumatrix_, val_acc=val_acc_, val_loss=val_loss_, val_confu_matrix=confu_matrix_,
                val_AUC=AUC_, val_fpr=fpr_, val_tpr=tpr_, val_trues=val_trues_, val_preds=val_preds_,val_midpre=val_midpre_,x1=val_x1_,x2=val_x2_,x3=val_x3_,pred=val_pred_)



if __name__ == '__main__':
    time = '5'
    fold = '3'
    phase = 'fusion_3P'
    # phase = 'T1WI'

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=20001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-3)
    # parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    # parser.add_argument('--wd', type=float, default=2e-1)
    parser.add_argument('--mm', type=float, default=0.9)
    parser.add_argument('--data_path_train1', type=str,
                        default=r'H:\Bayer\data\augmentation/AP')
    parser.add_argument('--data_path_train2', type=str,
                        default=r'H:\Bayer\data\augmentation/HBP')
    parser.add_argument('--data_path_train3', type=str,
                        default=r'H:\Bayer\data\augmentation/PP')
    parser.add_argument('--data_path_train4', type=str,
                        default=r'H:\Bayer\data\augmentation/T1WI')
    parser.add_argument('--data_path_val1', type=str,
                        default=r'H:\Bayer\data\val/AP')
    parser.add_argument('--data_path_val2', type=str,
                        default=r'H:\Bayer\data\val/HBP')
    parser.add_argument('--data_path_val3', type=str,
                        default=r'H:\Bayer\data\val/PP')
    parser.add_argument('--data_path_val4', type=str,
                        default=r'H:\Bayer\data\val/T1WI')
    parser.add_argument('--train_txt_path', type=str,
                        default=r'H:\Bayer\bbox_ing\aug_model\txt\train_' + fold + '.txt')
    parser.add_argument('--val_txt_path', type=str,
                        default=r'H:\Bayer\bbox_ing\aug_model\txt\val_' + fold + '.txt')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--name', type=str, default=time+'_'+fold+'_'+phase)
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
