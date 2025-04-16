import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, find_model_using_name
import os
import warnings

sys.path.append('./')
warnings.filterwarnings("ignore")
import config
from dataloader_ft import get_ft_loader
from models import *


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False):
    preds, labels = [], []
    dataset = args.dataset
    modality = args.modality
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data_idx, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()  # 重置优化器梯度（仅在训练时）

        ## read dataloader and generate all missing conditions
        audio_feature = data['audio']
        text_feature = data['text']
        video_feature = data['video']
        feature = data[modality]
        labels_ = data['label']

        ## add cuda for tensor
        if cuda:
            # audio_feature = audio_feature.to(device)
            # text_feature = text_feature.to(device)
            # video_feature = video_feature.to(device)
            feature = feature.to(device)
            labels_ = labels_.to(device)

        ## forward
        out, feature_ft = model(feature)

        ## calculate loss
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            loss = cls_loss(out, labels_)
        elif dataset in ['CMUMOSI', 'CMUMOSEI']:
            loss = reg_loss(out, labels_)
        else:
            raise NotImplementedError(f'Error: dataset {dataset} not implemented')

        ## save batch results
        preds.append(out.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        # print(f'---------------{mark} loss: {loss}-------------------')

        if train:  # 训练时，计算梯度并更新参数
            loss.backward()
            optimizer.step()

    assert preds != [], f'Error: no dataset in dataloader'
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        avg_acc = accuracy_score(labels, preds)
        avg_f1 = f1_score(labels, preds, average='weighted')
        return (avg_acc, avg_f1, loss.detach().cpu().numpy(),)

    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])  # remove 0, and remove mask
        avg_acc = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_f1 = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        return (avg_acc, avg_f1, loss.detach().cpu().numpy())

    else:
        raise NotImplementedError(f'Error: dataset {dataset} not implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--model_name', type=str, default=None, help='model name')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=2, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5,
                        help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of num_works')
    parser.add_argument('--modality', type=str, help='modality type')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"


    ## seed
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_torch(args.seed)

    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.n_classes = 1
    elif args.dataset == 'IEMOCAPFour':
        args.n_classes = 4
    elif args.dataset == 'IEMOCAPSix':
        args.n_classes = 6
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    ## reading data
    print(f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(
        video_root), f'features not exist!'
    train_loader, test_loader, adim, tdim, vdim = get_ft_loader(audio_root=audio_root,
                                                                text_root=text_root,
                                                                video_root=video_root,
                                                                num_folder=args.num_folder,
                                                                batch_size=args.batch_size,
                                                                dataset=args.dataset,
                                                                num_workers=args.num_workers)

    print(f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []

    start_time = time.time()

    print('-' * 80)
    print(f'Step1: build model (each folder has its own model)')
    features = {'audio': audio_feature, 'text': text_feature, 'video': video_feature}
    dims = {'audio': adim, 'text': tdim, 'video': vdim}
    model = find_model_using_name(args.model_name)
    modality = args.modality
    dim = dims[modality]
    train_model = model(dim, args.n_classes)
    reg_loss = torch.nn.MSELoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    if cuda:
        train_model.to(device)
    optimizer = optim.Adam([{'params': train_model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
    print('-' * 80)

    print(f'Step2: training (multiple epoches)')
    test_acc, test_f1, test_loss, models = [], [], [], []
    start_first_stage_time = time.time()

    print("------- Starting the first stage! -------")
    for epoch in range(args.epochs):
        _train_acc, _train_f1, _train_loss = train_or_eval_model(args, train_model, reg_loss, cls_loss,
                                                                          train_loader, optimizer, train=True)
        _test_acc, _test_f1, _test_loss = train_or_eval_model(args, train_model, reg_loss, cls_loss,
                                                                       test_loader, train=False)

        ## save
        models.append(train_model)
        test_acc.append(_test_acc)
        test_f1.append(_test_f1)
        test_loss.append(_test_loss)

        print(f'epoch:{epoch}; acc_train: {_train_acc} f1_train:{_train_f1:.3f} loss_train:{_train_loss}')
        print(f'epoch:{epoch}; acc_test:{_test_acc} f1_test:{_test_f1:.3f} loss_test:{_test_loss}')

        print('-' * 10)
        end_first_stage_time = time.time()

    end_second_stage_time = time.time()
    print("-" * 80)
    print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
    print("-" * 80)

    print(f'Step3: load best models and extract features')
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        # best_index_test = np.argmin(np.array(test_loss))
        sorted_indices = np.argsort(np.array(test_loss))  # loss 越小越好
        # sorted_indices = np.argsort(-np.array(test_f1))  # loss 越小越好
        best_index_test = sorted_indices[10]
        best_pred = test_loss[best_index_test]
    elif args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        best_index_test = np.argmax(np.array(test_acc))
        best_pred = test_acc[best_index_test]
    else:
        raise ValueError('dataset not found')
    print(f'best index: {best_index_test}')

    best_video_model = models[best_index_test]

    ## gain feature_name
    feature_name = features[modality]

    end_time = time.time()

    print('-' * 80)

    print(f'====== Saving =======')
    save_model = os.path.join(config.MODEL_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_model): os.makedirs(save_model)
    ## gain suffix_name
    suffix_name = f"{args.dataset}-fine-tuning"
    save_path = f'{save_model}/{suffix_name}-{feature_name}.pth'
    torch.save({'model': best_video_model.state_dict()}, save_path)
    print(save_path)
