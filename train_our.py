import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import *
from loss import MaskedCELoss, MaskedMSELoss, MaskedReconLoss
import os
import warnings
from retrieval import *
import faiss
from models import *

sys.path.append('./')
warnings.filterwarnings("ignore")
import config
import torch.nn as nn
import time


def train_or_eval_model(args, model, t_model, reg_loss, cls_loss, rec_loss, dataloader, optimizer=None, train=False,
                        first_stage=True, indexs=None):
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses, losses1, losses2, losses3 = [], [], [], []
    preds_test_condition = []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda
    top_1_acc, top_3_acc, top_5_acc = [], [], []
    case_study = {'high_hard': None, 'low_hard': None}

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data_idx, data in enumerate(dataloader):
        vidnames = []
        if train: optimizer.zero_grad()  # 重置优化器梯度（仅在训练时）

        ## read dataloader and generate all missing conditions
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]        主讲者
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]     客讲者
        qmask: speakers, [batch, seqlen]        说话人掩码
        umask: has utt, [batch, seqlen]         句子掩码
        label: [batch, seqlen]       标签
        数据包含主讲者（host）和客讲者（guest）的音频、文本、视觉信息，以及掩码和标签。
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames += data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        ## using cmp-net masking manner [at least one view exists]
        ## host mask
        ## 是六种缺失条件
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)  # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)  # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        ## add cuda for tensor
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)

        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3]
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                              audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
        '''
        # masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        '''
        _labels = None
        if train and not first_stage:
            # if 't' not in args.test_condition:
            features = masked_input_features[0]
            # 加载索引
            index_a, index_t, index_v, metadata = indexs[0], indexs[1], indexs[2], indexs[-1]
            # 设置检索参数
            k = args.top_k

            # 分离特征
            a_dim, t_dim, v_dim = audio_host.size(-1), text_host.size(-1), visual_host.size(-1)
            audio_features, text_features, visual_features = split_features(features, a_dim, t_dim,
                                                                            v_dim)  # [n_samples, dim]
            # 检索可用模态相似的样本，然后将检索的索引进行整合
            integrate_indices = search_and_integrate(audio_features, text_features, visual_features, index_a, index_t,
                                                     index_v, k, test_condition=args.test_condition, index_type="FlatL2")


            # # 利用教师模型计算特征重构难度
            out = t_model(features, input_features_mask[0], umask, first_stage)
            # # 计算重构难度
            rec_feats = out['rec_feats']
            mi_at, mi_av, mi_tv = out['mi_at'], out['mi_av'], out['mi_tv']
            total_mi = mi_at + mi_av + mi_tv
            _, loss_rec_batch = rec_loss(rec_feats, masked_input_features[0], input_features_mask[0], umask, adim, tdim,
                                         vdim)
            # 对loss_rec_batch和total_mi进行归一化
            eps = 1e-5
            loss_rec_batch = loss_rec_batch.detach().cpu().numpy()
            diff_rec = np.zeros_like(loss_rec_batch)  # 先创建全零数组
            range_val = loss_rec_batch.max() - loss_rec_batch.min()
            if range_val > eps:  # 只有 range_val 不是零时才进行归一化
                diff_rec = (loss_rec_batch - loss_rec_batch.min()) / (range_val + eps)
            diff_mi = (total_mi - total_mi.min()) / (total_mi.max() - total_mi.min())
            diff_mi = diff_mi.detach().cpu().numpy()
            # 计算总的学习难度，并将难度放缩到[0, 1]区间
            total_diff = 1 / (
                    1 + np.exp(-args.beta * (args.alpha_1 * diff_rec + args.alpha_2 * diff_mi)))  # [batch, seqlen]
            total_diff = total_diff.reshape(-1)

            # 根据困难度对样本进行筛选，并重新检索特征
            integrate_indices = secondary_filtering(integrate_indices, total_diff, k, min_k=1)
            retrieved_features, retrieved_labels, retrieved_filenames = integrated_features(integrate_indices, index_a, index_t, index_v, k, metadata)
            retrieved_a, retrieved_t, retrieved_v = retrieved_features['a'], retrieved_features['t'], \
                retrieved_features['v']

            for item in retrieved_labels:
                top_k_acc = top_k_accuracy(item)
                if len(item) <= 1 + 1:
                    top_1_acc.append(top_k_acc)
                if len(item) <= 3 + 1:
                    top_3_acc.append(top_k_acc)
                if len(item) <= 5 + 1:
                    top_5_acc.append(top_k_acc)

            # 保存case_study的数据
            high_hard_idx = np.argmax(total_diff)
            high_hard = total_diff[high_hard_idx]
            high_labels = retrieved_labels[high_hard_idx]
            high_filenames = retrieved_filenames[high_hard_idx]
            if case_study['high_hard'] is None:
                case_study['high_hard'] = [high_hard]
                case_study['high_labels'] = [high_labels]
                case_study['high_filenames'] = [high_filenames]
            else:
                if high_hard > case_study['high_hard']:
                    case_study['high_hard'] = [high_hard]
                    case_study['high_labels'] = [high_labels]
                    case_study['high_filenames'] = [high_filenames]

            low_hard_idx = np.argmin(total_diff)
            low_hard = total_diff[low_hard_idx]
            low_labels = retrieved_labels[low_hard_idx]
            low_filenames = retrieved_filenames[low_hard_idx]
            if case_study['low_hard'] is None:
                case_study['low_hard'] = [low_hard]
                case_study['low_labels'] = [low_labels]
                case_study['low_filenames'] = [low_filenames]
            else:
                if low_hard < case_study['low_hard']:
                    case_study['low_hard'] = [low_hard]
                    case_study['low_labels'] = [low_labels]
                    case_study['low_filenames'] = [low_filenames]

            # 将检索到的特征整合为一个列表，每个元素是[batch, seqlen, dim]的格式
            concatenated_features = concatenate_and_reshape_samples(retrieved_a,
                                                                    retrieved_t, retrieved_v, seqlen, batch)
            masked_input_features = [torch.tensor(concatenated_features[:, :, i, :], dtype=torch.float32).to(device) for
                                     i in range(concatenated_features.shape[-2])]

        # 将文本特征进行L2归一化
        a_dim, t_dim, v_dim = audio_host.size(-1), text_host.size(-1), visual_host.size(-1)
        for i in range(len(masked_input_features)):
            features = masked_input_features[i]
            seq_len, batch_size, _ = features.shape
            audio_feature, text_feature, video_feature = split_features(features, a_dim, t_dim, v_dim)
            faiss.normalize_L2(text_feature)
            audio_feature = torch.tensor(audio_feature.reshape(seq_len, batch, a_dim)).to(device)
            text_feature = torch.tensor(text_feature.reshape(seq_len, batch, t_dim)).to(device)
            video_feature = torch.tensor(video_feature.reshape(seq_len, batch, v_dim)).to(device)
            masked_input_features[i] = torch.cat([audio_feature, text_feature, video_feature], dim=-1)

        ## forward
        for i in range(len(masked_input_features)):
            out = model(masked_input_features[i], input_features_mask[0],
                                                                  umask, first_stage)

            ## calculate loss
            lp_ = out['out'].view(-1, out['out'].size(2))  # [batch*seq_len, n_classes]
            lp_a, lp_t, lp_v = (out['out_a'].view(-1, out['out_a'].size(2)),
                                out['out_t'].view(-1, out['out_t'].size(2)),
                                out['out_v'].view(-1, out['out_v'].size(2)))


            labels_ = label.reshape(-1)  # [batch*seq_len]
            if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
                if first_stage:
                    loss_a = cls_loss(lp_a, labels_, umask)
                    loss_t = cls_loss(lp_t, labels_, umask)
                    loss_v = cls_loss(lp_v, labels_, umask)
                else:
                    loss = cls_loss(lp_, labels_, umask)
                    loss_rec, _ = rec_loss(out['rec_feats'], masked_input_features[i], input_features_mask[0],
                                           umask, model.adim, model.tdim, model.vdim)
            if dataset in ['CMUMOSI', 'CMUMOSEI']:
                if first_stage:
                    loss_a = reg_loss(lp_a, labels_, umask)
                    loss_t = reg_loss(lp_t, labels_, umask)
                    loss_v = reg_loss(lp_v, labels_, umask)
                else:
                    loss = reg_loss(lp_, labels_, umask)
                    loss_rec, _ = rec_loss(out['rec_feats'], masked_input_features[i], input_features_mask[0],
                                           umask, model.adim, model.tdim, model.vdim)

            ## save batch results
            preds_a.append(lp_a.data.cpu().numpy())
            preds_t.append(lp_t.data.cpu().numpy())
            preds_v.append(lp_v.data.cpu().numpy())
            preds.append(lp_.data.cpu().numpy())
            labels.append(labels_.data.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())

            # print(f'---------------{mark} loss: {loss}-------------------')
            preds_test_condition.append(out['out'].view(-1, out['out'].size(2)).data.cpu().numpy())

            if train and first_stage:
                total_loss = loss_a + loss_t + loss_v
                total_loss.backward()
                optimizer.step()
            if train and not first_stage:
                total_loss = loss + loss_rec
                total_loss.backward()
                optimizer.step()
    if train and not first_stage:
        print(f'top_1_acc: {np.mean(top_1_acc)}, top_3_acc: {np.mean(top_3_acc)}, top_5_acc: {np.mean(top_5_acc)}')
        print(f'case_study is: {case_study}')

    assert preds != [], f'Error: no dataset in dataloader'
    preds = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        avg_loss = round(np.sum(losses) / np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss

    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])  # remove 0, and remove mask
        avg_loss = round(np.sum(losses) / np.sum(masks), 4)
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
        avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
        avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')
    parser.add_argument('--teacher_model', type=str, default=None, help='which teacher model will be loaded')
    # parser.add_argument('--teacher_model_path', type=str, default=None, help='where the teacher model will be loaded')    # 在config里配置
    parser.add_argument('--student_model', type=str, default=None, help='which student model will be loaded')

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
    parser.add_argument('--test_condition', type=str, default='atv', choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv'],
                        help='test conditions')
    parser.add_argument('--stage_epoch', type=float, default=100, help='number of epochs of the first stage')
    parser.add_argument('--top_k', type=int, default=3, help='how many top-k to select')
    parser.add_argument('--beta', type=float, default=4)
    parser.add_argument('--alpha_1', type=float, default=0.6)
    parser.add_argument('--alpha_2', type=float, default=0.4)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f'Using device: {device}')
    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    sys.stdout = Logger(
        filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_test-condition-{args.test_condition}.txt",
        stream=sys.stdout)


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
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    ## reading data
    print(f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FT_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FT_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FT_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(
        video_root), f'features not exist!'
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                text_root=text_root,
                                                                video_root=video_root,
                                                                num_folder=args.num_folder,
                                                                batch_size=args.batch_size,
                                                                dataset=args.dataset,
                                                                num_workers=0)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    print(f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    for ii in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on the {ii + 1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('-' * 80)
        print(f'Step1: build model and indexes (each folder has its own model and indexes)')
        model = create_model(args, args.student_model, adim, tdim, vdim)
        # 加载教师模型
        teacher_model = create_model(args, args.teacher_model, adim, tdim, vdim)
        teacher_model = teacher_model.to(device)
        teacher_state_dict_root = os.path.join(config.MODEL_DIR, f'{args.teacher_model}', f'{save_folder_name}',
                                               f'{args.test_condition}')
        if not os.path.exists(teacher_state_dict_root): raise 'Error: teacher model not exist!'
        teacher_state_dict_file = glob.glob(
            os.path.join(teacher_state_dict_root, f'{ii + 1}_{args.teacher_model}_*.pth'))
        print(f'teacher_model_path is: {teacher_state_dict_file}')
        # 制定教师模型加载到对应的GPU上
        teacher_model.load_state_dict(torch.load(teacher_state_dict_file[0], map_location=device))
        teacher_model.eval()
        print(f'teacher model has been removed to {next(teacher_model.parameters()).device}')

        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        rec_loss = MaskedReconLoss()
        if cuda:
            model.to(device)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])

        # 加载索引
        index_a, index_t, index_v = load_index(args.dataset, folds=ii, gpu_id=args.gpu)
        fold_pkl = load_pkl(args.dataset, folds=ii)
        # fold_idx2label = {item[0]: item[2] for item in fold_pkl}
        print('-' * 80)

        print(f'Step2: training (multiple epoches)')
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        models = []
        total_MI = []  # 记录每个epoch的MI
        total_rec_loss = []
        start_first_stage_time = time.time()

        print("------- Starting the first stage! -------")
        for epoch in range(args.epochs):
            first_stage = True if epoch < args.stage_epoch else False
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss = train_or_eval_model(
                args, model, teacher_model, reg_loss, cls_loss, rec_loss, train_loader, \
                optimizer=optimizer, train=True, first_stage=first_stage, indexs=[index_a, index_t, index_v, fold_pkl])
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss = train_or_eval_model(
                args, model, teacher_model, reg_loss, cls_loss, rec_loss, test_loader, \
                optimizer=None, train=False, first_stage=first_stage)

            ## save
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            models.append(model)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])

            if first_stage:
                print(
                    f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(
                    f'epoch:{epoch}; a_acc_test:{train_acc_atv[0]:.3f}; t_acc_test:{train_acc_atv[1]:.3f}; v_acc_test:{train_acc_atv[2]:.3f}')
            else:
                print(
                    f'epoch:{epoch}; train_mae_{args.test_condition}:{train_mae:.3f}; train_corr_{args.test_condition}:{train_corr:.3f}; train_fscore_{args.test_condition}:{train_fscore:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}; train_loss_{args.test_condition}:{train_loss}')
                print(
                    f'epoch:{epoch}; test_mae_{args.test_condition}:{test_mae:.3f}; test_corr_{args.test_condition}:{test_corr:.3f}; test_fscore_{args.test_condition}:{test_fscore:2.2%}; test_acc_{args.test_condition}:{test_acc:2.2%}; test_loss_{args.test_condition}:{test_loss}')
            print('-' * 10)
            ## update the parameter for the 2nd stage
            # 可以在这里建立index
            if epoch == args.stage_epoch - 1:
                model = models[-1]

                model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
                print(f'best_epoch_a: {model_idx_a}')
                model_a = models[model_idx_a]
                transformer_a_para_dict = {k: v for k, v in model_a.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_a_para_dict)

                model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
                print(f'best_epoch_t: {model_idx_t}')
                model_t = models[model_idx_t]
                transformer_t_para_dict = {k: v for k, v in model_t.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_t_para_dict)

                model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
                print(f'best_epoch_v: {model_idx_v}')
                model_v = models[model_idx_v]
                transformer_v_para_dict = {k: v for k, v in model_v.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_v_para_dict)

                end_first_stage_time = time.time()
                print("------- Starting the second stage! -------")

        end_second_stage_time = time.time()
        print("-" * 80)
        print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print("-" * 80)

        print(f'Step3: saving and testing on the {ii + 1} folder')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_index_test = np.argmax(np.array(test_fscores))
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            best_index_test = np.argmax(np.array(test_accs))

        bestmae = test_maes[best_index_test]
        bestcorr = test_corrs[best_index_test]
        bestf1 = test_fscores[best_index_test]
        bestacc = test_accs[best_index_test]
        bestmodel = models[best_index_test]

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        end_time = time.time()

        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(
                f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.")
            print(f'>>>>> Finish: training on the {ii + 1} folder, duration: {end_time - start_time} >>>>>')
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(
                f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
            print(f'>>>>> Finish: training on the {ii + 1} folder, duration: {end_time - start_time} >>>>>')

    print('-' * 80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(
            f"Folder avg: test_condition ({args.test_condition}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        print(
            f"Folder avg: test_condition ({args.test_condition}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")

    print(f'====== Saving =======')
    save_model = os.path.join(config.MODEL_DIR, 'main_result', args.student_model, f'{save_folder_name}')
    if not os.path.exists(save_model): os.makedirs(save_model)
    ## gain suffix_name
    suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
    ## gain feature_name
    feature_name = f'{audio_feature};{text_feature};{video_feature}'
    ## gain res_name
    mean_mae = np.mean(np.array(folder_mae))
    mean_corr = np.mean(np.array(folder_corr))
    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        res_name = f'mae-{mean_mae:.3f}_corr-{mean_corr:.3f}_acc-{mean_acc:.4f}_f1-{mean_f1:.4f}'
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        res_name = f'acc-{mean_acc:.4f}_ua-{mean_corr:.4f}'
    save_path = f'{save_model}/{suffix_name}_features-{feature_name}_{res_name}_test-condition-{args.test_condition}.pth'
    torch.save({'model': model.state_dict()}, save_path)
    print(save_path)
