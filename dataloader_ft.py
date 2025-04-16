import os
import time
import glob
import tqdm
import pickle
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import SubsetRandomSampler
from models import FineTuning
from torch.utils.data import DataLoader


## gain name2features
def read_data(label_path, feature_root):
    ## gain (names, speakers)
    names = []
    if 'IEMOCAP' in label_path:
        videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(open(label_path, "rb"),
                                                                                             encoding='latin1')
        vids = sorted(list(trainVid | testVid))
        for ii, vid in enumerate(vids):
            uids_video = videoIDs[vid]
            names.extend(uids_video)
    else:
        videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, valVids, testVid = pickle.load(
            open(label_path, "rb"), encoding='latin1')
        for ii, vid in enumerate(videoIDs):
            uids_video = videoIDs[vid]
            names.extend(uids_video)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        feature = []
        if 'IEMOCAP' in label_path:
            feature_dir = glob.glob(os.path.join(feature_root, name + '*'))
            assert len(feature_dir) == 1
            feature_path = feature_dir[0]
        else:
            feature_path = os.path.join(feature_root, name + '.npy')
            feature_dir = os.path.join(feature_root, name)

        if feature_path.endswith('.npy'):  # audio/text => belong to speaker
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze()  # [Dim, ] or [Time, Dim]
            feature.append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else:  ## exists dir, faces => belong to speaker in 'facename'
            facenames = os.listdir(feature_path)
            for facename in sorted(facenames):
                if 'IEMOCAP' in label_path:
                    assert facename.find('F') >= 0 or facename.find('M') >= 0
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature.append(facefeat)

        single_feature = np.array(feature).squeeze()
        if len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)
        feature = single_feature

        features.append(feature)

    ## save (names, features)
    print(f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]

    return name2feats, feature_dim


class IEMOCAPDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root):

        ## read utterance feats
        self.name2audio, self.adim = read_data(label_path, audio_root)
        self.name2text, self.tdim = read_data(label_path, text_root)
        self.name2video, self.vdim = read_data(label_path, video_root)

        ## gain video feats
        self.max_len = -1
        self.videoLabelsNew = {}
        self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences, self.trainVid, self.testVid = pickle.load(
            open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(list(self.trainVid | self.testVid))
        for ii, vid in enumerate(self.vids):
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            self.max_len = max(self.max_len, len(uids))
            for ii, uid in enumerate(uids):
                self.videoLabelsNew[uid] = labels[ii]
        self.uids = sorted(list(self.videoLabelsNew.keys()))
        self.trainUids = []
        self.testUids = []
        for ii, vid in enumerate(self.trainVid):
            self.trainUids += self.videoIDs[vid]
        for ii, vid in enumerate(self.testVid):
            self.testUids += self.videoIDs[vid]
        assert len(self.uids) == len(self.name2audio.keys()) == len(self.name2text.keys()) == len(
            self.name2video.keys()), \
            'length not match'

    ## return host(A, T, V) and guest(A, T, V)
    def __getitem__(self, index):
        uid = self.uids[index]
        return torch.FloatTensor(self.name2audio[uid]), torch.FloatTensor(self.name2text[uid]), \
            torch.FloatTensor(self.name2video[uid]), torch.tensor(self.videoLabelsNew[uid]), uid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        A, T, V, labels, uids = [], [], [], [], []
        for i, d in enumerate(data):
            audio_feature, text_feature, video_feature, label, uid = d
            A.append(audio_feature)
            T.append(text_feature)
            V.append(video_feature)
            labels.append(label)
            uids.append(uid)
        A = torch.stack(A)
        T = torch.stack(T)
        V = torch.stack(V)
        # print(labels)
        labels = torch.stack(labels)
        return {
            'audio': A,
            'text': T,
            'video': V,
            'label': labels,
            'uid': uids
        }


class CMUMOSIDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root):

        ## read utterance feats
        self.name2audio, self.adim = read_data(label_path, audio_root)
        self.name2text, self.tdim = read_data(label_path, text_root)
        self.name2video, self.vdim = read_data(label_path, video_root)

        ## gain video feats
        self.max_len = -1
        self.videoLabelsNew = {}
        self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences, self.trainVids, self.valVids, self.testVids = pickle.load(
            open(label_path, "rb"), encoding='latin1')

        self.vids = []
        for vid in sorted(self.trainVids): self.vids.append(vid)
        for vid in sorted(self.valVids): self.vids.append(vid)
        for vid in sorted(self.testVids): self.vids.append(vid)

        for ii, vid in enumerate(sorted(self.videoIDs)):
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]

            self.max_len = max(self.max_len, len(uids))
            for ii, uid in enumerate(uids):
                self.videoLabelsNew[uid] = labels[ii]

        self.uids = sorted(self.videoLabelsNew.keys())
        self.trainUids = []
        self.valUids = []
        self.testUids = []
        for ii, vid in enumerate(self.trainVids):
            self.trainUids += self.videoIDs[vid]
        for ii, vid in enumerate(self.valVids):
            self.valUids += self.videoIDs[vid]
        for ii, vid in enumerate(self.testVids):
            self.testUids += self.videoIDs[vid]
        print(f'train: {len(self.trainUids)}; val: {len(self.valUids)}; test: {len(self.testUids)}')
        assert len(self.trainUids) + len(self.valUids) + len(self.testUids) == len(self.uids)

    ## return host(A, T, V) and guest(A, T, V)
    def __getitem__(self, index):
        uid = self.uids[index]
        return torch.tensor(self.name2audio[uid], dtype=torch.float32), torch.tensor(self.name2text[uid],
                                                                                     dtype=torch.float32), \
            torch.tensor(self.name2video[uid], dtype=torch.float32), torch.tensor(self.videoLabelsNew[uid],
                                                                                  dtype=torch.float32), uid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        A, T, V, labels, uids = [], [], [], [], []
        for i, d in enumerate(data):
            audio_feature, text_feature, video_feature, label, uid = d
            A.append(audio_feature)
            T.append(text_feature)
            V.append(video_feature)
            labels.append(label)
            uids.append(uid)
        A = torch.stack(A)
        T = torch.stack(T)
        V = torch.stack(V)
        # print(labels)
        labels = torch.stack(labels)
        return {
            'audio': A,
            'text': T,
            'video': V,
            'label': labels,
            'uid': uids
        }


# 两个TotalDataset仅用于读取数据，不做训练使用
class TotalIEMOCAP(Dataset):
    def __init__(self, audio_root, text_root, video_root):
        # 既要保证能读取到所有特征，又要保证记录所有特征的名字，用字典应该是做合适的
        self.name2audio = {}
        self.name2text = {}
        self.name2video_F = {}  # compress_F
        self.name2video_M = {}  # compress_M
        for file in os.listdir(audio_root):  # 读取所有特征
            feature = np.load(os.path.join(audio_root, file))
            self.name2audio[file.split('.')[0]] = feature
        for file in os.listdir(text_root):
            feature = np.load(os.path.join(text_root, file))
            self.name2text[file.split('.')[0]] = feature
        for file_path in os.listdir(video_root):
            F_feature = np.load(os.path.join(video_root, file_path, 'compress_F.npy'))
            M_feature = np.load(os.path.join(video_root, file_path, 'compress_M.npy'))
            self.name2video_F[file_path] = F_feature
            self.name2video_M[file_path] = M_feature
        self.names = sorted(list(self.name2audio.keys()))

    def __getitem__(self, index):
        name = self.names[index]
        return torch.tensor(self.name2audio[name], dtype=torch.float32), \
            torch.tensor(self.name2text[name], dtype=torch.float32), \
            torch.tensor(self.name2video_F[name], dtype=torch.float32), \
            torch.tensor(self.name2video_M[name], dtype=torch.float32), \
            name

    def __len__(self):
        return len(self.names)

    def collate_fn(self, data):
        A, T, V_F, V_M, uids = [], [], [], [], []
        for i, d in enumerate(data):
            A.append(d[0])
            T.append(d[1])
            V_F.append(d[2])
            V_M.append(d[3])
            uids.append(d[4])
        A = torch.stack(A)
        T = torch.stack(T)
        V_F = torch.stack(V_F)
        V_M = torch.stack(V_M)
        return {
            'audio': A,
            'text': T,
            'video_F': V_F,
            'video_M': V_M,
            'uid': uids
        }


class TotalCMU(Dataset):
    def __init__(self, audio_root, text_root, video_root):
        self.name2audio = {}
        self.name2text = {}
        self.name2video = {}
        for file in os.listdir(audio_root):
            feature = np.load(os.path.join(audio_root, file))
            self.name2audio[file.split('.')[0]] = feature
        for file in os.listdir(text_root):
            feature = np.load(os.path.join(text_root, file))
            self.name2text[file.split('.')[0]] = feature
        for file in os.listdir(video_root):
            feature = np.load(os.path.join(video_root, file))
            self.name2video[file.split('.')[0]] = feature
        self.names = sorted(list(self.name2audio.keys()))

    def __getitem__(self, index):
        name = self.names[index]
        return torch.tensor(self.name2audio[name], dtype=torch.float32), \
            torch.tensor(self.name2text[name], dtype=torch.float32), \
            torch.tensor(self.name2video[name], dtype=torch.float32), \
            name

    def __len__(self):
        return len(self.names)

    def collate_fn(self, data):
        A, T, V, uids = [], [], [], []
        for i, d in enumerate(data):
            A.append(d[0])
            T.append(d[1])
            V.append(d[2])
            uids.append(d[3])
        A = torch.stack(A)
        T = torch.stack(T)
        V = torch.stack(V)
        return {
            'audio': A,
            'text': T,
            'video': V,
            'uid': uids
        }


def get_ft_loader(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers):
    ## CMU datasets
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum + valNum))
        # val_idxs = list(range(trainNum, trainNum + valNum))
        test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))
    ## IEMOCAP dataset
    elif dataset in ['IEMOCAPFour', 'IEMOCAPSix']:  ## five folder cross-validation, each fold contains (train, test)
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        train_num = len(dataset.trainUids)
        test_num = len(dataset.testUids)
        train_idxs = list(range(0, train_num))
        test_idxs = list(range(train_num, train_num + test_num))
        print(
            f'train idxs: {len(train_idxs)}, test idxs: {len(test_idxs)}, total idxs: {len(train_idxs) + len(test_idxs)}')

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idxs),
                              collate_fn=dataset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=False)
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=SubsetRandomSampler(test_idxs),
                             collate_fn=dataset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=False)

    ## return loaders
    adim, tdim, vdim = dataset.get_featDim()
    return train_loader, test_loader, adim, tdim, vdim


if __name__ == '__main__':
    # 加载Dataloader
    from torch.utils.data import DataLoader

    audio_root = os.path.join(config.PATH_TO_FEATURES['IEMOCAPFour'], 'wav2vec-large-c-UTT')
    text_root = os.path.join(config.PATH_TO_FEATURES['IEMOCAPFour'], 'deberta-large-4-UTT')
    video_root = os.path.join(config.PATH_TO_FEATURES['IEMOCAPFour'], 'manet_UTT')
    label_path = config.PATH_TO_LABEL['IEMOCAPFour']

    # dataset = IEMOCAPDataset(label_path, audio_root, text_root, video_root)
    train_loader, test_loader, adim, tdim, vidm = get_ft_loader(
        audio_root, text_root, video_root, 1, 'IEMOCAPFour', 8, 4)

    for i, data in enumerate(train_loader):
        audio_feature = data['audio']
        text_feature = data['text']
        video_feature = data['video']
        label = data['label']
        uid = data['uid']
        print(f'audio feature shape is: {audio_feature.shape},'
              f' text feature shape is: {text_feature.shape},'
              f' video feature shape is: {video_feature.shape},'
              f' label shape is: {label.shape}')
        if i > 5:
            break
