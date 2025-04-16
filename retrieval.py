import faiss
import numpy as np
import torch
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # 导入tqdm库
import glob
import pickle
import config


def get_video_feat_path(feat_path):
    video_feat_path = None
    feature_name = os.path.basename(feat_path)
    speaker = feature_name.split('_')[-1][0]

    facenames = os.listdir(feat_path)
    for facename in sorted(facenames):
        assert facename.find('F') >= 0 or facename.find('M') >= 0
        if facename.find(speaker) >= 0:
            video_feat_path = (os.path.join(feat_path, facename))
    return video_feat_path


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_pkl(dataset, root):
    import pickle

    # for IEMOCAP
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        ## gain index for cross-validation
        with open(root, 'rb') as f:
            videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(f)
        f.close()
        vids = sorted(list(trainVid | testVid))

        num_folder = 5
        session_to_idx = {}
        for idx, vid in enumerate(vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder):  # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        # print(videoIDs)
        total_train_idxs = []
        total_train_labels = []
        for i in range(len(train_test_idxs)):
            train_idxs, _ = train_test_idxs[i]
            flod_train = []
            flod_train_labels = []
            for item in train_idxs:
                vid = vids[item]
                uid = videoIDs[vid]
                labels = videoLabels[vid]
                flod_train.extend(uid)
                flod_train_labels.extend(labels)
            total_train_idxs.append(flod_train)
            total_train_labels.append(flod_train_labels)
    else:  # for MOSI and MOSEI
        with open(root, 'rb') as f:
            videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids = pickle.load(f)
        f.close()
        names = []
        for ii, vid in enumerate(videoIDs):
            uids_video = videoIDs[vid]
            names.extend(uids_video)
            # total_train_labels.append(labels)
        total_train_idxs = []
        train_idxs = []
        total_train_labels = []
        train_labels = []
        for trainVid in trainVids:
            labels = videoLabels[trainVid]
            # labels_num = len(labels)
            _i = 0
            for name in names:
                train_name = name.split('_')[0]
                if trainVid == train_name:
                    # print(f'name is: {name}, trainVid is: {trainVid}, label is: {labels}, _i is: {_i}')
                    train_idxs.append(name)
                    train_labels.append(labels[_i])
                    _i += 1
        total_train_idxs.append(train_idxs)
        total_train_labels.append(train_labels)
    # print(total_train_labels)
    # print(np.array(total_train_idxs).shape, np.array(total_train_labels).shape)
    for i in range(len(total_train_idxs)):
        print(
            f'第{i + 1}折的标签数量为：{len(total_train_labels[i])}, 第{i + 1}折的对话数量为：{len(total_train_idxs[i])}')
    return total_train_idxs, total_train_labels


def get_root(dataset, isFt=True, label=False):
    assert dataset in ['CMUMOSEI', 'CMUMOSI', 'IEMOCAPFour', 'IEMOCAPSix'], \
        'dataset must be IEMOCAPFour, IEMOCAPSix, CMUMOSEI or CMUMOSI'
    if isFt:
        audio_feat_root = os.path.join(config.PATH_TO_FT_FEATURES[dataset], 'wav2vec-large-c-UTT')
        text_feat_root = os.path.join(config.PATH_TO_FT_FEATURES[dataset], 'deberta-large-4-UTT')
        video_feat_root = os.path.join(config.PATH_TO_FT_FEATURES[dataset], 'manet_UTT')
        index_root = config.PATH_TO_FT_INDEX[dataset]
    else:
        audio_feat_root = os.path.join(config.PATH_TO_FEATURES[dataset], 'wav2vec-large-c-UTT')
        text_feat_root = os.path.join(config.PATH_TO_FEATURES[dataset], 'deberta-large-4-UTT')
        video_feat_root = os.path.join(config.PATH_TO_FEATURES[dataset], 'manet_UTT')
        index_root = config.PATH_TO_INDEX[dataset]
    pkl_root = config.PATH_TO_LABEL[dataset]
    if not label:
        return audio_feat_root, text_feat_root, video_feat_root, pkl_root, index_root
    else:
        label_root = os.path.join(config.PATH_TO_FT_FEATURES[dataset], 'label')
        return audio_feat_root, text_feat_root, video_feat_root, pkl_root, index_root, label_root


def build_index(dataset, adim=512, tdim=1024, vdim=1024, index_type="FlatL2", isFt=True):
    assert dataset in ['CMUMOSEI', 'CMUMOSI', 'IEMOCAPFour', 'IEMOCAPSix'], \
        'dataset must be IEMOCAPFour, IEMOCAPSix, CMUMOSEI or CMUMOSI'
    # 设置三种模态的文件夹路径
    audio_feat_root, text_feat_root, video_feat_root, pkl_root, index_root = get_root(dataset, isFt)
    # 获取k折数据的文件名
    print(f'=========================检查{dataset}_Fold=============================')
    fold_files, fold_labels = check_pkl(dataset, pkl_root)
    for i in range(0, len(fold_files)):
        print(f'=========================第{i + 1}折index开始创建=============================')
        print(f'=========================  1 - 获取对话文件名称和对应的label  =============================')
        fold_list = fold_files[i]
        fold_label = fold_labels[i]
        metadata={}
        print(f'=========================  2 - 创建index及存储路径=========================')
        fold_index_root = os.path.join(index_root, f'{i}')
        check_dir(fold_index_root)
        audio_index = faiss.IndexFlatL2(adim)
        video_index = faiss.IndexFlatL2(vdim)
        if index_type == "FlatL2":
            text_index = faiss.IndexFlatL2(tdim)
        elif index_type == "FlatIP":  # 使用余弦相似度
            text_index = faiss.IndexFlatIP(tdim)
        else:
            raise NotImplementedError(f'index_type {index_type} not implemented')
        print(f'=========================  3 - 加载不同模态特征=========================')
        for idx, filename in tqdm(enumerate(fold_list)):
            audio_feat_path = os.path.join(audio_feat_root, f'{filename}.npy')
            text_feat_path = os.path.join(text_feat_root, f'{filename}.npy')
            # 视频特征的路径与其他模态不同，需要额外处理
            if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
                video_feat_path = os.path.join(video_feat_root, filename)
                video_feat_path = get_video_feat_path(video_feat_path)
            else:
                video_feat_path = os.path.join(video_feat_root, f'{filename}.npy')
            # print(f'=========================  4 - 加载模态数据=========================')
            audio_feat = np.load(audio_feat_path).astype('float32').reshape(1, -1)
            text_feat = np.load(text_feat_path).astype('float32').reshape(1, -1)
            video_feat = np.load(video_feat_path).astype('float32').reshape(1, -1)
            # 同步建立样本名称与标签的映射
            metadata[idx] = {
                'filename': filename,
                'label': fold_label[idx]
            }
            # 特征归一化，为构建基于余弦相似度的索引做准备
            if index_type == "FlatIP":  # 使用余弦相似度
                faiss.normalize_L2(text_feat)
            # 将特征添加到索引中
            audio_index.add(audio_feat)
            text_index.add(text_feat)
            video_index.add(video_feat)
        print(f'=========================  5 - 保存索引=========================')
        faiss.write_index(audio_index, os.path.join(fold_index_root, 'wav2vec-large-c-UTT.index'))
        faiss.write_index(text_index, os.path.join(fold_index_root, 'deberta-large-4-UTT.index'))
        faiss.write_index(video_index, os.path.join(fold_index_root, 'manet_UTT.index'))
        # 保存索引对应的样本和标签
        with open(os.path.join(fold_index_root, 'metadata.pkl'), "wb") as f:
            pickle.dump(metadata, f)
        print(f'=========================第{i + 1}折index创建完成=============================')


def search_and_integrate(audio_features, text_features, visual_features, index_a, index_t, index_v, k=0,
                         test_condition='avt', index_type="FlatL2"):
    _audio_features = audio_features
    _text_features = text_features
    _visual_features = visual_features
    if index_type == "FlatIP":  # 如果使用的是 IndexFlatIP，需要先归一化
        faiss.normalize_L2(_text_features)
    n_samples = audio_features.shape[0]  # 假设所有模态的 n_samples 相同
    retrieved_indices = [[] for _ in range(n_samples)]  # 初始化每个样本的索引列表

    # 进行检索并收集索引
    # 因为在检索的时候会检索到自身，所以为了保证检索K个最相似特征，需要共检索K+1个特征
    if 'a' in test_condition:
        distances_a, indices_a = index_a.search(_audio_features, k + 1)  # [n_samples, k+1], [n_samples, k+1]
        for i in range(n_samples):
            retrieved_indices[i].extend(indices_a[i].tolist())

    if 'v' in test_condition:
        distances_v, indices_v = index_v.search(_visual_features, k + 1)  # [n_samples, k+1], [n_samples, k+1]
        for i in range(n_samples):
            retrieved_indices[i].extend(indices_v[i].tolist())

    if 't' in test_condition:
        distances_t, indices_t = index_t.search(_text_features, k + 1)  # [n_samples, k+1], [n_samples, k+1]
        for i in range(n_samples):
            retrieved_indices[i].extend(indices_t[i].tolist())

    # 整合索引，去除重复
    integrated_indices = []  # [n_samples, k]
    for idx_list in retrieved_indices:
        seen = set()
        unique_indices = []
        for idx in idx_list:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        # 确保至少保留一个索引
        if len(unique_indices) == 0:
            unique_indices.append(idx_list[0])  # 保留第一个索引
        integrated_indices.append(unique_indices)

    return integrated_indices


def integrated_features(integrated_indices, index_a, index_t, index_v, k=5, metadata=None):
    retrieved_features = {'a': [], 't': [], 'v': []}
    labels = []
    filenames = []
    for sample_indices in integrated_indices:
        d = {}
        label = []
        filename = []
        # 初始化每个模态的特征列表
        sample_features_a = []
        sample_features_t = []
        sample_features_v = []
        a_0 = index_a.reconstruct(int(sample_indices[0]))
        t_0 = index_t.reconstruct(int(sample_indices[0]))
        v_0 = index_v.reconstruct(int(sample_indices[0]))
        # 添加被检索样本的label
        sample_features_a.append(a_0)
        sample_features_t.append(t_0)
        sample_features_v.append(v_0)
        label.append(metadata[sample_indices[0]]['label'])
        filename.append(metadata[sample_indices[0]]['filename'])
        for idx in sample_indices[1: ]:
            feature_a = index_a.reconstruct(int(idx))
            feature_v = index_v.reconstruct(int(idx))
            feature_t = index_t.reconstruct(int(idx))
            a_idx_dis = np.linalg.norm(feature_a - a_0)
            t_idx_dis = np.linalg.norm(feature_t - t_0)
            v_idx_dis = np.linalg.norm(feature_v - v_0)
            mean_dis = (a_idx_dis + t_idx_dis + v_idx_dis) / 3
            if len(d) < k + 1:
                d[idx] = mean_dis
            else:
                max_key = max(d, key=d.get)  # 找到当前字典中值最大的键
                if mean_dis < d[max_key]:  # 如果新距离小于最大距离
                    del d[max_key]  # 删除值最大的键值对
                    d[idx] = mean_dis  # 添加新键值对
        ######################################################################
        # 这里的填充策略需要变一下
        for idx in d.keys():
            sample_features_a.append(index_a.reconstruct(int(idx)))
            sample_features_v.append(index_v.reconstruct(int(idx)))
            sample_features_t.append(index_t.reconstruct(int(idx)))
            label.append(metadata[int(idx)]['label'])
            filename.append(metadata[int(idx)]['filename'])

        # # 从后面开始填充
        # if len(sample_features_a) < k + 1:
        #     sample_features_a += [a_0] * (k + 1 - len(sample_features_a))
        #     sample_features_v += [v_0] * (k + 1 - len(sample_features_v))
        #     sample_features_t += [t_0] * (k + 1 - len(sample_features_t))
        # 从前面开始填充
        if len(sample_features_a) < k + 1:
            sample_features_a = [a_0] * (k + 1 - len(sample_features_a)) + sample_features_a
            sample_features_v = [v_0] * (k + 1 - len(sample_features_v)) + sample_features_v
            sample_features_t = [t_0] * (k + 1 - len(sample_features_t)) + sample_features_t
        ######################################################################
        # 将特征列表转换为 NumPy 数组并添加到结果中
        retrieved_features['a'].append(np.array(sample_features_a))
        retrieved_features['v'].append(np.array(sample_features_v))
        retrieved_features['t'].append(np.array(sample_features_t))
        labels.append(label)
        filenames.append(filename)

    return retrieved_features, labels, filenames


def load_index(dataset, folds, gpu_id, isFt=True):
    # 创建一个StandardGpuResources对象，用于管理GPU资源
    res = faiss.StandardGpuResources()
    torch.cuda.set_device(gpu_id)

    # 定义索引的路径
    if isFt:
        audio_index_path = os.path.join(config.PATH_TO_FT_INDEX[dataset], f'{folds}', f'wav2vec-large-c-UTT.index')
        text_index_path = os.path.join(config.PATH_TO_FT_INDEX[dataset], f'{folds}', f'deberta-large-4-UTT.index')
        video_index_path = os.path.join(config.PATH_TO_FT_INDEX[dataset], f'{folds}', f'manet_UTT.index')
    else:
        audio_index_path = os.path.join(config.PATH_TO_INDEX[dataset], f'{folds}', f'wav2vec-large-c-UTT.index')
        text_index_path = os.path.join(config.PATH_TO_INDEX[dataset], f'{folds}', f'deberta-large-4-UTT.index')
        video_index_path = os.path.join(config.PATH_TO_INDEX[dataset], f'{folds}', f'manet_UTT.index')

    print(audio_index_path)
    print(text_index_path)
    print(video_index_path)

    # 读取索引
    audio_index = faiss.read_index(audio_index_path)
    text_index = faiss.read_index(text_index_path)
    video_index = faiss.read_index(video_index_path)
    print('All indexes have been loaded!')

    # # 将索引移到GPU上
    audio_index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, audio_index)
    text_index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, text_index)
    video_index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, video_index)
    print(f'All index has been removed to {torch.cuda.current_device()}')

    return audio_index_gpu, text_index_gpu, video_index_gpu


def load_pkl(dataset, folds):
    index_root = config.PATH_TO_FT_INDEX[dataset]
    fold_index_root = os.path.join(index_root, f'{folds}')
    with open(os.path.join(fold_index_root, 'metadata.pkl'), "rb") as f:
        metadata = pickle.load(f)
    return metadata


if __name__ == '__main__':
    # # # 建立各个数据集的检索目录 index
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Build index for retrieval.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--index_type", type=str, help="Type of the index to build")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数创建index
    print(args.dataset, args.index_type)
    build_index(args.dataset, index_type=args.index_type)
