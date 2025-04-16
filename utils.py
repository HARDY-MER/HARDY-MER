import sys
import numpy as np
import torch
from dataloader_iemocap import IEMOCAPDataset
from dataloader_cmumosi import CMUMOSIDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model_expert_softmoe import MoMKE
import torch.nn as nn
import importlib

sys.path.append('./')
import config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers):
    ## CMU datasets
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum + valNum))
        test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))

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
        train_loaders = [train_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim

    ## IEMOCAP dataset
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:  ## five folder cross-validation, each fold contains (train, test)
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
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

        ## gain train and test loaders
        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs),  # random sampler will shuffle index
                                      collate_fn=dataset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=False)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     num_workers=num_workers,
                                     pin_memory=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = MoMKE(args,
                  adim, tdim, vdim, D_e,
                  n_classes=args.n_classes,
                  depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                  attn_drop_rate=args.attn_drop_rate,
                  no_cuda=args.no_cuda)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    return model


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
        model_filename, target_model_name))
        exit(0)

    return model


def create_model(args, model_name, adim, tdim, vdim):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(args)
    """
    model = find_model_using_name(model_name)
    instance = model(args, adim, tdim, vdim, args.hidden,
                     n_classes=args.n_classes,
                     depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                     attn_drop_rate=args.attn_drop_rate)
    print("model [%s] was created" % type(instance).__name__)
    return instance

def generate_mask(seqlen, batch, test_condition, first_stage):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """
    if first_stage:
        audio_mask = np.array([1])
        text_mask = np.array([1])
        visual_mask = np.array([1])
    else:
        audio_mask = np.array([1 if 'a' in test_condition else 0])
        text_mask = np.array([1 if 't' in test_condition else 0])
        visual_mask = np.array([1 if 'v' in test_condition else 0])
    # 重复掩码以匹配数据维度
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask = text_mask.repeat(seqlen * batch)
    visual_mask = visual_mask.repeat(seqlen * batch)

    matrix = [audio_mask, text_mask, visual_mask]
    return matrix


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    # 在特征维度上拼接主视图的音频、文本、图像特征
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2)  # [seqlen, batch, featdim=adim+tdim+vdim]
    # 在特征维度上拼接客视图的音频、文本、图像特征
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    # 获得拼接后的维度
    featdim = feat1.size(-1)
    # 将掩码转置，以匹配特征维度
    tmask = qmask.transpose(0, 1)  # [batch, seqlen] -> [seqlen, batch]
    # 将掩码扩展到特征维度
    tmask = tmask.unsqueeze(2).repeat(1, 1, featdim)  # -> [seqlen, batch, featdim]
    # 根据掩码选择主视图或客视图的特征，这一步的逻辑应该来自与GCNet，不太理解这里怎么做的
    select_feat = torch.where(tmask == 0, feat1, feat2)  # -> [seqlen, batch, featdim]
    input_features.append(select_feat)  # 1 * [seqlen, batch, dim]
    return input_features


################################# 以下是课程学习的内容 ############################################

def calculate_reconstruction_difficulty(original_features, teacher_model):
    reconstructed_features = teacher_model(original_features)  # [seqlen, batch, feat_dim]
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(original_features, reconstructed_features)  # [seqlen, batch, feat_dim]
    reconstruction_difficulty = loss.mean(dim=-1)  # [seqlen, batch]
    return reconstruction_difficulty


def split_features(combined_features, a_dim, t_dim, v_dim):
    seqlen, batch_size, feat_dim = combined_features.shape
    assert feat_dim == a_dim + t_dim + v_dim, "feat_dim 不等于 a_dim + t_dim + v_dim"
    # -->[batch * seq_len, a_dim]
    audio_features = combined_features[:, :, :a_dim].reshape(-1, a_dim).cpu().numpy().astype('float32')
    text_features = combined_features[:, :, a_dim:a_dim + t_dim].reshape(-1, t_dim).cpu().numpy().astype(
        'float32')
    visual_features = combined_features[:, :, a_dim + t_dim:].reshape(-1, v_dim).cpu().numpy().astype(
        'float32')
    return audio_features, text_features, visual_features


def secondary_filtering(retrieved_indices, difficulty, k=0, min_k=1):
    # 因为在检索的时候会检索到自身，所以为了保证检索出自身外K个最相似特征，需要共检索K+1个特征
    _k = (difficulty * (k + 1 - min_k) + min_k).astype(int)
    # print(_k)
    filtered_indices = []
    for i in range(len(retrieved_indices)):
        current_k = _k[i]
        filtered = retrieved_indices[i][:current_k + 1]
        # if current_k < k + 1:
        #     filtered += [retrieved_indices[i][0]] * (k + 1 - current_k)
        filtered_indices.append(filtered)
    return filtered_indices


def reshape_to_original(seqlen, batch_size, concatenated_features):
    reshaped_features = concatenated_features.reshape(seqlen, batch_size, -1)
    return reshaped_features


def retrieve_features_from_indices(filtered_indices_a, filtered_indices_t, filtered_indices_v, index_a, index_t,
                                   index_v):
    retrieved_a = [index_a.reconstruct(int(idx)) for sublist in filtered_indices_a for idx in sublist]
    retrieved_t = [index_t.reconstruct(int(idx)) for sublist in filtered_indices_t for idx in sublist]
    retrieved_v = [index_v.reconstruct(int(idx)) for sublist in filtered_indices_v for idx in sublist]
    retrieved_a = np.array(retrieved_a).reshape(len(filtered_indices_a), -1, index_a.d)
    retrieved_t = np.array(retrieved_t).reshape(len(filtered_indices_t), -1, index_t.d)
    retrieved_v = np.array(retrieved_v).reshape(len(filtered_indices_v), -1, index_v.d)
    return retrieved_a, retrieved_t, retrieved_v


def prepare_training_batches(seqlen, batch_size, k, retrieved_a, retrieved_t, retrieved_v):
    training_batches = []
    n_samples = retrieved_a.shape[0]
    for i in range(n_samples):
        num_a = retrieved_a.shape[1]
        num_t = retrieved_t.shape[1]
        num_v = retrieved_v.shape[1]
        selected_a = retrieved_a[i][:num_a]  # [num_a, a_dim]
        selected_t = retrieved_t[i][:num_t]  # [num_t, t_dim]
        selected_v = retrieved_v[i][:num_v]  # [num_v, v_dim]
        concatenated = np.concatenate([selected_a, selected_t, selected_v], axis=1)  # [num_a, feat_dim]
        concatenated_tensor = torch.tensor(concatenated, dtype=torch.float32)  # [num_a, feat_dim]
        if concatenated_tensor.size(0) != seqlen:
            raise ValueError("序列长度不匹配")
        concatenated_tensor = concatenated_tensor.unsqueeze(1)  # [seqlen, 1, feat_dim]
        training_batches.append(concatenated_tensor)
    training_batches = torch.cat(training_batches, dim=1)  # [seqlen, n_samples, feat_dim]
    return training_batches


def concatenate_and_reshape_samples(retrieved_a, retrieved_t, retrieved_v, seq_len, batch_size, retrieved_labels=None):
    """
    将音频、文本和视觉模态的特征按最后一维拼接，并调整形状为 [seq_len, batch_size, feat]。

    参数：
    - retrieved_a (list of np.ndarray): 音频特征列表，每个元素形状为 [k, feat_a]
    - retrieved_t (list of np.ndarray): 文本特征列表，每个元素形状为 [k, feat_t]
    - retrieved_v (list of np.ndarray): 视觉特征列表，每个元素形状为 [k, feat_v]
    - seq_len (int): 序列长度
    - batch_size (int): 批大小

    返回：
    - retrieved_all (np.ndarray): 拼接后的特征矩阵，形状为 [seq_len, batch_size, feat]
    """
    # 计算每个模态特征的维度
    feat_a = retrieved_a[0].shape[-1]  # 假设每个样本的特征维度相同
    feat_t = retrieved_t[0].shape[-1]
    feat_v = retrieved_v[0].shape[-1]
    feat = feat_a + feat_t + feat_v  # 拼接后的特征维度

    # 拼接所有特征矩阵，假设各自是形状 [n_samples, k, feat_x] 的数组
    retrieved_all = []
    for i in range(len(retrieved_a)):
        # 对每个样本进行拼接
        sample_a = np.array(retrieved_a[i])  # 转换为 NumPy 数组
        sample_t = np.array(retrieved_t[i])
        sample_v = np.array(retrieved_v[i])

        # 在最后一维拼接
        sample_all = np.concatenate([sample_a, sample_t, sample_v], axis=-1)
        retrieved_all.append(sample_all)

    # 将拼接后的结果转换为一个 NumPy 数组，形状为 [n_samples, k, feat_a + feat_t + feat_v]
    retrieved_all = np.array(retrieved_all)

    # 将拼接后的特征矩阵拆分为 [seq_len, batch_size, feat]
    retrieved_all = retrieved_all.reshape(seq_len, batch_size, -1, feat)
    if retrieved_labels is not None:
        _retrieved_labels = np.array(retrieved_labels)# -1 自动计算最后一维的大小
        _retrieved_labels = _retrieved_labels.reshape(seq_len, batch_size, -1)
        return retrieved_all, _retrieved_labels
    else:
        return retrieved_all


def compute_samplewise_mutual_information_batch(A, B, bandwidth=1.0, n_components=5):
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    """
    批量计算 A 和 B 的逐样本互信息
    A: [batch_size, feature_dim]
    B: [batch_size, feature_dim]
    返回：长度为 batch_size 的互信息数组
    """
    assert A.shape == B.shape, "A 和 B 的形状必须相同"  # 假设 A 和 B 是 PyTorch 张量

    A = A.reshape(-1, A.shape[-1])
    B = B.reshape(-1, B.shape[-1])
    A = A.cpu().detach().numpy()  # 将 A 从 PyTorch 张量转换为 NumPy 数组
    B = B.cpu().detach().numpy()  # 将 B 从 PyTorch 张量转换为 NumPy 数组
    # batch_size, feature_dim = A.shape

    # 联合特征 [batch_size, 2 * feature_dim]
    # AB = np.hstack([A, B])    # 降维
    n_components = min(A.shape[-1], n_components)

    pca = PCA(n_components=n_components)
    A_reduced = pca.fit_transform(A)
    B_reduced = pca.fit_transform(B)

    AB_reduced = np.hstack([A_reduced, B_reduced])

    # KDE 对联合分布进行估计
    kde_joint = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(AB_reduced)
    log_p_ab = kde_joint.score_samples(AB_reduced)  # [batch_size]

    # KDE 对边缘分布进行估计
    kde_a = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(A_reduced)
    log_p_a = kde_a.score_samples(A_reduced)  # [batch_size]

    kde_b = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(B_reduced)
    log_p_b = kde_b.score_samples(B_reduced)  # [batch_size]

    # 使用互信息公式计算
    mis = log_p_ab - log_p_a - log_p_b
    return mis


def mutual_information(feat_a, feat_b, feat_joint):
    h_a = information_entropy(feat_a)
    h_b = information_entropy(feat_b)
    h_ab = information_entropy(feat_joint)
    mi = h_a + h_b - h_ab
    return mi


def information_entropy(feat, _tao=1, omega=1e-6, epsilon=1e-7):
    # 检查输入是否包含 NaN 或 Inf
    # if torch.isnan(feat).any() or torch.isinf(feat).any():
    #     raise ValueError("输入特征包含 NaN 或 Inf 值。请检查输入数据。")
    if torch.isnan(feat).any():
        raise ValueError("输入特征包含 NaN 值。请检查输入数据。")
    if torch.isinf(feat).any():
        raise ValueError("输入特征包含 Inf 值。请检查输入数据。")
    # 计算特征向量的信息熵
    # feat: [batch_size, seq_len, feature_dim]
    # return: [batch_size, seq_len]
    # 计算概率分布
    prob = torch.softmax(feat / _tao, dim=-1)
    prob = torch.clamp(prob, min=omega, max=1.0)  # 确保概率在 [omega, 1] 之间
    # 计算对数概率
    log_prob = torch.log(prob + epsilon)  # 防止 log(0)
    # 计算信息熵
    entropy = -torch.sum(prob * log_prob, dim=-1)
    return entropy


def top_k_accuracy(samples):
    query_label = samples[0]
    retrieved_labels = samples[1:]

    # Top-K Accuracy: 判断是否至少命中一次
    acc = int(query_label in retrieved_labels)

    return acc


def recall_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    """
    y_true: shape (batch_size, num_labels) -> binary multi-hot ground truth
    y_scores: shape (batch_size, num_labels) -> predicted scores
    """
    topk = torch.topk(y_scores, k=k, dim=1).indices  # (batch_size, k)
    relevant = y_true.gather(1, topk)  # (batch_size, k), only keep relevant items in topk
    hits = relevant.sum(dim=1).float()  # (batch_size,)
    total_relevant = y_true.sum(dim=1).float()  # (batch_size,)
    recall = torch.where(total_relevant > 0, hits / total_relevant, torch.zeros_like(hits))
    return recall.mean().item()

def compute_topk_acc_and_recall(samples: list):

    query_label = samples[0]
    retrieved_labels = samples[1:]

    # Top-K Accuracy: 判断是否至少命中一次
    acc = int(query_label in retrieved_labels)

    # Recall@K: 看命中了多少个
    recall = sum(1 for label in retrieved_labels if label == query_label)

    return acc, recall