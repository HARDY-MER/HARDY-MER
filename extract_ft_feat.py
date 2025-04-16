import os.path

import tqdm
import argparse

from dataloader_ft import TotalIEMOCAP, TotalCMU
from models import *
from torch.utils.data import DataLoader
import config
from utils import find_model_using_name


def extract_iemocap(model, modality, dataloader, device, save_path):
    ## extract features
    print(f'====== Extracting Features =======')
    total_samples = 0
    for batch in tqdm.tqdm(dataloader):
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        video_F = batch['video_F'].to(device)
        video_M = batch['video_M'].to(device)
        if modality == 'video':
            feat_F = batch['video_F'].to(device)
            feat_M = batch['video_M'].to(device)
        else:
            feat = batch[modality].to(device)
        uid = batch['uid']
        with torch.no_grad():
            if modality == 'video':
                _, feat_ft_F = model(feat_F)
                _, feat_ft_M = model(feat_M)
            else:
                _, feat = model(feat)
        for i in range(len(uid)):
            # np.save(os.path.join(audio_save_path, uid[i] + '.npy'), audio_ft[i].cpu().numpy())
            # np.save(os.path.join(text_save_path, uid[i] + '.npy'), text_ft[i].cpu().numpy())
            # if not os.path.exists(os.path.join(video_save_path, uid[i])):
            #     os.makedirs(os.path.join(video_save_path, uid[i]))
            # np.save(os.path.join(video_save_path, uid[i], 'compress_F.npy'), video_ft_F[i].cpu().numpy())
            # np.save(os.path.join(video_save_path, uid[i], 'compress_M.npy'), video_ft_M[i].cpu().numpy())
            if modality == 'video':
                np.save(os.path.join(save_path, uid[i], 'compress_F.npy'), feat_ft_F[i].cpu().numpy())
                np.save(os.path.join(save_path, uid[i], 'compress_M.npy'), feat_ft_M[i].cpu().numpy())
            else:
                np.save(os.path.join(save_path, uid[i] + '.npy'), feat[i].cpu().numpy())
            pass
        total_samples += len(uid)
    print(f'====== Extracted {total_samples} samples =======')


def extract_mosi(model, modality, dataloader, device, save_path):
    ## extract features
    print(f'====== Extracting Features =======')
    total_samples = 0
    for batch in tqdm.tqdm(dataloader):
        feature = batch[modality].to(device)
        uid = batch['uid']
        with torch.no_grad():
            _, feat = model(feature)
        for i in range(len(uid)):
            np.save(os.path.join(save_path, uid[i] + '.npy'), feat[i].cpu().numpy())
        total_samples += len(uid)
    print(f'====== Extracted {total_samples} samples =======')


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
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num-folder', type=int, default=5,
                        help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--num_workers', type=int, default=4, help='number of num_works')
    parser.add_argument('--modality', type=str, help='modality type')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.n_classes = 1
    elif args.dataset == 'IEMOCAPFour':
        args.n_classes = 4
    elif args.dataset == 'IEMOCAPSix':
        args.n_classes = 6
    cuda = torch.cuda.is_available() and not args.no_cuda
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

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = TotalCMU(audio_root, text_root, video_root)
    elif args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        dataset = TotalIEMOCAP(audio_root, text_root, video_root)
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=dataset.collate_fn)

    ## model
    print(f'====== Loading Model =======')
    adim = 512
    tdim = 1024
    vdim = 1024

    features = {'audio': audio_feature, 'text': text_feature, 'video': video_feature}
    dims = {'audio': adim, 'text': tdim, 'video': vdim}
    modality = args.modality

    ModelClass = find_model_using_name(args.model_name)
    model = ModelClass(dims[modality], args.n_classes)
    # 加载模型参数
    save_model = os.path.join(config.MODEL_DIR, 'main_result', f'{args.dataset}')
    ## gain suffix_name
    suffix_name = f"{args.dataset}-fine-tuning"
    ## gain feature_name
    feature_name = features[modality]

    save_path = f'{save_model}/{suffix_name}-{feature_name}.pth'

    model.load_state_dict(torch.load(save_path)['model'])
    # 将模型移动到GPU
    model.to(device)
    # 设置模型为评估模式
    model.eval()

    feat_save_path = os.path.join(config.PATH_TO_FT_FEATURES[args.dataset], feature_name)
    if not os.path.exists(feat_save_path):
        os.makedirs(feat_save_path)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        extract_mosi(model, modality, dataloader, device, feat_save_path)
    elif args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        extract_iemocap(model, modality, dataloader, device, feat_save_path)
