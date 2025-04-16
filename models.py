import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention_softmoe import *
from utils import compute_samplewise_mutual_information_batch, mutual_information
import math


class EasyReconstructModel(nn.Module):
    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0,
                 attn_drop_rate=0):
        """
        初始化多模态重构模型。

        参数：
        - feat_a (int): 音频特征维度。默认 1024。
        - feat_t (int): 文本特征维度。默认 512。
        - feat_v (int): 视觉特征维度。默认 1024。
        - encoded_dim_a (int): 音频编码器输出维度。默认 512。
        - encoded_dim_t (int): 文本编码器输出维度。默认 256。
        - encoded_dim_v (int): 视觉编码器输出维度。默认 512。
        - autoencoder_encoded_dim (int): 自编码器编码器输出维度。默认 512。
        """
        super(EasyReconstructModel, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        D = 3 * D_e
        self.num_heads = num_heads
        self.device = args.device
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        # 模态编码器，将不同模态特征投影到公共空间
        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        # Dropout，防止过拟合
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)
        # 模态互信息映射模块
        self.a_in_mi = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.t_in_mi = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.v_in_mi = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        # 模态交叉注意力模块
        # a-t
        self.cross_attn_at = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_ta = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        # a-v
        self.cross_attn_av = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_va = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        # t-v
        self.cross_attn_tv = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_vt = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        #  Expert模块
        self.block = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)
        # 自编码器解码器
        self.autoencoder_decoder = nn.Sequential(
            nn.Linear(D, 1024),
            nn.ReLU(),
            nn.Linear(1024, adim + tdim + vdim)
        )

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        input_features_mask -> ?*[seqlen, batch, 3]
        umask -> [batch, seqlen]
        """
        # 分割输入特征为三个模态
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
            inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # 编码每个模态
        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)
        # --> [batch, 3*seqlen, dim]
        # 对音频、文本和视觉投影后的特征分别应用 Transformer 块。self.block对应文中的Expert，三种模态共用一个Expert。
        x_a = self.block(proj_a, first_stage, attn_mask, 'a')
        x_t = self.block(proj_t, first_stage, attn_mask, 't')
        x_v = self.block(proj_v, first_stage, attn_mask, 'v')

        # 计算模态间的互信息
        # _proj_a, _proj_t, _proj_v = proj_a.clone(), proj_t.clone(), proj_v.clone()
        # mi_a = self.a_in_mi(_proj_a, first_stage, attn_mask, 'a')
        # mi_t = self.t_in_mi(_proj_t, first_stage, attn_mask, 't')
        # mi_v = self.v_in_mi(_proj_v, first_stage, attn_mask, 'v')
        #
        # mi_at = compute_samplewise_mutual_information_batch(mi_a, mi_t)
        # mi_av = compute_samplewise_mutual_information_batch(mi_a, mi_v)
        # mi_tv = compute_samplewise_mutual_information_batch(mi_t, mi_v)
        # 计算跨模态表征
        if first_stage:
            mi_a, mi_t, mi_v = x_a.clone(), x_t.clone(), x_v.clone()
        else:
            mi_a, mi_t, mi_v = x_a.clone(), x_t.clone(), x_v.clone()
            mi_a = mi_a.reshape(B, seq_len, 3, self.D_e)
            mi_t = mi_t.reshape(B, seq_len, 3, self.D_e)
            mi_v = mi_v.reshape(B, seq_len, 3, self.D_e)
            mi_a = torch.sum(mi_a, dim=2)
            mi_t = torch.sum(mi_t, dim=2)
            mi_v = torch.sum(mi_v, dim=2)

        feat_at = self.cross_attn_at(mi_a, mi_t, mi_t, mask_modality_a='a', mask_modality_b='t', mask=attn_mask)
        feat_ta = self.cross_attn_ta(mi_t, mi_a, mi_a, mask_modality_a='t', mask_modality_b='a', mask=attn_mask)
        joint_at = feat_at + feat_ta
        feat_av = self.cross_attn_va(mi_a, mi_v, mi_v, mask_modality_a='a', mask_modality_b='v', mask=attn_mask)
        feat_va = self.cross_attn_av(mi_v, mi_a, mi_a, mask_modality_a='v', mask_modality_b='a', mask=attn_mask)
        joint_av = feat_av + feat_va
        feat_tv = self.cross_attn_tv(mi_t, mi_v, mi_v, mask_modality_a='t', mask_modality_b='v', mask=attn_mask)
        feat_vt = self.cross_attn_vt(mi_v, mi_t, mi_t, mask_modality_a='v', mask_modality_b='t', mask=attn_mask)
        joint_tv = feat_tv + feat_vt
        # if not first_stage:
        #     print(f'feat_at is: {feat_at}, feat_ta is: {feat_ta}, mi_a is: {mi_a}, mi_t is: {mi_t}')
        #     print(f'feat_av is: {feat_av}, feat_va is: {feat_va}, mi_a is: {mi_a}, mi_v is: {mi_v}')
        #     print(f'feat_tv is: {feat_tv}, feat_vt is: {feat_vt}, mi_t is: {mi_t}, mi_v is: {mi_v}')
        # 计算互信息
        mi_at = mutual_information(mi_a, mi_t, joint_at)  # [batch, seqlen]
        mi_av = mutual_information(mi_a, mi_v, joint_av)  # [batch, seqlen]
        mi_tv = mutual_information(mi_t, mi_v, joint_tv)  # [batch, seqlen]
        # print(f'mi_at: {mi_at.mean()}, mi_av: {mi_av.mean()}, mi_tv: {mi_tv.mean()}')

        # 直接将不同模态的输出拿去分类
        if first_stage:
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            # meaningless
            out_a, out_t, out_v = torch.rand((B, seq_len, self.n_classes)), torch.rand(
                (B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))

            x_unweighted_a = x_a.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_t = x_t.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_v = x_v.reshape(B, seq_len, 3, self.D_e)
            x_out_a = torch.sum(x_unweighted_a, dim=2)
            x_out_t = torch.sum(x_unweighted_t, dim=2)
            x_out_v = torch.sum(x_unweighted_v, dim=2)
            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)

        x[attn_mask == 0] = 0
        # 将拼接后的特征 x 分割回音频、文本和视觉三个模态的特征
        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2 * seq_len, :], x[:, 2 * seq_len:, :]
        # 将音频、文本和视觉的特征沿特征维度拼接在一起，得到 x_joint
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        # 投影和残差连接，得到隐藏层向量
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        # 将隐藏层向量输入到分类头，得到最终的分类输出
        out = self.nlp_head(hidden)

        # 利用隐藏层特征，重构inputfeats
        _hidden = hidden.permute(1, 0, 2)  # [seqlen, batch, dim]
        rec_feats = self.autoencoder_decoder(_hidden)

        return {
            'out': out,  # 分类输出
            'out_a': out_a,  # 音频分类输出
            'out_t': out_t,  # 文本分类输出
            'out_v': out_v,  # 视觉分类输出
            'rec_feats': rec_feats,  # 重构特征
            'mi_at': mi_at,  # 音频-文本互信息
            'mi_av': mi_av,  # 音频-视觉互信息
            'mi_tv': mi_tv  # 文本-视觉互信息
        }


class TeacherModel(nn.Module):
    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0,
                 attn_drop_rate=0):
        super(TeacherModel, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        D = 3 * D_e
        self.num_heads = num_heads
        self.device = args.device
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        # 将不同模态特征投影到公共空间
        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))

        # 模态编码器
        self.encoder_a = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.encoder_t = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.encoder_v = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        # Dropout，防止过拟合
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)
        # 模态互信息映射模块
        self.a_in_mi = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.t_in_mi = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.v_in_mi = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        # 模态交叉注意力模块
        # a-t
        self.cross_attn_at = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_ta = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        # a-v
        self.cross_attn_av = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_va = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        # t-v
        self.cross_attn_tv = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
        self.cross_attn_vt = CrossAttention(D_e, num_heads, mlp_ratio, drop_rate, attn_drop_rate)

        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)
        # 自编码器解码器
        self.autoencoder_decoder = nn.Sequential(
            nn.Linear(D, 1024),
            nn.ReLU(),
            nn.Linear(1024, adim + tdim + vdim)
        )

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        input_features_mask -> ?*[seqlen, batch, 3]
        umask -> [batch, seqlen]
        """
        # 对输入进行预处理
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
            inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape
        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # 编码每个模态
        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        x_a = self.encoder_a(proj_a, attn_mask, 'a')
        x_t = self.encoder_t(proj_t, attn_mask, 't')
        x_v = self.encoder_v(proj_v, attn_mask, 'v')

        # 计算跨模态表征
        mi_a, mi_t, mi_v = x_a.clone(), x_t.clone(), x_v.clone()

        feat_at = self.cross_attn_at(mi_a, mi_t, mi_t, mask_modality_a='a', mask_modality_b='t', mask=attn_mask)
        feat_ta = self.cross_attn_ta(mi_t, mi_a, mi_a, mask_modality_a='t', mask_modality_b='a', mask=attn_mask)
        joint_at = feat_at + feat_ta
        feat_av = self.cross_attn_va(mi_a, mi_v, mi_v, mask_modality_a='a', mask_modality_b='v', mask=attn_mask)
        feat_va = self.cross_attn_av(mi_v, mi_a, mi_a, mask_modality_a='v', mask_modality_b='a', mask=attn_mask)
        joint_av = feat_av + feat_va
        feat_tv = self.cross_attn_tv(mi_t, mi_v, mi_v, mask_modality_a='t', mask_modality_b='v', mask=attn_mask)
        feat_vt = self.cross_attn_vt(mi_v, mi_t, mi_t, mask_modality_a='v', mask_modality_b='t', mask=attn_mask)
        joint_tv = feat_tv + feat_vt

        # 计算互信息
        mi_at = mutual_information(mi_a, mi_t, joint_at)  # [batch, seqlen]
        mi_av = mutual_information(mi_a, mi_v, joint_av)  # [batch, seqlen]
        mi_tv = mutual_information(mi_t, mi_v, joint_tv)  # [batch, seqlen]

        # 直接将不同模态的输出拿去分类
        if first_stage:
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            # 随机生成一些模态输出，保证forward返回结果的统一性
            out_a, out_t, out_v = torch.rand((B, seq_len, self.n_classes)), torch.rand(
                (B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))
            x = torch.cat([x_a, x_t, x_v], dim=1)

        x[attn_mask == 0] = 0
        # 将拼接后的特征 x 分割回音频、文本和视觉三个模态的特征
        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2 * seq_len, :], x[:, 2 * seq_len:, :]
        # 将音频、文本和视觉的特征沿特征维度拼接在一起，得到 x_joint
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        # 投影和残差连接，得到隐藏层向量
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        # 将隐藏层向量输入到分类头，得到最终的分类输出
        out = self.nlp_head(hidden)

        # 利用隐藏层特征，重构inputfeats
        _hidden = hidden.permute(1, 0, 2)  # [seqlen, batch, dim]
        rec_feats = self.autoencoder_decoder(_hidden)

        return {
            'out': out,  # 分类输出
            'out_a': out_a,  # 音频分类输出
            'out_t': out_t,  # 文本分类输出
            'out_v': out_v,  # 视觉分类输出
            'rec_feats': rec_feats,  # 重构特征
            'mi_at': mi_at,  # 音频-文本互信息
            'mi_av': mi_av,  # 音频-视觉互信息
            'mi_tv': mi_tv  # 文本-视觉互信息
        }


# 微调特征
class FineTuning(nn.Module):
    def __init__(self, input_size, output_size):
        super(FineTuning, self).__init__()
        # 两层微调，三层分类
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, int(input_size / 2))
        self.fc4 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc5 = nn.Linear(int(input_size / 4), output_size)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.output_size = output_size

    def forward(self, input):
        # x 和 y 的形状均为 [batch, dim]
        out = self.relu(self.fc1(input))
        feature = self.fc2(out)
        out = self.relu(feature)
        out = self.relu(self.fc3(out))  # [batch, input_size/2]
        out = self.relu(self.fc4(out))  # [batch, input_size/4]
        out = self.fc5(out)  # [batch, output_size]
        if self.output_size > 1:
            out = self.softmax(out)
        return out, feature


class CMUVideoFineTuning(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 两层微调，三层分类
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, int(input_size / 2))
        self.fc4 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc5 = nn.Linear(int(input_size / 4), output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_size = output_size

    def forward(self, input):
        # x 和 y 的形状均为 [batch, dim]
        out = self.fc1(input)
        out = self.dropout(out)
        out = self.relu(out)
        feature = self.fc2(out)
        out = self.relu(feature)
        out = self.relu(self.fc3(out))  # [batch, input_size/2]
        out = self.relu(self.fc4(out))  # [batch, input_size/4]
        out = self.fc5(out)  # [batch, output_size]
        return out, feature


class CMUTextFineTuning(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 增加层数和Batch Normalization
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, int(input_size / 2))
        self.fc4 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc5 = nn.Linear(int(input_size / 4), output_size)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.batch_norm2 = nn.BatchNorm1d(input_size)
        self.batch_norm3 = nn.BatchNorm1d(int(input_size / 2))

        self.output_size = output_size

    def forward(self, input):
        # 使用Batch Normalization和不同激活函数
        out = self.fc1(input)
        out = self.batch_norm1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        feature = self.fc2(out)
        out = self.batch_norm2(feature)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)  # 输出层
        return out, feature


class CMUAudioFineTuning(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 增加层数和Batch Normalization
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, int(input_size / 2))
        self.fc4 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc5 = nn.Linear(int(input_size / 4), output_size)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.batch_norm2 = nn.BatchNorm1d(input_size)
        self.batch_norm3 = nn.BatchNorm1d(int(input_size / 2))

        self.output_size = output_size

    def forward(self, input):
        # 使用Batch Normalization和不同激活函数
        out = self.fc1(input)
        feature = out
        out = self.batch_norm1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        # feature = self.fc2(out)
        # out = self.batch_norm2(feature)
        # out = self.relu(out)

        out = self.fc3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)  # 输出层
        return out, feature

class OurModel(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0,
                 attn_drop_rate=0, no_cuda=False):
        super(OurModel, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        self.block = OurBlock(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)
        self.autoencoder_decoder = nn.Sequential(
            nn.Linear(D, 1024),
            nn.ReLU(),
            nn.Linear(1024, adim + tdim + vdim)
        )

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        input_features_mask -> ?*[seqlen, batch, 3]
        umask -> [batch, seqlen]
        """
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
            inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # --> [batch, 3*seqlen, dim]
        # 对音频、文本和视觉投影后的特征分别应用 Transformer 块。self.block对应文中的Expert，三种模态共用一个Expert。
        x_a = self.block(proj_a, attn_mask, 'a')
        x_t = self.block(proj_t, attn_mask, 't')
        x_v = self.block(proj_v, attn_mask, 'v')
        if first_stage:
            # 直接将不同模态的输出拿去分类
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            # meaningless
            # 生成随机的分类输出 out_a、out_t、out_v，作为占位符
            out_a, out_t, out_v = torch.rand((B, seq_len, self.n_classes)), torch.rand(
                (B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))
            # 将音频、文本和视觉的投影特征重新整形为 [batch, seqlen, 3, D_e]，以便与权重进行逐元素相乘。
            x_out_a = x_a.reshape(B, seq_len, self.D_e)
            x_out_t = x_t.reshape(B, seq_len, self.D_e)
            x_out_v = x_v.reshape(B, seq_len, self.D_e)
            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)

        x[attn_mask == 0] = 0
        # 将拼接后的特征 x 分割回音频、文本和视觉三个模态的特征
        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2 * seq_len, :], x[:, 2 * seq_len:, :]
        # 将音频、文本和视觉的特征沿特征维度拼接在一起，得到 x_joint
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        # 投影和残差连接，得到隐藏层向量
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        # 将隐藏层向量输入到分类头，得到最终的分类输出
        out = self.nlp_head(hidden)

        if not first_stage:  # 如果不是第一阶段，则使用自编码器解码器对隐藏层向量进行解码，得到重构特征
            _hidden = hidden.permute(1, 0, 2)  # -->[seqlen, batch, dim]
            rec_feats = self.autoencoder_decoder(_hidden)

            return {'out': out,
                    'out_a': out_a,
                    'out_t': out_t,
                    'out_v': out_v,
                    'rec_feats': rec_feats
                    }
        else:
            return {'out': out,
                    'out_a': out_a,
                    'out_t': out_t,
                    'out_v': out_v
                    }
