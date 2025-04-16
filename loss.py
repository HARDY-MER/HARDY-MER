import torch
import torch.nn as nn
import torch.nn.functional as F


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        """
        pred -> [batch*seq_lentrain_transformer_expert_missing_softmoe.py, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        if first_stage:
            umask = umask.view(-1,1) # [batch*seq_len, 1]
            mask = umask.clone()

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1,1) # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
            loss = self.loss(pred*mask*mask_m, (target*mask*mask_m).squeeze().long()) / torch.sum(mask*mask_m)
            return loss
        else:
            assert first_stage == False
            umask = umask.view(-1, 1)  # [batch*seq_len, 1]
            mask = umask.clone()

            # l = mask.size(0)//7
            # mask[:4*l] = 0
            # mask[1*l:] = 0

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1, 1)  # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1)  # [batch*seqlen, n_classes]
            loss = self.loss(pred * mask * mask_m, (target * mask * mask_m).squeeze().long()) / torch.sum(mask * mask_m)
            if torch.isnan(loss) == True:
                loss = 0
            return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()

        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]

        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)

        return loss

class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.loss_batch = MaskedBatchReconLoss()

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        loss_batch = self.loss_batch(recon_input, target_input, input_mask, umask, adim, tdim, vdim)

        recon = recon_input  # [seqlen, batch, dim]
        target = target_input  # [seqlen, batch, dim]
        mask = input_mask  # [seqlen, batch, 3]

        recon = torch.reshape(recon, (-1, recon.size(2)))  # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2)))  # [seqlen*batch, dim]
        mask = torch.reshape(mask, (-1, mask.size(2)))  # [seqlen*batch, 3] 1(exist); 0(mask)
        umask = torch.reshape(umask, (-1, 1))  # [seqlen*batch, 1]
        _, batch_size, _ = recon_input.size()

        A_rec = recon[:, :adim]
        L_rec = recon[:, adim:adim + tdim]
        V_rec = recon[:, adim + tdim:]
        A_full = target[:, :adim]
        L_full = target[:, adim:adim + tdim]
        V_full = target[:, adim + tdim:]
        A_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 2], (-1, 1))

        loss_recon1 = self.loss(A_rec * umask, A_full * umask) * -1 * (A_miss_index - 1)
        loss_recon2 = self.loss(L_rec * umask, L_full * umask) * -1 * (L_miss_index - 1)
        loss_recon3 = self.loss(V_rec * umask, V_full * umask) * -1 * (V_miss_index - 1)

        # 获取每个样本的重构损失
        assert loss_recon1.shape[0] == loss_recon2.shape[0] == loss_recon3.shape[0] == umask.shape[0], \
            'loss_recon1.shape: {}, loss_recon2.shape: {}, loss_recon3.shape: {}, umask.shape: {}'.format(
                loss_recon1.shape, loss_recon2.shape, loss_recon3.shape, umask.shape)

        # 获取整体的重构损失
        loss_recon1 = torch.sum(loss_recon1) / adim
        loss_recon2 = torch.sum(loss_recon2) / tdim
        loss_recon3 = torch.sum(loss_recon3) / vdim
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon, loss_batch


class MaskedBatchReconLoss(nn.Module):
    def __init__(self):
        super(MaskedBatchReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim):
        """
        计算重构损失。

        参数:
            recon_input  -> [seqlen, batch, dim]
            target_input -> [seqlen, batch, dim]
            input_mask   -> [seqlen, batch, 3]  # 3 个模态
            umask        -> [batch, seqlen]     # 1(exist), 0(mask)
            adim         -> int                 # A 模态的维度
            tdim         -> int                 # T 模态的维度
            vdim         -> int                 # V 模态的维度
        返回:
            loss_recon   -> 标量张量
            loss_batch   -> [batch, seqlen] 张量
        """
        # 保持原始形状 [seqlen, batch, dim]
        recon = recon_input  # [seqlen, batch, dim]
        target = target_input  # [seqlen, batch, dim]
        mask = input_mask  # [seqlen, batch, 3]
        umask = umask  # [batch, seqlen]

        # 分割各个模态的重构和目标
        A_rec = recon[:, :, :adim]  # [seqlen, batch, adim]
        L_rec = recon[:, :, adim:adim + tdim]  # [seqlen, batch, tdim]
        V_rec = recon[:, :, adim + tdim:]  # [seqlen, batch, vdim]

        A_full = target[:, :, :adim]  # [seqlen, batch, adim]
        L_full = target[:, :, adim:adim + tdim]  # [seqlen, batch, tdim]
        V_full = target[:, :, adim + tdim:]  # [seqlen, batch, vdim]

        # 获取每个模态的掩码
        A_miss_mask = (mask[:, :, 0] == 0).float()  # [seqlen, batch]
        L_miss_mask = (mask[:, :, 1] == 0).float()  # [seqlen, batch]
        V_miss_mask = (mask[:, :, 2] == 0).float()  # [seqlen, batch]

        # 扩展 umask 以匹配重构张量的形状
        umask_expanded = umask.permute(1, 0).unsqueeze(-1)  # [seqlen, batch, 1]

        # 计算 MSE 损失，不进行缩减
        loss_recon1 = self.loss(A_rec * umask_expanded, A_full * umask_expanded)  # [seqlen, batch, adim]
        loss_recon2 = self.loss(L_rec * umask_expanded, L_full * umask_expanded)  # [seqlen, batch, tdim]
        loss_recon3 = self.loss(V_rec * umask_expanded, V_full * umask_expanded)  # [seqlen, batch, vdim]

        # 应用掩码，仅计算被屏蔽的位置的损失
        loss_recon1 = (loss_recon1 * A_miss_mask.unsqueeze(-1)).mean(dim=-1)  # [seqlen, batch]
        loss_recon2 = (loss_recon2 * L_miss_mask.unsqueeze(-1)).mean(dim=-1)  # [seqlen, batch]
        loss_recon3 = (loss_recon3 * V_miss_mask.unsqueeze(-1)).mean(dim=-1)  # [seqlen, batch]

        # 计算每个样本每个时间步的平均损失
        loss_batch = (loss_recon1 + loss_recon2 + loss_recon3) / 3  # [seqlen, batch]

        # 转换为 [batch, seqlen]
        loss_batch = loss_batch.permute(1, 0)  # [batch, seqlen]

        return loss_batch
