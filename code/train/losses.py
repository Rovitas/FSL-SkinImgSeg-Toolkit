import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    - Smooths the loss with epsilon to avoid division by zero.
    - Supports logits input (applies sigmoid internally).
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits, target):
        if self.from_logits:
            pred = torch.sigmoid(logits)
        else:
            pred = logits

        target = target.float()
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class HausdorffLoss(nn.Module):
    """
    Memory-efficient Hausdorff Loss for CPU training.
    - Iterates over batches without GPU synchronization penalties.
    - Pre-allocates tensors and hoists constants to minimize overhead.
    """
    def __init__(self, alpha=1.0, sample_points=100, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.sample_points = sample_points
        self.from_logits = from_logits

    def forward(self, logits, target):
        if self.from_logits:
            pred = torch.sigmoid(logits)
        else:
            pred = logits
        target = target.float()
        assert pred.shape == target.shape

        N, C, H, W = pred.shape
        device = pred.device

        pred_flat = pred.view(N, -1)      # (N, H*W)
        target_flat = target.view(N, -1)  # (N, H*W)

        max_dist = torch.sqrt(torch.tensor(H*H + W*W, dtype=torch.float32, device=device))
        
        hd_tensor = torch.empty(N, dtype=torch.float32, device=device)

        for b in range(N):
            fg_pred = (pred_flat[b] > 0.5).nonzero(as_tuple=True)[0]
            fg_target = (target_flat[b] > 0.5).nonzero(as_tuple=True)[0]

            if len(fg_pred) == 0 or len(fg_target) == 0:
                hd_val = max_dist
            else:
                idx_pred = fg_pred[torch.randperm(len(fg_pred))[:self.sample_points]]
                idx_target = fg_target[torch.randperm(len(fg_target))[:self.sample_points]]

                # [优化4] 运用整除切片，结构保持轻量
                x_pred = (idx_pred % W).float()
                y_pred = (idx_pred // W).float()
                pts_pred = torch.stack([x_pred, y_pred], dim=1)  # (M, 2)

                x_target = (idx_target % W).float()
                y_target = (idx_target // W).float()
                pts_target = torch.stack([x_target, y_target], dim=1)  # (K, 2)

                diff = pts_pred[:, None, :] - pts_target[None, :, :]  # (M, K, 2)
                dists = torch.norm(diff, dim=2)  # (M, K)

                min_dists_pred_to_target = dists.min(dim=1).values  # (M,)
                min_dists_target_to_pred = dists.min(dim=0).values  # (K,)

                hd1 = min_dists_pred_to_target.max()
                hd2 = min_dists_target_to_pred.max()
                hd_val = torch.max(hd1, hd2)

            hd_tensor[b] = hd_val ** self.alpha

        return hd_tensor.mean()


class TverskyHausdorffLoss(nn.Module):
    """
    Combined loss: w1 * TverskyLoss + w2 * HausdorffLoss
    """
    def __init__(
        self,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        hd_alpha=1.0,
        hd_sample_points=100,
        weight_tversky=0.8,
        weight_hd=0.2,
        from_logits=True
    ):
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            from_logits=False
        )
        self.hd = HausdorffLoss(
            alpha=hd_alpha,
            sample_points=hd_sample_points,
            from_logits=False
        )
        self.weight_tversky = weight_tversky
        self.weight_hd = weight_hd
        self.from_logits = from_logits

    def forward(self, logits, target):
        if self.from_logits:
            pred = torch.sigmoid(logits)
        else:
            pred = logits

        loss_tversky = self.tversky(pred, target)
        loss_hd = self.hd(pred, target)

        return self.weight_tversky * loss_tversky + self.weight_hd * loss_hd
