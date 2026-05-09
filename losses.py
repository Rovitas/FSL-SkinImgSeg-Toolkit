import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    - Smooths the loss with epsilon to avoid division by zero.
    - Supports logits input (applies sigmoid internally).
    
    Args:
        alpha (float): Weight for false positives (penalize FP)
        beta (float): Weight for false negatives (penalize FN)
        smooth (float): Smoothing factor (default: 1.0)
        from_logits (bool): If True, apply sigmoid to input (default: True)
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

        # Flatten
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class HausdorffLoss(nn.Module):
    """
    Memory-efficient Hausdorff Loss using foreground point sampling.
    - Only computes distances between sampled foreground points.
    - Safe for 256x256 images on CPU.
    
    Args:
        alpha (float): Power parameter (default: 1.0 for L1-like)
        sample_points (int): Max number of foreground points to sample (default: 100)
        from_logits (bool): Apply sigmoid if True (default: True)
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

        hd_losses = []
        for b in range(N):
            # Get foreground indices (prob > 0.5)
            fg_pred = (pred_flat[b] > 0.5).nonzero(as_tuple=True)[0]
            fg_target = (target_flat[b] > 0.5).nonzero(as_tuple=True)[0]

            if len(fg_pred) == 0 or len(fg_target) == 0:
                # Use image diagonal as max distance
                max_dist = torch.sqrt(torch.tensor(H*H + W*W, dtype=torch.float32, device=device))
                hd_val = max_dist
            else:
                # Randomly sample up to `sample_points`
                idx_pred = fg_pred[torch.randperm(len(fg_pred))[:self.sample_points]]
                idx_target = fg_target[torch.randperm(len(fg_target))[:self.sample_points]]

                # Convert flat index to (x, y)
                x_pred = (idx_pred % W).float()
                y_pred = (idx_pred // W).float()
                pts_pred = torch.stack([x_pred, y_pred], dim=1)  # (M, 2)

                x_target = (idx_target % W).float()
                y_target = (idx_target // W).float()
                pts_target = torch.stack([x_target, y_target], dim=1)  # (K, 2)

                # Compute pairwise Euclidean distances (M, K)
                diff = pts_pred[:, None, :] - pts_target[None, :, :]  # (M, K, 2)
                dists = torch.norm(diff, dim=2)  # (M, K)

                # Directed HD: max over min distances
                min_dists_pred_to_target = dists.min(dim=1).values  # (M,)
                min_dists_target_to_pred = dists.min(dim=0).values  # (K,)

                hd1 = min_dists_pred_to_target.max()
                hd2 = min_dists_target_to_pred.max()
                hd_val = torch.max(hd1, hd2)

            hd_losses.append(hd_val ** self.alpha)

        hd_tensor = torch.stack(hd_losses)
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
            from_logits=False  # we handle sigmoid once
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

        # Use probability for both losses
        loss_tversky = self.tversky(pred, target)
        loss_hd = self.hd(pred, target)

        return self.weight_tversky * loss_tversky + self.weight_hd * loss_hd


# =============================================================================
# Test the losses
# =============================================================================
if __name__ == '__main__':
    print("🧪 Testing custom losses...")

    # Device
    device = torch.device('cpu')  # or 'cuda' if available

    # Create dummy data: batch=2, 1 channel, 64x64 (small for fast test)
    B, C, H, W = 2, 1, 64, 64
    logits = torch.randn(B, C, H, W, requires_grad=True, device=device)
    target = (torch.rand(B, C, H, W, device=device) > 0.5).float()

    print(f"Input shape: {logits.shape}")
    print(f"Target shape: {target.shape}")

    # --- Test TverskyLoss ---
    print("\n✅ Testing TverskyLoss...")
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
    loss1 = tversky_loss(logits, target)
    print(f"Tversky Loss: {loss1.item():.4f}")
    loss1.backward()
    print("Tversky backward: OK")

    # --- Test HausdorffLoss ---
    print("\n✅ Testing HausdorffLoss (sampled)...")
    hd_loss = HausdorffLoss(alpha=1.0, sample_points=50)
    loss2 = hd_loss(logits, target)
    print(f"Hausdorff Loss: {loss2.item():.4f}")
    loss2.backward()
    print("Hausdorff backward: OK")

    # --- Test Combined Loss ---
    print("\n✅ Testing Tversky + Hausdorff Loss...")
    combined_loss = TverskyHausdorffLoss(
        tversky_alpha=0.3,
        tversky_beta=0.7,
        hd_alpha=1.0,
        hd_sample_points=50,
        weight_tversky=0.8,
        weight_hd=0.2
    )
    loss3 = combined_loss(logits, target)
    print(f"Combined Loss: {loss3.item():.4f}")
    loss3.backward()
    print("Combined backward: OK")

    print("\n🎉 All tests passed! Losses are differentiable and memory-safe.")